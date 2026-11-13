"""Fork-based process pool for per-target-gene loops (one BLAS thread per worker, results returned in index order)."""

import faulthandler
import io
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from grn_sim.inference._native import NATIVE_LOCK

_STATE: dict = {}


def default_n_jobs(cap: int = 6) -> int:
    """Half the available CPUs, at most ``cap``."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    return max(1, min(cap, cpus // 2))


# A pool whose workers burn (almost) no CPU for this long while chunks are still pending is deadlocked: observed once
# at 500 genes (a forked G2S worker sat in futex_wait for an hour and the whole job queue waited behind it). Real
# chunks use CPU continuously, so the guard cannot fire on slow-but-working workers.
STALL_IDLE_SECONDS = float(os.environ.get("GRN_SIM_POOL_STALL_SECONDS", "300"))
_BUSY_TICKS = 50                       # clock ticks (0.5 s at 100 Hz) of CPU across all workers that count as progress


EXIT_GRACE_SECONDS = 30.0   # how long the parent waits for a finished worker to exit before killing it


def reset_std_streams() -> None:
    """
    Worker side: give the child fresh stdout / stderr objects.

    A thread of the parent (a request handler writing its log line) can hold a stream's buffer lock at the instant of the
    fork; the child inherits that lock held, and ``multiprocessing`` flushes both streams when the worker exits, so the
    worker hangs for ever after finishing its work and the parent waits for it to exit (seen at 500 genes: a G2S job
    "running" for an hour, then reproduced by polling the API while G2S ran; the workers' own stack dumps ended in
    ``util._flush_std_streams``). The inherited objects are replaced without being flushed or touched.
    """
    for name, fd in (("stdout", 1), ("stderr", 2)):
        try:
            encoding = getattr(getattr(sys, name), "encoding", None) or "utf-8"
            setattr(sys, name, io.TextIOWrapper(io.FileIO(fd, "w", closefd=False), encoding=encoding,
                                                errors="replace", line_buffering=True))
        except Exception:  # pragma: no cover - a stream without a descriptor (captured output)
            pass


def prepare_worker() -> None:
    """First thing every forked worker does (pool initializers)."""
    reset_std_streams()
    arm_stall_trace()


def arm_stall_trace() -> None:
    """Worker side: write every thread's Python stack to a file if this worker is still running after the stall time."""
    try:
        path = os.path.join(tempfile.gettempdir(), f"grn_sim_worker_{os.getpid()}.trace")
        handle = open(path, "w")
        faulthandler.dump_traceback_later(STALL_IDLE_SECONDS * 2, repeat=False, file=handle)
    except Exception:  # pragma: no cover - diagnostics only
        pass


def _cpu_ticks(pids: Sequence[int]) -> Optional[int]:
    """User + system clock ticks used so far by ``pids`` (Linux ``/proc``); None where that cannot be read."""
    total = 0
    try:
        for pid in pids:
            with open(f"/proc/{pid}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
            total += int(fields[11]) + int(fields[12])
    except (OSError, IndexError, ValueError):
        return None
    return total


def run_chunks(fn: Callable[[Any], Any], chunks: Sequence[Any], n_jobs: int,
               initializer: Optional[Callable[..., None]] = None, initargs: Tuple[Any, ...] = (),
               stall_seconds: Optional[float] = None) -> Tuple[Dict[int, Any], bool]:
    """Runs ``fn(chunk)`` for every chunk on a fork pool; returns ``({chunk index: result}, stalled)``.

    When the workers make no progress (no chunk finished and no CPU used) for ``stall_seconds`` the pool is killed and
    the finished chunks are returned with ``stalled=True``: the caller computes the missing ones itself. Exceptions
    raised by ``fn`` propagate as usual.
    """
    stall = STALL_IDLE_SECONDS if stall_seconds is None else stall_seconds
    results: Dict[int, Any] = {}
    stalled = False
    ex = ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork"), initializer=initializer,
                             initargs=initargs)
    try:
        futures = {ex.submit(fn, chunk): k for k, chunk in enumerate(chunks)}
        pending = set(futures)
        ref_time, ref_ticks = time.monotonic(), None
        while pending:
            done, pending = wait(pending, timeout=min(10.0, max(0.05, stall / 4)), return_when=FIRST_COMPLETED)
            for fut in done:
                results[futures[fut]] = fut.result()
            now = time.monotonic()
            ticks = _cpu_ticks([p.pid for p in list(getattr(ex, "_processes", {}).values())])
            if done or ticks is None or ref_ticks is None or ticks - ref_ticks > _BUSY_TICKS:
                ref_time, ref_ticks = now, ticks
            elif pending and now - ref_time >= stall:
                stalled = True
                print(f"[grn_sim] worker pool made no progress for {stall:.0f} s; killing it and finishing "
                      f"{len(pending)} chunk(s) in this process", file=sys.stderr, flush=True)
                break
    finally:
        if stalled:
            for proc in list(getattr(ex, "_processes", {}).values()):
                try:
                    proc.kill()
                except Exception:  # pragma: no cover
                    pass
        procs = list(getattr(ex, "_processes", None).values()) if getattr(ex, "_processes", None) else []
        ex.shutdown(wait=False, cancel_futures=True)
        if not stalled:
            # workers that have finished should exit at once; one that does not (a stuck exit flush) is killed, not waited for
            deadline = time.monotonic() + EXIT_GRACE_SECONDS
            for proc in procs:
                while proc.is_alive() and time.monotonic() < deadline:   # join() can return while the child is still exiting
                    time.sleep(0.02)
                if proc.is_alive():
                    print(f"[grn_sim] worker {proc.pid} did not exit within {EXIT_GRACE_SECONDS:.0f} s; killing it",
                          file=sys.stderr, flush=True)
                    try:
                        proc.kill()
                    except Exception:  # pragma: no cover
                        pass
    return results, stalled


def _init() -> None:
    prepare_worker()
    try:
        from threadpoolctl import threadpool_limits
        _STATE["limits"] = threadpool_limits(limits=1, user_api="blas")
    except Exception:  # pragma: no cover - threadpoolctl missing
        pass


def _run_chunk(indices: List[int]):
    fn, payload = _STATE["job"]
    return [(i, fn(payload, i)) for i in indices]


def map_indices(fn: Callable[[Any, int], Any], payload: Any, n: int, n_jobs: int,
                min_parallel: int = 32) -> List[Any]:
    """``[fn(payload, i) for i in range(n)]``, on a fork pool of ``n_jobs`` workers when worthwhile."""
    n_jobs = max(1, min(int(n_jobs), n))
    if n_jobs <= 1 or n < min_parallel or "fork" not in mp.get_all_start_methods():
        return [fn(payload, i) for i in range(n)]
    _STATE["job"] = (fn, payload)          # inherited by the forked workers, nothing is pickled
    try:
        n_chunks = min(n, n_jobs * 8)
        chunks = [list(range(n))[c::n_chunks] for c in range(n_chunks)]
        with NATIVE_LOCK:             # no other thread loads native libraries while this one forks (see _native.py)
            done, _stalled = run_chunks(_run_chunk, chunks, n_jobs, initializer=_init)
        out = [r for k in sorted(done) for r in done[k]]
        for k in range(len(chunks)):  # chunks a stalled pool did not finish
            if k not in done:
                out.extend((i, fn(payload, i)) for i in chunks[k])
    finally:
        _STATE.pop("job", None)
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]
