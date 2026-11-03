"""Fork-based process pool for per-target-gene loops (one BLAS thread per worker, results returned in index order)."""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional

from grn_sim.inference._native import NATIVE_LOCK

_STATE: dict = {}


def default_n_jobs(cap: int = 6) -> int:
    """Half the available CPUs, at most ``cap``."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count() or 1
    return max(1, min(cap, cpus // 2))


def _init() -> None:
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
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork"), initializer=_init) as pool:
                out = [r for part in pool.map(_run_chunk, chunks) for r in part]
    finally:
        _STATE.pop("job", None)
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]
