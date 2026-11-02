"""
Background job queue for the web studio.

A job is a small pipeline of up to two stages, run one at a time (FIFO) by a single worker thread:

* ``generate`` - simulate a synthetic network / perturbation data set from a parameter dict,
* ``infer``    - run the selected reverse-engineering algorithms on a network.

The infer stage targets the network the job just generated, a network a *previous* job generated (``depends_on``,
which may still be queued - "generate now, infer afterwards"), or an archived run (``run_id``).

The manager knows nothing about the simulator: the two stages are injected callables, which keeps it testable
without running a simulation. Progress is counted in steps (one for the generation, one per algorithm) and the
infer stage checks for cancellation between algorithms. The networks and inference results jobs create are archived
like any other run. With a ``store_path`` the job list itself survives a restart too: the active jobs and the last
``keep_finished`` finished ones are saved after every change, and jobs that were queued or running when the server
stopped come back as ``failed`` ("interrupted") - their networks, if they got that far, are in the archive.
"""

from __future__ import annotations

import collections
import datetime
import itertools
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)
FINISHED = (DONE, FAILED, CANCELLED)

MAX_BATCH = 50          # networks one submit may queue
MAX_LOG_LINES = 200     # per job


class JobCancelled(Exception):
    """Raised inside a stage (via ``JobContext.check_cancel``) to stop the job."""


class QueueFull(Exception):
    """The queue holds the maximum number of active jobs."""


def _stamp(ts: Optional[float]) -> Optional[str]:
    return None if ts is None else datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


class Job:
    def __init__(self, job_id: str, label: str, params: Optional[Dict[str, Any]], run_id: Optional[str],
                 depends_on: Optional[str], algorithms: List[str]):
        self.id = job_id
        self.label = label
        self.params = params
        self.run_id = run_id
        self.depends_on = depends_on
        self.algorithms = algorithms
        self.status = QUEUED
        self.stage: Optional[str] = None
        self.steps_total = (1 if params is not None else 0) + len(algorithms)
        self.steps_done = 0
        self.current: Optional[str] = None
        self.error: Optional[str] = None
        self.summary: Dict[str, Any] = {}
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.cancel_requested = False
        self.log: Deque[str] = collections.deque(maxlen=MAX_LOG_LINES)

    @property
    def kind(self) -> str:
        if self.params is not None:
            return "generate+infer" if self.algorithms else "generate"
        return "infer"

    def to_record(self) -> Dict[str, Any]:
        """Everything needed to show this job again after a restart."""
        return {
            "id": self.id, "label": self.label, "params": self.params, "run_id": self.run_id,
            "depends_on": self.depends_on, "algorithms": self.algorithms, "status": self.status, "stage": self.stage,
            "steps_total": self.steps_total, "steps_done": self.steps_done, "error": self.error,
            "summary": self.summary, "created": self.created, "started": self.started, "finished": self.finished,
            "log": list(self.log),
        }

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "Job":
        job = cls(str(rec["id"]), str(rec.get("label") or rec["id"]), rec.get("params"), rec.get("run_id"),
                  rec.get("depends_on"), list(rec.get("algorithms") or []))
        job.status = rec.get("status", FAILED)
        job.stage = rec.get("stage")
        job.steps_total = int(rec.get("steps_total", job.steps_total))
        job.steps_done = int(rec.get("steps_done", 0))
        job.error = rec.get("error")
        job.summary = dict(rec.get("summary") or {})
        job.created = float(rec.get("created") or time.time())
        job.started = rec.get("started")
        job.finished = rec.get("finished")
        job.log.extend(rec.get("log") or [])
        return job

    def to_dict(self, with_log: bool = False) -> Dict[str, Any]:
        end = self.finished or time.time()
        out: Dict[str, Any] = {
            "id": self.id, "label": self.label, "kind": self.kind, "status": self.status, "stage": self.stage,
            "run_id": self.run_id, "depends_on": self.depends_on, "algorithms": list(self.algorithms),
            "steps_done": self.steps_done, "steps_total": self.steps_total, "current": self.current,
            "progress_pct": 100.0 if self.status == DONE else (
                round(100.0 * self.steps_done / self.steps_total, 1) if self.steps_total else 0.0),
            "error": self.error, "summary": self.summary, "cancel_requested": self.cancel_requested,
            "created": _stamp(self.created), "started": _stamp(self.started), "finished": _stamp(self.finished),
            "elapsed_s": round(end - self.started, 1) if self.started else 0.0,
            "queued_s": round((self.started or time.time()) - self.created, 1),
            "params": self.params,
        }
        if with_log:
            out["log"] = list(self.log)
        return out


class JobContext:
    """What a stage may do to its job: report progress, log, and honour a cancel request."""

    def __init__(self, manager: "JobManager", job: Job):
        self._manager = manager
        self._job = job

    def log(self, message: str) -> None:
        with self._manager._lock:
            self._job.log.append(f"{datetime.datetime.now():%H:%M:%S}  {message}")

    def check_cancel(self) -> None:
        if self._job.cancel_requested:
            raise JobCancelled()

    def progress(self, label: Optional[str] = None, advance: bool = False) -> None:
        with self._manager._lock:
            if label is not None:
                self._job.current = label
            if advance:
                self._job.steps_done = min(self._job.steps_total, self._job.steps_done + 1)


GenerateFn = Callable[[Dict[str, Any], JobContext], Dict[str, Any]]
InferFn = Callable[[str, List[str], JobContext], Dict[str, Any]]


class JobManager:
    def __init__(self, generate: GenerateFn, infer: InferFn, known_algorithms: Iterable[str],
                 validate_params: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 can_infer: Optional[Callable[[str], bool]] = None,
                 max_active: int = 200, keep_finished: int = 100, store_path: Optional[Path] = None):
        self._generate = generate
        self._infer = infer
        self._known = tuple(known_algorithms)
        self._validate_params = validate_params
        self._can_infer = can_infer
        self._max_active = max_active
        self._keep_finished = keep_finished
        self._jobs: Dict[str, Job] = {}
        self._queue: Deque[str] = collections.deque()
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._worker: Optional[threading.Thread] = None
        self._stopping = False
        self._store_path = Path(store_path) if store_path else None
        self._load()

    # ---- persistence ------------------------------------------------------------------------------------------
    def _load(self) -> None:
        """Restores the saved job list. Never raises: a missing or damaged file just means an empty list."""
        if self._store_path is None or not self._store_path.is_file():
            return
        try:
            records = json.loads(self._store_path.read_text(encoding="utf-8")).get("jobs", [])
            restored = [Job.from_record(r) for r in records]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"[jobs] ignoring unreadable job list {self._store_path}: {exc}", file=sys.stderr)
            return
        highest = 0
        for job in restored:
            if job.status in ACTIVE:  # the process that was running it is gone
                was = job.status
                job.status, job.finished, job.stage, job.current = FAILED, time.time(), None, None
                job.error = f"interrupted: the server restarted while this job was {was}" + (
                    f"; its network {job.run_id} is in the archive" if job.run_id else "")
                job.log.append(f"{datetime.datetime.now():%H:%M:%S}  {job.error}")
            self._jobs[job.id] = job
            digits = "".join(c for c in job.id if c.isdigit())
            highest = max(highest, int(digits) if digits else 0)
        self._ids = itertools.count(highest + 1)
        self._trim()

    def _save(self) -> None:
        """Writes the job list atomically. Call with the lock held. A failing disk must not break the queue."""
        if self._store_path is None:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._store_path.with_name(self._store_path.name + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "jobs": [j.to_record() for j in self._jobs.values()]}),
                           encoding="utf-8")
            os.replace(tmp, self._store_path)
        except OSError as exc:
            print(f"[jobs] could not save the job list to {self._store_path}: {exc}", file=sys.stderr)

    def _trim(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.status in FINISHED), key=lambda j: j.finished or 0)
        for old in finished[:max(0, len(finished) - self._keep_finished)]:
            del self._jobs[old.id]

    # ---- submitting -------------------------------------------------------------------------------------------
    def submit(self, spec: Dict[str, Any]) -> List[Job]:
        """Validate ``spec`` and queue its job(s). Raises ValueError (bad request) or QueueFull."""
        if not isinstance(spec, dict):
            raise ValueError("a job request must be a JSON object")
        algs = self._clean_algorithms(spec.get("algorithms"))
        params = spec.get("params")
        run_id = spec.get("run_id")
        depends_on = spec.get("depends_on")
        label = str(spec.get("label") or "").strip()[:80]
        try:
            count = int(spec.get("count", 1))
        except (TypeError, ValueError):
            raise ValueError("count must be an integer") from None
        if not 1 <= count <= MAX_BATCH:
            raise ValueError(f"count must be between 1 and {MAX_BATCH}")

        if sum(x is not None for x in (params, run_id, depends_on)) != 1:
            raise ValueError("give exactly one of 'params' (generate a network), 'depends_on' (a queued job's network) "
                             "or 'run_id' (an existing network)")
        if params is not None:
            if not isinstance(params, dict):
                raise ValueError("params must be an object")
            clean = dict(self._validate_params(params) if self._validate_params else params)
            clean["auto_infer"] = False  # inference is its own stage
        else:
            clean = None
            if count != 1:
                raise ValueError("count only applies to jobs that generate networks")
            if not algs:
                raise ValueError("an inference job needs at least one algorithm")
        if run_id is not None:
            run_id = str(run_id)
            if self._can_infer is not None and not self._can_infer(run_id):
                raise ValueError(f"network {run_id!r} is not in memory (generated before the last server restart?); "
                                 "queue a new generation instead")

        with self._lock:
            if depends_on is not None:
                dep = self._jobs.get(str(depends_on))
                if dep is None:
                    raise ValueError(f"unknown job {depends_on!r} to depend on")
                if dep.params is None and dep.depends_on is None and dep.run_id is None:
                    raise ValueError(f"job {depends_on!r} does not produce a network")
                depends_on = dep.id
            if sum(j.status in ACTIVE for j in self._jobs.values()) + count > self._max_active:
                raise QueueFull(f"the queue is full ({self._max_active} active jobs)")
            jobs: List[Job] = []
            for i in range(count):
                p = None
                if clean is not None:
                    p = dict(clean)
                    if count > 1 and p.get("seed") is not None:
                        p["seed"] = int(p["seed"]) + i  # a batch is a set of different networks
                job = Job(f"job-{next(self._ids):04d}", label or self._default_label(p, run_id, depends_on, algs),
                          p, run_id, depends_on, list(algs))
                job.log.append(f"{datetime.datetime.now():%H:%M:%S}  queued ({job.kind})")
                self._jobs[job.id] = job
                self._queue.append(job.id)
                jobs.append(job)
            self._save()
            self._ensure_worker()
            self._wake.notify()
            return jobs

    def _clean_algorithms(self, algs: Any) -> List[str]:
        if algs is None:
            return []
        if isinstance(algs, str):
            algs = [algs]
        if not isinstance(algs, Sequence):
            raise ValueError("algorithms must be a list of algorithm keys")
        seen: List[str] = []
        for a in algs:
            if a not in self._known:
                raise ValueError(f"unknown algorithm {a!r}")
            if a not in seen:
                seen.append(a)
        return seen

    @staticmethod
    def _default_label(params: Optional[Dict[str, Any]], run_id: Optional[str], depends_on: Optional[str],
                       algs: List[str]) -> str:
        if params is not None:
            n = params.get("num_genes", "?")
            return f"{params.get('topology', 'network')} n={n} seed={params.get('seed', '-')}"
        return f"infer on {run_id or 'output of ' + str(depends_on)} ({len(algs)} alg)"

    # ---- inspecting / controlling -----------------------------------------------------------------------------
    def _describe(self, job: Job, with_log: bool = False) -> Dict[str, Any]:
        out = job.to_dict(with_log=with_log)
        # can more algorithms still be run on this job's network? (it must still be in memory)
        run_id = job.run_id
        out["infer_ok"] = run_id is not None and (self._can_infer is None or bool(self._can_infer(run_id)))
        return out

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._describe(j) for j in sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)]

    def get(self, job_id: str, with_log: bool = True) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._describe(job, with_log) if job else None

    def running(self) -> Optional[Dict[str, Any]]:
        """The job being worked on now, if any."""
        with self._lock:
            return next((j.to_dict() for j in self._jobs.values() if j.status == RUNNING), None)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            c = {s: 0 for s in (QUEUED, RUNNING, DONE, FAILED, CANCELLED)}
            for j in self._jobs.values():
                c[j.status] += 1
            return c

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status == QUEUED:
                self._finish(job, CANCELLED, "cancelled before it started")
            elif job.status == RUNNING:
                job.cancel_requested = True
                job.log.append(f"{datetime.datetime.now():%H:%M:%S}  cancel requested")
            return job.to_dict(with_log=True)

    def clear_finished(self) -> int:
        with self._lock:
            gone = [i for i, j in self._jobs.items() if j.status in FINISHED]
            for i in gone:
                del self._jobs[i]
            self._save()
            return len(gone)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            self._stopping = True
            self._wake.notify_all()
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    # ---- worker -----------------------------------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._stopping = False
            self._worker = threading.Thread(target=self._loop, name="grn-sim-jobs", daemon=True)
            self._worker.start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stopping:
                    self._wake.wait()
                if self._stopping:
                    return
                job = self._jobs.get(self._queue.popleft())
                if job is None or job.status != QUEUED:
                    continue  # cancelled while waiting
                job.status = RUNNING
                job.started = time.time()
                job.log.append(f"{datetime.datetime.now():%H:%M:%S}  started")
                self._save()
            self._run(job)

    def _run(self, job: Job) -> None:
        ctx = JobContext(self, job)
        try:
            if job.params is not None:
                self._stage_generate(job, ctx)
            elif job.depends_on is not None:
                self._adopt_dependency(job, ctx)
            if job.algorithms:
                self._stage_infer(job, ctx)
        except JobCancelled:
            with self._lock:
                note = "cancelled"
                if job.run_id and job.params is not None:
                    note += f" (network {job.run_id} was kept)"
                self._finish(job, CANCELLED, note)
            return
        except Exception as exc:  # a failing job must never kill the worker
            traceback.print_exc()
            with self._lock:
                self._finish(job, FAILED, f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            self._finish(job, DONE, None)

    def _stage_generate(self, job: Job, ctx: JobContext) -> None:
        assert job.params is not None
        job.stage = "generate"
        ctx.progress("generating network")
        ctx.log(f"generating network ({job.params.get('num_genes', '?')} genes, {job.params.get('topology', '?')})")
        result = self._generate(job.params, ctx)
        with self._lock:
            job.run_id = result["run_id"]
            job.summary.update({k: v for k, v in result.items() if k != "run_id"})
        ctx.progress(advance=True)
        ctx.log(f"network {job.run_id} generated")
        ctx.check_cancel()

    def _adopt_dependency(self, job: Job, ctx: JobContext) -> None:
        with self._lock:
            dep = self._jobs.get(job.depends_on or "")
        if dep is None:
            raise RuntimeError(f"job {job.depends_on} no longer exists")
        if dep.status != DONE or not dep.run_id:
            if dep.status == CANCELLED and dep.run_id:
                pass  # it was cancelled after its network existed: that network is still usable
            else:
                raise RuntimeError(f"job {dep.id} {dep.status}" + (f": {dep.error}" if dep.error else ""))
        job.run_id = dep.run_id
        ctx.log(f"using the network {dep.run_id} from {dep.id}")

    def _stage_infer(self, job: Job, ctx: JobContext) -> None:
        assert job.run_id
        job.stage = "infer"
        ctx.log(f"inferring on {job.run_id} with {', '.join(job.algorithms)}")
        result = self._infer(job.run_id, list(job.algorithms), ctx)
        with self._lock:
            job.summary.update(result)

    def _finish(self, job: Job, status: str, error: Optional[str]) -> None:
        job.status = status
        job.stage = None if status != RUNNING else job.stage
        job.current = None
        job.finished = time.time()
        if status != DONE:
            job.error = error
        detail = error if status == CANCELLED else (f"{status}: {error}" if error else status)
        job.log.append(f"{datetime.datetime.now():%H:%M:%S}  {detail or status}")
        self._trim()
        self._save()
