"""A job's infer stage stores each algorithm as it finishes and survives one algorithm failing."""
import pytest

from grn_sim.web import jobs, server


class _Ctx:
    def __init__(self, cancel_after=None):
        self.lines, self.calls, self.cancel_after = [], 0, cancel_after

    def check_cancel(self):
        self.calls += 1
        if self.cancel_after is not None and self.calls > self.cancel_after:
            raise jobs.JobCancelled()

    def progress(self, *a, **k):
        pass

    def log(self, line):
        self.lines.append(line)


@pytest.fixture
def fake_run(monkeypatch):
    run_id = "test-job-infer-run"
    server.SIMULATION_RUN_STORE[run_id] = {"state": {"run_id": run_id}, "inferred_payload": {}}
    seen = []

    def fake_execute(algs, on_algorithm=None, previous=None):
        (alg,) = algs  # exactly one algorithm per call
        seen.append((alg, sorted((previous or {}).get("metrics", {}))))
        if alg == "boom":
            raise ValueError("kaboom")
        payload = dict(previous or {})
        payload["metrics"] = {**payload.get("metrics", {}), alg: {"f1_score": 0.5, "total_errors": 3}}
        server.SIMULATION_RUN_STORE[run_id]["inferred_payload"] = payload
        return payload

    monkeypatch.setattr(server, "_execute_inference", fake_execute)
    yield run_id, seen
    server.SIMULATION_RUN_STORE.pop(run_id, None)


def test_results_accumulate_and_a_failure_is_skipped(fake_run):
    run_id, seen = fake_run
    out = server._job_infer(run_id, ["a", "boom", "b"], _Ctx())
    assert [s[0] for s in seen] == ["a", "boom", "b"]
    assert seen[2][1] == ["a"]  # each call sees the earlier stored results
    assert out["algorithms_stored"] == ["a", "b"]
    assert out["algorithms_failed"] == {"boom": "ValueError: kaboom"}


def test_job_fails_only_when_every_algorithm_failed(fake_run):
    run_id, _ = fake_run
    with pytest.raises(RuntimeError, match="boom"):
        server._job_infer(run_id, ["boom"], _Ctx())


def test_cancel_keeps_finished_results(fake_run):
    run_id, _ = fake_run
    with pytest.raises(jobs.JobCancelled):
        server._job_infer(run_id, ["a", "b", "c"], _Ctx(cancel_after=1))
    assert list(server.SIMULATION_RUN_STORE[run_id]["inferred_payload"]["metrics"]) == ["a"]
