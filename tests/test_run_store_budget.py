"""The in-memory run store is bounded by bytes, and never evicts a run that is active or still needed by a job."""
import numpy as np
import pytest

from grn_sim.web import server


@pytest.fixture
def store(monkeypatch):
    saved = dict(server.SIMULATION_RUN_STORE)
    server.SIMULATION_RUN_STORE.clear()
    monkeypatch.setattr(server, "RUN_STORE_BUDGET_BYTES", 3_000_000)
    yield server.SIMULATION_RUN_STORE
    server.SIMULATION_RUN_STORE.clear()
    server.SIMULATION_RUN_STORE.update(saved)


def _entry(mb=1.0):
    return {"state": {"experiment": None}, "sim_payload": {"X": np.zeros(int(mb * 125_000))}, "inferred_payload": {}}


def test_estimate_tracks_arrays_and_lists():
    assert server._approx_size(np.zeros(1000)) == 8000
    nested = [[0.5] * 200 for _ in range(200)]
    assert 200 * 200 * 8 <= server._approx_size(nested) <= 200 * 200 * 60
    assert server._approx_size({"a": "x" * 1000}) > 1000


def test_oldest_runs_are_dropped_when_over_budget(store):
    for i in range(6):
        store[f"r{i}"] = _entry()
        server._trim_run_store()
    assert list(store) == ["r3", "r4", "r5"] or len(store) <= 3
    assert "r5" in store and "r0" not in store


def test_newest_active_and_job_pinned_runs_survive(store, monkeypatch):
    monkeypatch.setattr(server.JOB_MANAGER, "pinned_run_ids", lambda: {"r1"})
    monkeypatch.setitem(server.CURRENT_STATE, "run_id", "r0")
    for i in range(8):
        store[f"r{i}"] = _entry()
        server._trim_run_store()
    assert {"r0", "r1", "r7"} <= set(store)


def test_job_manager_pins_the_network_of_queued_dependents():
    import threading
    import time

    from grn_sim.web import jobs

    gate = threading.Event()

    def generate(params, ctx):
        if params.get("block"):
            gate.wait(10)
        return {"run_id": f"net-{params.get('seed')}"}

    m = jobs.JobManager(generate, lambda r, a, c: {}, ["s2g"])
    try:
        (gen,) = m.submit({"params": {"seed": 1}})
        end = time.time() + 10
        while time.time() < end and m.get(gen.id)["status"] != jobs.DONE:
            time.sleep(0.02)
        assert m.pinned_run_ids() == set()  # nothing queued needs it yet
        m.submit({"params": {"seed": 2, "block": True}})  # keeps the worker busy
        m.submit({"depends_on": gen.id, "algorithms": ["s2g"]})
        assert m.pinned_run_ids() == {"net-1"}
        gate.set()
    finally:
        gate.set()
        m.shutdown()
