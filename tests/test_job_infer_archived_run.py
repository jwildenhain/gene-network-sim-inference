"""A job can add algorithms to a network generated before the server restarted (loaded back from the archive)."""
import pytest

from grn_sim.web import database as db
from grn_sim.web import server


class _Ctx:
    def check_cancel(self): pass
    def progress(self, *a, **k): pass
    def log(self, line): pass


@pytest.fixture
def archived_run(tmp_path, monkeypatch):
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    saved_state, saved_store = dict(server.CURRENT_STATE), dict(server.SIMULATION_RUN_STORE)
    server._execute_simulation({"num_genes": 12, "topology": "random", "k_av": 1.8, "seed": 11})
    run_id = server.CURRENT_STATE["run_id"]
    server._execute_inference(["s2g"])
    server.SIMULATION_RUN_STORE.clear()  # what a server restart does
    yield run_id, str(tmp_path / "t.sqlite")
    server.CURRENT_STATE.clear(); server.CURRENT_STATE.update(saved_state)
    server.SIMULATION_RUN_STORE.clear(); server.SIMULATION_RUN_STORE.update(saved_store)


def test_job_can_infer_knows_archived_runs(archived_run):
    run_id, _ = archived_run
    assert server._job_can_infer(run_id) is True
    assert server._job_can_infer("no-such-run") is False


def test_job_adds_algorithms_to_an_archived_run_and_keeps_the_old_ones(archived_run):
    run_id, path = archived_run
    before = server.CURRENT_STATE.get("run_id")
    out = server._job_infer(run_id, ["by_chance", "mean_difference"], _Ctx())
    assert set(out["algorithms_stored"]) == {"s2g", "by_chance", "mean_difference"}
    assert out["algorithms_failed"] == {}
    assert server.CURRENT_STATE.get("run_id") == before  # the studio's network was put back
    stored = db.load_run_from_db(run_id, db_path=path)["inferred_payload"]
    assert set(stored["metrics"]) == {"s2g", "by_chance", "mean_difference"}


def test_unknown_run_is_reported(archived_run):
    with pytest.raises(LookupError):
        server._job_infer("no-such-run", ["by_chance"], _Ctx())


def test_adopting_runs_one_after_another_respects_the_memory_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    saved_state, saved_store = dict(server.CURRENT_STATE), dict(server.SIMULATION_RUN_STORE)
    try:
        ids = []
        for seed in (21, 22, 23):
            server._execute_simulation({"num_genes": 12, "topology": "random", "k_av": 1.8, "seed": seed})
            ids.append(server.CURRENT_STATE["run_id"])
        server.SIMULATION_RUN_STORE.clear()
        monkeypatch.setattr(server, "RUN_STORE_BUDGET_BYTES", 1)  # everything is over budget: only the newest may stay
        for rid in ids:
            server._job_infer(rid, ["by_chance"], _Ctx())
        assert list(server.SIMULATION_RUN_STORE) == [ids[-1]]
    finally:
        server.CURRENT_STATE.clear(); server.CURRENT_STATE.update(saved_state)
        server.SIMULATION_RUN_STORE.clear(); server.SIMULATION_RUN_STORE.update(saved_store)


def test_job_on_a_run_that_is_in_memory_without_its_data(archived_run):
    """The start-up run and runs opened from the archive are in the store without an experiment."""
    run_id, _ = archived_run
    server.CURRENT_STATE["experiment"] = None
    server._load_archived_run(run_id)  # what the server does at start-up
    assert server.SIMULATION_RUN_STORE[run_id]["state"].get("experiment") is None
    out = server._job_infer(run_id, ["by_chance"], _Ctx())
    assert out["algorithms_failed"] == {} and "by_chance" in out["algorithms_stored"]


def test_a_run_adopted_while_another_network_is_open_uses_its_own_data(archived_run):
    """The loader copies CURRENT_STATE, so a job must not pick up the experiment of the network the studio has open."""
    import numpy as np
    run_id, _ = archived_run
    own_X = np.array(server._load_archived_run(run_id)["X"], dtype=float)
    server.SIMULATION_RUN_STORE.clear()
    server._execute_simulation({"num_genes": 12, "topology": "random", "k_av": 1.8, "seed": 99})  # another network is open
    other = server.CURRENT_STATE["experiment"]
    assert not np.allclose(np.asarray(other.X), own_X)
    stored = server._adopt_archived_run(run_id)
    assert np.allclose(stored["state"]["experiment"].X, own_X)
