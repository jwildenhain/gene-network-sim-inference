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
