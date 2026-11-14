import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_archive.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="scripts/ is not shipped")


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("clean_archive", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE nw (simid TEXT PRIMARY KEY, nz INTEGER, inferred_payload_json TEXT)")
    con.execute("CREATE TABLE alg (simid TEXT, algid TEXT, uid TEXT)")
    return con


def test_plan_and_apply(mod):
    con = _db()
    for simid, nz in (("a", 10), ("b", 2000)):
        con.execute("INSERT INTO nw VALUES (?, ?, '{}')", (simid, nz))
    con.executemany("INSERT INTO alg VALUES (?, ?, 'u')", [("a", "ridge"), ("b", "s2g"), ("c", "ridge"), ("c", "ridge_cv")])
    p = mod.plan(con, 1000)
    assert p["remove_runs"] == ["b"] and p["rename"] == [("a", "u", "ridge")] and p["drop_duplicate"] == [("c", "u", "ridge")]
    mod.apply(con, p)
    assert con.execute("SELECT simid, algid FROM alg ORDER BY simid, algid").fetchall() == [("a", "ridge_cv"), ("c", "ridge_cv")]
    assert con.execute("SELECT simid FROM nw").fetchall() == [("a",)]


def test_verify_reports_missing_and_passes_complete_runs(mod):
    from grn_sim.inference import registry
    full = sorted(s.key for s in registry.specs())
    con = _db()
    con.execute("INSERT INTO nw VALUES ('ok', 10, ?)", (json.dumps({"metrics": {k: {} for k in full}}),))
    con.executemany("INSERT INTO alg VALUES ('ok', ?, 'u')", [(k,) for k in full])
    con.execute("INSERT INTO nw VALUES ('short', 10, ?)", (json.dumps({"metrics": {full[0]: {}}}),))
    con.execute("INSERT INTO alg VALUES ('short', 'ridge', 'u')")
    bad = mod.verify(con)
    assert len(bad) == 1 and bad[0].startswith("short")


def test_history_entries_are_synced_to_the_response_type_column(mod):
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE nw (simid TEXT PRIMARY KEY, response_type TEXT, history_entry_json TEXT)")
    con.executemany("INSERT INTO nw VALUES (?, ?, ?)", [
        ("flat-run", "flat", json.dumps({"run_id": "flat-run", "response_type": "responsive", "keep": 1})),
        ("ok-run", "responsive", json.dumps({"response_type": "responsive"})),
        ("no-entry", "flat", "{}"),
    ])
    assert mod.response_type_mismatches(con) == [("flat-run", "flat", "responsive")]
    assert mod.sync_response_types(con) == 1
    entry = json.loads(con.execute("SELECT history_entry_json FROM nw WHERE simid = 'flat-run'").fetchone()[0])
    assert entry == {"run_id": "flat-run", "response_type": "flat", "keep": 1}            # only that field changed
    assert mod.response_type_mismatches(con) == [] and mod.sync_response_types(con) == 0


def test_inference_on_an_archived_run_keeps_its_dynamics_type(tmp_path, monkeypatch):
    """A job on an archived flat run used to write the previously open run's type ("responsive") into its history entry."""
    from grn_sim.web import database as db, server
    sid = "iorn-flat-test"
    sim = {"run_id": sid, "num_genes": 12, "gene_names": [f"G{i + 1}" for i in range(12)], "response_type": "flat",
           "true_matrix": [[0] * 12 for _ in range(12)], "X": [[0.0] * 12] * 12, "U": [[0.0] * 12] * 12}
    db.save_simulation_to_db(run_id=sid, sim_payload=sim, inferred_payload={"metrics": {}},
                             history_entry={"run_id": sid, "response_type": "flat"}, db_path=str(tmp_path / "t.sqlite"))
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    saved, store = dict(server.CURRENT_STATE), dict(server.SIMULATION_RUN_STORE)
    try:
        server.SIMULATION_RUN_STORE.pop(sid, None)
        server.CURRENT_STATE["response_type"] = "responsive"                    # the previously open run
        assert server._load_archived_run(sid) is not None
        assert server.CURRENT_STATE["response_type"] == "flat"
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)
        server.SIMULATION_RUN_STORE.clear()
        server.SIMULATION_RUN_STORE.update(store)
