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
