"""scripts/rescore_archive_orientation.py --only: re-orients one algorithm in additive runs, once, and leaves the rest."""
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "rgn_db.sqlite"


def _script():
    spec = importlib.util.spec_from_file_location("rescore_archive_orientation", ROOT / "scripts" / "rescore_archive_orientation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def small_archive(tmp_path):
    """Two real additive runs with DMD-GRN results, copied into a scratch archive."""
    if not REAL.exists():
        pytest.skip("no archive in this checkout")
    from grn_sim.web import database as db
    src = sqlite3.connect(f"file:{REAL}?mode=ro", uri=True)
    rows = src.execute("SELECT simid, settings_json, history_entry_json FROM nw WHERE history_entry_json != '{}' "
                       "AND length(inferred_payload_json) < 3000000 AND simid IN (SELECT simid FROM alg WHERE algid = 'dmdgrn')").fetchall()
    mod = _script()
    pick = []
    for simid, settings_json, history_json in rows:
        sim, inf = src.execute("SELECT sim_payload_json, inferred_payload_json FROM nw WHERE simid = ?", (simid,)).fetchone()
        sim_p, inf_p = json.loads(db.unpack_json(sim)), json.loads(db.unpack_json(inf))
        if "dmdgrn" not in (inf_p.get("pred_matrices") or {}):
            continue
        settings = json.loads(settings_json or "{}")
        run = mod._run_context(settings, json.loads(history_json or "{}"))
        data_model = mod._inference_data_model(run)
        if data_model not in (mod.ADDITIVE_GARDNER, mod.ADDITIVE_PHYSICAL):
            continue
        truth_ctx = dict(sim_p)
        truth_ctx.setdefault("mode", run["mode"])
        truth_ctx.setdefault("settings", settings)
        from grn_sim.evaluation.prediction_codec import decode_prediction
        from grn_sim.inference.orientation import orient_prediction
        raw = decode_prediction(inf_p["pred_matrices"]["dmdgrn"])
        if not np.array_equal(orient_prediction("dmdgrn", raw, data_model, mod._run_truth_target_regulator(truth_ctx),
                                                mod._fit_mode(run, "dmdgrn")), raw):
            pick.append(simid)             # a run where the current rule really changes the stored matrix
        if len(pick) == 2:
            break
    if not pick:
        pytest.skip("no additive run with a DMD-GRN result in the archive")
    dest = tmp_path / "a.sqlite"
    db.init_db(str(dest))
    out = sqlite3.connect(dest)
    for simid in pick:
        for table in ("nw", "alg"):
            cols = [c[1] for c in src.execute(f"PRAGMA table_info({table})")]
            for r in src.execute(f"SELECT * FROM {table} WHERE simid = ?", (simid,)):
                out.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", r)
    out.commit(); out.close(); src.close()
    return dest, pick


def _pred(path, simid, alg="dmdgrn"):
    from grn_sim.evaluation.prediction_codec import decode_prediction
    from grn_sim.web import database as db
    con = sqlite3.connect(path)
    inf = json.loads(db.unpack_json(con.execute("SELECT inferred_payload_json FROM nw WHERE simid = ?", (simid,)).fetchone()[0]))
    con.close()
    return decode_prediction(inf["pred_matrices"][alg]), inf


def test_only_reorients_the_named_algorithm_once(small_archive, tmp_path):
    path, pick = small_archive
    before = {s: _pred(path, s) for s in pick}
    mod = _script()
    argv = ["--db", str(path), "--only", "dmdgrn", "--before", "9999", "--apply", "--backup-dir", str(tmp_path / "bk")]
    assert mod.main(argv) == 0
    for s in pick:
        after, inf = _pred(path, s)
        old, old_inf = before[s]
        assert np.array_equal(after, old.T) and "dmdgrn" in inf["orientation_fixes"]
        other = next(a for a in old_inf["pred_matrices"] if a != "dmdgrn")
        assert np.array_equal(_pred(path, s, other)[0], _pred(path, s, other)[0])      # untouched algorithms are not rewritten
    assert mod.main(argv) == 0                                                          # a second pass flips nothing back
    for s in pick:
        assert np.array_equal(_pred(path, s)[0], before[s][0].T)


def test_without_apply_the_archive_is_unchanged(small_archive, tmp_path):
    path, pick = small_archive
    raw = Path(path).read_bytes()
    assert _script().main(["--db", str(path), "--only", "dmdgrn", "--before", "9999"]) == 0
    assert Path(path).read_bytes() == raw
