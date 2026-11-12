"""scripts/compact_archive.py shrinks large runs, keeps everything the page reads, and refuses to run without a backup."""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from grn_sim.evaluation.prediction_codec import decode_prediction
from grn_sim.web import database as db

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compact_archive.py"


@pytest.fixture
def compact():
    spec = importlib.util.spec_from_file_location("compact_archive", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _archive(tmp_path):
    path = str(tmp_path / "a.sqlite")
    n = 150
    pred = (np.random.default_rng(0).random((n, n)) < 0.01).astype(int)
    inferred = {"metrics": {"s2g": {"f1_score": 0.4}}, "tmp_statmat": {"s2g": {"specificity": 99.0, "power": 30.0}},
                "dots": {"s2g": "digraph {}"}, "pred_matrices": {"s2g": pred.tolist()},
                "svgs_dot": {"s2g": "<svg>" + "x" * 1_500_000}, "svgs_neato": {"s2g": "<svg/>"}, "svgs_dot_grid": {"s2g": "<svg/>"},
                "comparison_graphs": {"s2g": {"nodes": [], "links": []}}}
    sim = {"num_genes": n, "topology": "modular", "seed": 1, "pert_text": "p", "msmt_text": "m" * 2_000_000,
           "true_matrix": [[0]], "X": [[1.0]]}
    # simulate an archive written before the compression existed: plain text columns
    db.save_simulation_to_db("big", sim, inferred, {"timestamp": "t"}, db_path=path)
    con = sqlite3.connect(path)
    con.execute("UPDATE nw SET sim_payload_json = ?, inferred_payload_json = ?, mpet = 'legacy', mmsmt = 'legacy'",
                (json.dumps(sim), json.dumps(inferred)))
    con.execute("UPDATE alg SET dmf = 'legacy dot'")
    con.commit()
    con.close()
    return path, pred


def test_dry_run_changes_nothing(compact, tmp_path, monkeypatch):
    path, _ = _archive(tmp_path)
    before = Path(path).read_bytes()
    monkeypatch.setattr(sys, "argv", ["compact_archive.py", path])
    assert compact.main() == 0
    assert Path(path).read_bytes() == before


def test_apply_needs_a_new_backup_path(compact, tmp_path, monkeypatch):
    path, _ = _archive(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compact_archive.py", path, "--apply"])
    assert compact.main() == 2
    existing = tmp_path / "exists.sqlite"
    existing.write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["compact_archive.py", path, "--apply", "--backup", str(existing)])
    assert compact.main() == 2


def test_apply_shrinks_and_keeps_what_the_page_reads(compact, tmp_path, monkeypatch):
    path, pred = _archive(tmp_path)
    backup = tmp_path / "backup.sqlite"
    monkeypatch.setattr(sys, "argv", ["compact_archive.py", path, "--apply", "--backup", str(backup)])
    assert compact.main() == 0
    assert backup.exists() and Path(path).stat().st_size < backup.stat().st_size / 5
    con = sqlite3.connect(path)
    assert con.execute("select typeof(sim_payload_json), typeof(inferred_payload_json) from nw").fetchone() == ("blob", "text")  # the stripped result payload is under the 1 MB threshold
    assert con.execute("select mpet, mmsmt from nw").fetchone() == ("", "")
    assert con.execute("select dmf from alg").fetchone()[0] == ""
    loaded = db.load_run_from_db("big", db_path=path)
    inf = loaded["inferred_payload"]
    assert not {"svgs_dot", "svgs_neato", "svgs_dot_grid"} & set(inf)
    assert np.array_equal(decode_prediction(inf["pred_matrices"]["s2g"]), pred)
    assert inf["comparison_graphs"] == {"s2g": {"nodes": [], "links": []}} and inf["dots"] == {"s2g": "digraph {}"}
    assert loaded["sim_payload"]["msmt_text"] == "m" * 2_000_000
    # running it again changes nothing
    again = tmp_path / "backup2.sqlite"
    monkeypatch.setattr(sys, "argv", ["compact_archive.py", path, "--apply", "--backup", str(again)])
    assert compact.main() == 0
