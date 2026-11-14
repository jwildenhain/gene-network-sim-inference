"""docs/SCALE_BENCHMARK.md's table is rendered from the committed snapshot of the archive; the snapshot is sane."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_scale_benchmark.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists() or not (ROOT / "docs").exists(), reason="docs/ and scripts/ are not shipped")


def _mod():
    spec = importlib.util.spec_from_file_location("render_scale_benchmark", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_page_matches_the_snapshot():
    m = _mod()
    snap = json.loads(m.SNAPSHOT.read_text())
    assert m.render_page(snap) == m.PAGE.read_text(), "run scripts/render_scale_benchmark.py render"


def test_snapshot_covers_three_sizes_and_values_are_in_range():
    m = _mod()
    snap = json.loads(m.SNAPSHOT.read_text())
    assert set(snap["runs"]) == {"500", "1000", "2000"}
    algs = snap["algorithms"]
    for key in ("s2g", "g2s", "nir", "ridge_cv", "sfinn", "vote", "score_vote", "grnboost"):
        assert {"500", "1000", "2000"} <= set(algs[key]), key
    for alg, d in algs.items():
        for size, v in d.items():
            assert 0.0 <= v["f1"] <= 1.0 and 0.0 <= v["aupr"] <= 1.0, (alg, size)


def test_controls_are_left_out_of_the_table():
    m = _mod()
    table = m.render_table(json.loads(m.SNAPSHOT.read_text()))
    assert "by_chance" not in table.split("Cells are")[0] and "gene_features" not in table.split("Cells are")[0]


@pytest.mark.skipif(os.environ.get("GRN_SIM_ARCHIVE_CHECK") != "1", reason="set GRN_SIM_ARCHIVE_CHECK=1 to compare with data/rgn_db.sqlite (read-only)")
def test_snapshot_equals_the_archive():
    m = _mod()
    db = ROOT / "data" / "rgn_db.sqlite"
    assert json.loads(m.SNAPSHOT.read_text()) == json.loads(json.dumps(m.snapshot(db)))
