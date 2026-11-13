import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_archive_leaderboard.py"
pytestmark = pytest.mark.skipif(not SCRIPT.exists() or not (ROOT / "docs").exists(), reason="docs/ and scripts/ are not shipped")


def _mod():
    spec = importlib.util.spec_from_file_location("render_archive_leaderboard", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_page_matches_the_snapshot():
    m = _mod()
    assert m.PAGE.read_text() == m.render(json.loads(m.SNAPSHOT.read_text()))


def test_every_group_lists_every_algorithm():
    from grn_sim.inference import registry
    full = {s.key for s in registry.specs()}
    snap = json.loads(_mod().SNAPSHOT.read_text())
    for key, g in snap["groups"].items():
        if g["runs"]:
            assert set(g["algorithms"]) == full, (key, full ^ set(g["algorithms"]))
            assert all(a["runs"] == g["runs"] for a in g["algorithms"].values()), key


@pytest.mark.skipif(os.environ.get("GRN_SIM_ARCHIVE_CHECK") != "1", reason="set GRN_SIM_ARCHIVE_CHECK=1 to compare with data/rgn_db.sqlite (read-only)")
def test_snapshot_equals_the_archive():
    m = _mod()
    assert json.loads(m.SNAPSHOT.read_text()) == json.loads(json.dumps(m.snapshot(ROOT / "data" / "rgn_db.sqlite")))
