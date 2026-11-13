"""docs/results/audit_100_250.md is exactly what the renderer makes of audit_100_250.json."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("render_audit_page", ROOT / "scripts" / "render_audit_page.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_page_matches_the_data():
    m = _mod()
    assert m.PAGE.read_text() == m.render(json.loads(m.DATA.read_text()))


def test_every_registered_algorithm_is_in_both_sizes():
    from grn_sim.inference import registry
    d = json.loads((ROOT / "docs" / "results" / "audit_100_250.json").read_text())
    keys = {s.key for s in registry.specs()}
    for size, rows in d["runs"].items():
        assert {r["alg"] for r in rows} == keys, size
        assert all(r["seconds"] >= 0 and 0 <= r["f1"] <= 1 for r in rows)
