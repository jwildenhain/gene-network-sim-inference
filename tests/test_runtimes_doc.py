"""docs/results/algorithm_runtimes.md is rendered from stored results and lists every registered algorithm."""
import importlib.util
from pathlib import Path

import pytest

from grn_sim.inference import registry

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "results" / "algorithm_runtimes.md"


def _renderer():
    spec = importlib.util.spec_from_file_location("render_runtimes", ROOT / "scripts" / "render_runtimes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_page_matches_the_stored_data_and_lists_every_algorithm():
    if not PAGE.exists():
        pytest.skip("docs/results/algorithm_runtimes.md is not available")
    mod = _renderer()
    text = mod.render()
    assert PAGE.read_text() == text, "run python scripts/render_runtimes.py"
    for key in registry.keys():
        assert f"| `{key}` |" in text


def test_time_formatting():
    mod = _renderer()
    assert mod.fmt(None) == "-" and mod.fmt(0.004) == "<0.01 s" and mod.fmt(3.2) == "3.20 s"
    assert mod.fmt(42) == "42 s" and mod.fmt(150) == "2.5 min" and mod.fmt(7200) == "2.0 h"
