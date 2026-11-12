"""docs/results/mechanistic_vs_statistical_top5.md is rendered from the corpus; it must not drift from the data."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_top5_results.py"


@pytest.fixture(scope="module")
def renderer():
    if not (ROOT / "docs" / "results").exists():
        pytest.skip("docs/ is not shipped (Docker image)")
    spec = importlib.util.spec_from_file_location("render_top5_results", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_committed_page_matches_a_fresh_render(renderer):
    assert renderer.PAGE.read_text() == renderer.render(renderer.load()), "run scripts/render_top5_results.py"


def test_controls_are_never_ranked_and_the_tables_are_sorted(renderer):
    rows = renderer.load()
    for generators in ([renderer.MECHANISTIC], list(renderer.STATISTICAL)):
        summary = renderer.summarise([r for r in rows if r["generator"] in generators])
        assert not {t["algorithm"] for t in summary} & set(renderer.CONTROLS)
        assert [t["mean"] for t in summary] == sorted((t["mean"] for t in summary), reverse=True)
        assert len(summary) == 20


def test_group_sizes_match_the_corpus(renderer):
    rows = renderer.load()
    mech = renderer.summarise([r for r in rows if r["generator"] == renderer.MECHANISTIC])
    stat = renderer.summarise([r for r in rows if r["generator"] in renderer.STATISTICAL])
    assert {t["n"] for t in mech} == {200} and {t["n"] for t in stat} == {800}
