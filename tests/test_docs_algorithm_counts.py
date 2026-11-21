"""Live documentation and the page must not state an algorithm count that differs from the registry (a recurring staleness:
the README said 18 and 19 for releases after the registry had 24).

Historical pages (a "Historical" / "snapshot" banner in their first lines) and lines that say they quote an earlier state are skipped.
"""
import re
from pathlib import Path

import pytest

from grn_sim.inference import registry

ROOT = Path(__file__).resolve().parent.parent
LIVE_DOCS = ["README.md", "SPECIFICATION.md", "docs/README.md", "docs/algorithms/README.md", "docs/ARCHITECTURE.md",
             "docs/WORKFLOW_STEPS_PIPELINE.md", "docs/REAL_WORLD_BENCHMARK_PROVENANCE_AND_USAGE.md", "docs/PERTURBSEQ.md"]
PATTERN = re.compile(r"(?<![\d.])(\d{2})[ -](?:registered |reverse-engineering |inference |comparative |re-engineering )?(?:algorithms?|methods)\b"
                     r"|\ball (\d{2}) (?:reverse-engineering |inference )?(?:algorithms|methods)\b", re.I)
HISTORICAL = re.compile(r"histor|original|first|earlier|previous|at the time|snapshot|then|until|before|was |were |used to|not all|"
                        r"six methods|archive now|registry now|VCC|vcc_screen|stored simulations|with \d+ algorithms", re.I)


def _stated(line: str):
    return [int(g) for m in PATTERN.finditer(line) for g in m.groups() if g]


@pytest.mark.parametrize("name", LIVE_DOCS)
def test_live_docs_state_the_registry_size(name):
    path = ROOT / name
    if not path.exists():                                        # the Docker image ships no docs
        pytest.skip(f"{name} is not available")
    text = path.read_text()
    if re.search(r"historical snapshot", "\n".join(text.splitlines()[:8]), re.I):
        pytest.skip(f"{name} is a labelled historical snapshot")
    n = len(registry.keys())
    stale = [(i + 1, line.strip()[:110]) for i, line in enumerate(text.splitlines())
             if any(v != n for v in _stated(line)) and not HISTORICAL.search(line)]
    assert not stale, f"{name} states an algorithm count other than {n}: {stale}"


def test_the_page_defaults_match_the_registry():
    page = ROOT / "src" / "grn_sim" / "web" / "static" / "index.html"
    html = page.read_text()
    n = len(registry.keys())
    defaults = {int(v) for v in re.findall(r'class="alg-total">(\d+)</span>', html)}
    assert defaults == {n}, f"index.html shows {sorted(defaults)} before the registry loads; the registry has {n}"
    assert f".size || {n};" in html
    assert f"Select all {n} available" in html
