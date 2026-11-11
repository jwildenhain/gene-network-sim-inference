"""The findings document is computed from the stored simulations: it must match them exactly, and its statistics must be sound."""
import json
from pathlib import Path

import numpy as np
import pytest

from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.evaluation import vcc_findings as vf
from grn_sim.paths import bundled_path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rows():
    path = bundled_path("benchmarks", "vcc_screen_benchmark.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_findings_document_matches_the_stored_simulations(rows):
    if not (ROOT / "docs" / "results").is_dir():
        pytest.skip("docs/results is not available (installed package or Docker image)")
    doc = ROOT / "docs" / "results" / "vcc_screen_findings.md"
    assert doc.read_text() == vf.render(vf.compute(rows)), "regenerate with scripts/run_vcc_benchmark.py --report"


def test_the_corpus_only_holds_admitted_algorithms(rows):
    names = {r["algorithm"] for r in rows}
    assert names == {a.name for a in vb.algorithm_library()}
    assert not names & set(vb.DROPPED)
    assert all(r["status"] == "ok" for r in rows)


def test_matched_comparison_pairs_by_generator_and_seed():
    rows = []
    for gen in ("linear_sem", "boolean"):
        for seed in range(1, 9):
            for pairs, bump in ((0, 0.0), (50, 0.05)):
                for name, v in (("a", 0.3 + bump), ("b", 0.2 + 0.001 * seed)):
                    rows.append({**vb.Setting(generator=gen, pairs=pairs).to_dict(), "seed": seed, "algorithm": name,
                                 "status": "ok", "aupr_direct": v})
    m = vf.matched(rows, "pairs", [0, 50], vf.BASE, ["a", "b"])
    assert m["algorithms"]["a"]["50"]["n_pairs"] == 16
    assert m["algorithms"]["a"]["50"]["diff"] == pytest.approx(0.05) and m["algorithms"]["a"]["50"]["p"] < 0.01
    assert m["algorithms"]["b"]["50"]["diff"] == pytest.approx(0.0) and m["algorithms"]["b"]["50"]["p"] == 1.0
    assert np.isclose(m["algorithms"]["a"]["0"]["mean"], 0.3)
