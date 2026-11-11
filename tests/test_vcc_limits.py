"""
The size limits of the cross-atlas comparison follow from measured timings (docs/results/vcc_cross_atlas_timings.json), not
from assumptions: the largest measured size whose whole run fits the run budget and whose double-perturbation screen fits the
screen budget.
"""
import json
from pathlib import Path

from grn_sim.benchmarks import vcc_benchmark as vcc

TIMINGS = json.loads((Path(__file__).resolve().parents[1] / "docs" / "results" / "vcc_cross_atlas_timings.json").read_text())
RUN_BUDGET = TIMINGS["budget_seconds"]["whole_run"]
SCREEN_BUDGET = TIMINGS["budget_seconds"]["double_screen"]


def _largest_within(entries, key, budget):
    """Largest size whose WORST measurement fits the budget (sizes measured more than once count by their slowest run)."""
    worst = {}
    for e in entries:
        worst[e["n"]] = max(worst.get(e["n"], 0.0), e[key])
    ok = [n for n, v in worst.items() if v <= budget]
    return max(ok) if ok else 0


def test_mechanistic_gene_limit_is_the_largest_measured_size_within_the_run_budget():
    runs = [r for r in TIMINGS["runs"] if r["generator"] == "mechanistic"]
    assert vcc.MAX_GENES == _largest_within(runs, "total", RUN_BUDGET) == vcc.MAX_GENES_BY_GENERATOR["mechanistic"]
    assert any(r["total"] > RUN_BUDGET for r in runs), "the next measured size must exceed the budget"


def test_screen_limits_follow_the_measured_screen_cost():
    for gen, limit in vcc.SCREEN_MAX_GENES.items():
        screens = [s for s in TIMINGS["screens"] if s["generator"] == gen]
        assert screens, f"no screen timing for {gen}"
        expected = min(_largest_within(screens, "seconds", SCREEN_BUDGET), vcc.MAX_GENES_STATISTICAL)
        assert limit == expected, f"{gen}: limit {limit}, timings give {expected}"
        if limit < vcc.MAX_GENES_STATISTICAL:
            assert any(s["seconds"] > SCREEN_BUDGET for s in screens), f"{gen}: no measurement shows the screen over budget"


def test_each_generator_size_limit_is_the_largest_measured_size_within_the_run_budget():
    for gen, limit in vcc.MAX_GENES_BY_GENERATOR.items():
        runs = [r for r in TIMINGS["runs"] if r["generator"] == gen and r.get("limits") == "configured"] or \
               [r for r in TIMINGS["runs"] if r["generator"] == gen]
        assert runs, f"no measured run for {gen}"
        assert limit == min(_largest_within(runs, "total", RUN_BUDGET), vcc.MAX_GENES_STATISTICAL), gen
    assert vcc.MAX_GENES_HILL_REFIT >= vcc.MAX_GENES_STATISTICAL
