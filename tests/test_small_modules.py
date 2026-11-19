"""Smoke and value tests for modules that had no direct test: the agent loop, the thesis sweeps, the screen-design statistics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from grn_sim.agents.multi_agent_system import CoordinatorAgent
from grn_sim.benchmarks import parameter_sweep as ps
from grn_sim.benchmarks.synthetic_sweep import run_noise_sweep, run_topology_comparison
from grn_sim.inference import registry
from grn_sim.perturbseq.partial_stats import design_statistics


def test_coordinator_runs_a_study_end_to_end():
    out = CoordinatorAgent().run_study(n=8, k_av=1.5, net_type="power_law", noise=0.0, seed=1)
    assert out["network_parameters"]["genes"] == 8
    assert out["network_parameters"]["true_edges"] == int(np.sum(out["true_matrix"] != 0))
    assert out["metrics"] and set(out["metrics"]) == set(out["inferred_matrices"])
    for m in out["metrics"].values():
        assert 0.0 <= m["Coverage %"] <= 100.0 and 0.0 <= m["F1-Score"] <= 1.0
    # the same seed gives the same study
    again = CoordinatorAgent().run_study(n=8, k_av=1.5, net_type="power_law", noise=0.0, seed=1)
    assert again["metrics"] == out["metrics"]


def test_topology_comparison_shapes():
    res = run_topology_comparison(network_sizes=(8, 10), repeats=1, seed=3)
    assert set(res) == {"random", "power_law"}
    for algos in res.values():
        assert {"s2g", "g2s", "nir", "fast", "voting"} <= set(algos)
        assert all(len(v) == 2 for v in algos.values())


def test_noise_sweep_shapes_and_range():
    res = run_noise_sweep(network_size=8, noise_levels=(0.0, 0.05), repeats=1, seed=5)
    assert set(res) == {"s2g", "g2s", "nir", "voting"}
    assert all(len(v) == 2 and all(0.0 <= x <= 100.0 for x in v) for v in res.values())


def test_sweep_axes_and_metadata_follow_the_registry():
    assert ps.SIZE_VALUES == tuple(sorted(ps.SIZE_VALUES)) and ps.K_VALUES == tuple(sorted(ps.K_VALUES))
    assert 0.0 in ps.NOISE_VALUES and [t[0] for t in ps.TOPOLOGY_DEFS] == ["random", "power_law", "hierarchical"]
    meta = ps.ALGO_METADATA()
    assert meta and set(meta) == set(registry.sweep_metadata())


def _screen():
    # genes 0..3; singles on genes 0 and 1; edges 0->1, 1->2, 3->2 (adjacency[regulator, target])
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 2] = adj[3, 2] = 1.0
    eff = np.zeros((4, 2))
    eff[0, 0] = -2.0     # own knockdown, never counted
    eff[1, 0] = 1.0      # gene 1 responds to perturbing gene 0
    eff[2, 0] = 0.5
    eff[2, 1] = 1.0
    return SimpleNamespace(effect_threshold=0.3, n_genes=4, conditions=[(0,), (1,)], labels=["effective", "effective"],
                           total_effects=eff, adjacency=adj)


def test_design_statistics_counts_identifiable_edges():
    s = design_statistics(_screen())
    assert s["design"]["n_perturbed"] == 2 and s["design"]["fraction_perturbed"] == 0.5
    ident = s["identifiability"]
    assert (ident["edges_total"], ident["edges_P_to_P"], ident["edges_P_to_Q"], ident["edges_from_unperturbed"]) == (3, 1, 1, 1)
    assert ident["fraction_edges_identifiable"] == pytest.approx(2 / 3)
    cov = s["coverage"]
    assert cov["n_unperturbed"] == 2 and cov["fraction_unperturbed_responding"] == 0.5
    assert cov["median_perturbations_per_responding_gene"] == 2.0
    assert cov["fraction_unperturbed_with_perturbed_direct_regulator"] == 0.5


def test_design_statistics_threshold_changes_what_responds():
    strict = design_statistics(_screen(), effect_threshold=2.5)
    assert strict["coverage"]["fraction_unperturbed_responding"] == 0.0
    assert strict["coverage"]["median_perturbations_per_responding_gene"] == 0.0
