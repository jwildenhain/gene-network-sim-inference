"""Stage 4 of the Perturb-seq generator: kinetics calibration, presets, realism statistics and report."""

import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import NetworkTopology, TopologyType, generate_topology
from grn_sim.perturbseq import (
    CellConfig,
    MeasurementConfig,
    population_to_anndata,
    simulate_cells,
    simulate_conditions,
)
from grn_sim.perturbseq.calibration import PRESETS, NetworkPreset, calibrate_kinetics
from grn_sim.perturbseq.realism import (
    TARGETS,
    _bh,
    _hotelling_pvalue,
    format_report,
    observed_statistics,
    realism_report,
    realism_score,
    truth_statistics,
)
from grn_sim.simulator.kinetics import edge_elasticities


def activation_rich(n=60, seed=1):
    rng = np.random.default_rng(seed)
    topo = generate_topology(n, k_av=3.0, net_type=TopologyType.MODULAR, ratio=0.8, subdiv=3, rng=rng)
    return GeneNetwork(topo, rng=rng)


# --- calibration --------------------------------------------------------------------------------------

def test_anchored_kinetics_wake_up_silent_edges():
    net = activation_rich()
    A = net.topology.adjacency
    from grn_sim.simulator.steady_state import SteadyStateSolver

    x0, _, _, _ = SteadyStateSolver().solve_with_diagnostics(net)
    silent_before = np.mean(np.abs(edge_elasticities(x0, net)[A != 0]) < 0.05)
    diag = calibrate_kinetics(net, seed=2)
    silent_after = np.mean(np.abs(edge_elasticities(net.steady_state_unperturbed, net)[A != 0]) < 0.05)
    assert diag["steady"] and diag["final_q95_dlogx"] < 0.5   # a few bistable genes may keep flipping
    assert silent_after < 0.5 * silent_before and silent_before > 0.1
    assert np.all(net.params.n_k[A > 0] >= 1.0) and np.all(net.params.n_k[A > 0] <= 3.0)
    assert np.all(net.params.Ka[A == 0] == 0) and np.all(net.params.Ki[A == 0] == 0)


def test_calibration_converges_on_typical_networks():
    converged = []
    for seed in (1, 2, 4, 5):
        rng = np.random.default_rng(seed)
        net = GeneNetwork(generate_topology(200, k_av=2.5, net_type=TopologyType.MODULAR, ratio=0.8, subdiv=4,
                                            rng=rng), rng=rng)
        converged.append(calibrate_kinetics(net, seed=2)["converged"])
    assert all(converged)


@pytest.mark.parametrize("kwargs", [{"spread": -1}, {"max_rounds": 0}, {"damping": 0.0}, {"tol": 0},
                                    {"hill_range": (3.0, 1.0)}])
def test_bad_calibration_arguments(kwargs):
    with pytest.raises(ValueError):
        calibrate_kinetics(activation_rich(20), **kwargs)


def test_presets_build_the_requested_regime():
    net = PRESETS["replogle_like"].build(120, seed=3)
    A = net.topology.adjacency
    assert net.topology.topology_type == TopologyType.MODULAR
    assert abs((A > 0).sum() / (A != 0).sum() - 0.8) < 0.08
    assert np.all(np.sum(A != 0, axis=0) >= 1)
    thesis = PRESETS["thesis"].build(24, seed=3)
    assert thesis.topology.topology_type == TopologyType.POWER_LAW
    custom = NetworkPreset(topology="random", k_av=2.0, anchor_kinetics=False).build(30, seed=1)
    assert custom.topology.topology_type == TopologyType.RANDOM


# --- statistics ---------------------------------------------------------------------------------------

def chain_screen():
    # 0 -> 1 -> 2 -> 3 (activating) plus an isolated gene 4
    A = np.zeros((5, 5), dtype=int)
    A[0, 1] = A[1, 2] = A[2, 3] = 1
    net = GeneNetwork(NetworkTopology(A, TopologyType.RANDOM, 1.0, 1.0), rng=np.random.default_rng(0))
    calibrate_kinetics(net, spread=0.0, hill_range=(2.0, 2.0), seed=0)
    return simulate_conditions(net, mode="knockdown", efficiency=0.9, effect_threshold=0.05, seed=0)


def test_truth_statistics_separate_direct_and_mediated_effects():
    st = truth_statistics(chain_screen())
    # knocking down 0 moves 1 (direct), 2 and 3 (mediated); 1 moves 2 (direct), 3; 2 moves 3: 6 effects, 3 direct
    assert st["affected_pair_density"] == pytest.approx(6 / (5 * 4))
    assert st["fraction_effects_indirect"] == pytest.approx(0.5)
    assert st["direct_to_total_ratio"] == pytest.approx(0.5)
    assert st["bidirectional_share"] == 0.0
    assert st["fraction_with_effect_truth"] == pytest.approx(3 / 5)
    assert 0.85 < st["truth_on_target_knockdown"] < 0.95


def test_benjamini_hochberg_and_hotelling():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(_bh(p), [0.04, 0.16 / 3, 0.16 / 3, 0.5])  # 0.03*4/2 = 0.06 is capped by 0.04*4/3
    rng = np.random.default_rng(1)
    null = [_hotelling_pvalue(rng.standard_normal((40, 5)), rng.standard_normal((200, 5))) for _ in range(300)]
    assert 0.03 < np.mean(np.asarray(null) < 0.05) < 0.08, "calibrated under the null"
    shifted = _hotelling_pvalue(rng.standard_normal((40, 5)) + 0.8, rng.standard_normal((200, 5)))
    assert shifted < 1e-6


@pytest.fixture(scope="module")
def measured():
    net = PRESETS["replogle_like"].build(40, seed=5)
    screen = simulate_conditions(net, mode="knockdown", seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=40, n_control_cells=300,
                         config=CellConfig(guide_efficiency_mean=0.9), seed=1)
    return screen, population_to_anndata(pop, screen, seed=2, measurement=MeasurementConfig())


@pytest.mark.parametrize("test", ["hotelling", "energy"])
def test_observed_statistics(measured, test):
    screen, adata = measured
    st = observed_statistics(adata, n_perm=50, phenotype_test=test)
    assert st["n_tested_perturbations"] > 30
    assert 0.0 <= st["fraction_with_phenotype"] <= 1.0
    assert 0.6 < st["on_target_knockdown"] < 0.95
    assert st["control_false_positive_rate"] <= 0.25
    assert 0.8 < st["cells_kept_fraction"] < 1.0


def test_observed_statistics_rejects_bad_input(measured):
    _, adata = measured
    with pytest.raises(ValueError):
        observed_statistics(adata, phenotype_test="anova")
    with pytest.raises(ValueError):
        observed_statistics(adata[~adata.obs["is_control"].values].copy())


def test_report_scores_distance_to_targets():
    inside = {name: (t.low + t.high) / 2 for name, t in TARGETS.items()}
    rows = realism_report(inside)
    assert len(rows) == len(TARGETS) and all(r["within"] for r in rows) and realism_score(rows) == 0.0
    off = dict(inside, fraction_with_phenotype=0.9)
    rows = realism_report(off)
    bad = [r for r in rows if not r["within"]]
    assert [r["statistic"] for r in bad] == ["fraction_with_phenotype"] and realism_score(rows) > 0
    assert "fraction_with_phenotype" in format_report(rows)
