"""Stage 1 of the Perturb-seq generator: kinetics kernel, precise steady-state solver, condition screen and export."""

import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import NetworkTopology, TopologyType
from grn_sim.perturbseq import LABELS, design_matrix, screen_to_anndata, simulate_conditions
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.kinetics import evaluate_derivatives, evaluate_jacobian, edge_elasticities
from grn_sim.simulator.ode_solver import ODESolver
from grn_sim.simulator.steady_state import SteadyStateSolver


def network(n=30, seed=2, net_type=TopologyType.RANDOM, k_av=1.8):
    return GeneNetwork.create(n=n, k_av=k_av, net_type=net_type, ratio=0.55, seed=seed)


def custom_network(adjacency, seed=0):
    A = np.asarray(adjacency, dtype=int)
    return GeneNetwork(NetworkTopology(A, TopologyType.RANDOM, k_av=1.0, ratio=0.5), rng=np.random.default_rng(seed))


def reference_derivatives(x, net, u=None, scale=None):
    """The original per-gene loop of kinetics.py, kept here as the reference implementation."""
    n = net.n
    T, p = net.topology.adjacency, net.params
    eff_x = np.maximum(x + (u if u is not None else np.zeros(n)), 1e-12)
    prod = np.ones(n)
    for k in range(n):
        term = 1.0
        for j in np.where(T[:, k] != 0)[0]:
            if T[j, k] == 1:
                val_m = eff_x[j] ** p.n_k[j, k]
                term *= 1.0 + val_m / (val_m + p.Ka[j, k] ** p.n_k[j, k] + 1e-12)
            else:
                val_m = eff_x[j] ** p.n_i[j, k]
                ki_m = p.Ki[j, k] ** p.n_i[j, k]
                term *= ki_m / (val_m + ki_m + 1e-12)
        prod[k] = term
    production = prod * p.v
    if scale is not None:
        production = production * scale
    return production - p.B * x


# --- kinetics -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("net_type", list(TopologyType))
def test_vectorised_kinetics_match_the_per_gene_loop(net_type):
    # Same operations in the same order, but numpy may evaluate array powers with SIMD code (e.g. AVX-512 on some
    # CI runners) that differs from the scalar power in the last bit, so compare to rounding, not bit for bit.
    rng = np.random.default_rng(1)
    for seed in range(3):
        net = network(24, seed, net_type)
        x = rng.uniform(0.01, 3.0, 24)
        u = np.zeros(24)
        u[seed] = 1.5
        scale = np.ones(24)
        scale[5] = 0.2
        np.testing.assert_allclose(evaluate_derivatives(x, net, u, scale), reference_derivatives(x, net, u, scale),
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(evaluate_derivatives(x, net), reference_derivatives(x, net), rtol=1e-12, atol=1e-14)


def test_analytic_jacobian_matches_finite_differences_and_sparse_form():
    net = network(20, 4)
    x = np.random.default_rng(0).uniform(0.2, 2.5, 20)
    scale = np.ones(20)
    scale[3] = 0.3
    J = evaluate_jacobian(x, net, production_scale=scale)
    h = 1e-6
    Jn = np.column_stack([
        (evaluate_derivatives(x + h * e, net, production_scale=scale) - evaluate_derivatives(x - h * e, net, production_scale=scale)) / (2 * h)
        for e in np.eye(20)
    ])
    assert np.allclose(J, Jn, atol=1e-6)
    assert np.allclose(evaluate_jacobian(x, net, production_scale=scale, sparse=True).toarray(), J)


def test_elasticity_is_zero_off_the_edges_and_signed_on_them():
    net = network(20, 5)
    E = edge_elasticities(np.ones(20), net)
    A = net.topology.adjacency
    assert np.all(E[A == 0] == 0)
    assert np.all(E[A == 1] >= 0) and np.all(E[A == -1] <= 0)


# --- solvers ------------------------------------------------------------------------------------------

def test_precise_solver_finds_the_thesis_fixed_point_with_a_far_smaller_residual():
    for seed in range(4):
        net = network(30, seed)
        x0 = 0.5 + np.random.default_rng(seed).uniform(size=30)
        x_thesis, ok_t, _, _ = ODESolver().solve_with_diagnostics(net, x0=x0.copy())
        x_fast, ok_f, _, diag = SteadyStateSolver(check_stability=True).solve_with_diagnostics(net, x0=x0.copy())
        assert ok_t and ok_f
        assert diag.final_efnorm < 1e-8
        assert np.max(np.abs(evaluate_derivatives(x_fast, net))) < 1e-8
        assert np.max(np.abs(np.log2(x_fast / x_thesis))) < 0.02


def test_lsoda_relaxation_agrees_with_euler_relaxation():
    net = network(30, 1)
    scale = np.ones(30)
    scale[7] = 0.2
    a, ok_a, _, _ = SteadyStateSolver(relax="euler").solve_with_diagnostics(net, x0=np.ones(30), production_scale=scale)
    b, ok_b, _, _ = SteadyStateSolver(relax="lsoda").solve_with_diagnostics(net, x0=np.ones(30), production_scale=scale)
    assert ok_a and ok_b and np.allclose(a, b, rtol=1e-6)


def test_invalid_relaxation_is_rejected():
    with pytest.raises(ValueError):
        SteadyStateSolver(relax="rk4")


def test_simulator_knockout_mode_silences_the_gene():
    exp = PerturbationSimulator(perturbation_mode="knockout").run(network(12, 3), seed=3)
    assert np.allclose(np.diag(exp.U), -1.0)
    assert np.all(np.diag(exp.X) < -0.99)


# --- condition screen ---------------------------------------------------------------------------------

def test_knockdown_screen_hits_its_targets_and_labels_every_single_perturbation():
    net = network(30, 2)
    screen = simulate_conditions(net, mode="knockdown", efficiency=0.8, seed=0)
    assert screen.means.shape == (30, 30) and screen.total_effects.shape == (30, 30)
    assert screen.steady.all()
    own = np.array([screen.total_effects[j, j] for j in range(30)])
    assert np.median(own) < -1.5, "an 80% knockdown cuts the gene's own level by roughly log2(0.2) = -2.3"
    out_deg = np.sum(net.topology.adjacency != 0, axis=1)
    assert set(screen.labels) <= set(LABELS)
    for j, lab in enumerate(screen.labels):
        assert (lab == "no_targets") == (out_deg[j] == 0)
        off = np.delete(np.abs(screen.total_effects[:, j]), j)
        assert (lab == "effective") == (out_deg[j] > 0 and off.max() >= screen.effect_threshold)


def test_knockout_is_stronger_than_knockdown():
    net = network(30, 2)
    kd = simulate_conditions(net, mode="knockdown", efficiency=0.5, seed=0)
    ko = simulate_conditions(net, mode="knockout", seed=0)
    off = kd.off_target_mask()
    assert np.abs(ko.total_effects[off]).mean() > np.abs(kd.total_effects[off]).mean()
    assert np.allclose(ko.means[np.arange(30), np.arange(30)], 0.0)


def test_linear_response_predicts_weak_knockdowns():
    net = network(30, 2)
    screen = simulate_conditions(net, mode="knockdown", efficiency=0.02, seed=0)
    predicted = screen.linear_response * np.log(0.98) / np.log(2.0)
    assert np.corrcoef(predicted.ravel(), screen.total_effects.ravel())[0, 1] > 0.98


def test_double_knockdowns_of_independent_genes_are_log_additive():
    # Two separate chains 0 -> 1 -> 2 and 3 -> 4 -> 5 plus a shared target 6 of genes 1 and 4.
    A = np.zeros((7, 7), dtype=int)
    A[0, 1] = A[1, 2] = 1
    A[3, 4] = A[4, 5] = 1
    A[1, 6] = A[4, 6] = 1
    net = custom_network(A, seed=3)
    screen = simulate_conditions(net, mode="knockdown", pairs=[(0, 3), (2, 5)], pair_pool="all", seed=0)
    assert screen.condition_names[-2:] == ["Gene_001+Gene_004", "Gene_003+Gene_006"]
    assert screen.pair_deviation is not None and screen.epistasis is not None
    # Genes 2 and 5 are leaves: knocking both down touches nobody else, so the pair is exactly additive.
    assert screen.epistasis[1] < 1e-6
    # Chains 0->1->2 and 3->4->5 are independent apart from the shared target 6.
    dev = screen.pair_deviation[:, 0]
    assert np.all(np.abs(dev[[1, 2, 4, 5]]) < 1e-6)


def test_random_pairs_come_from_effective_genes_by_default():
    net = network(30, 2)
    screen = simulate_conditions(net, mode="knockdown", pairs=10, seed=0)
    effective = {j for j, lab in enumerate(screen.labels) if lab == "effective"}
    pairs = [screen.conditions[k] for k in screen.pair_columns]
    assert len(pairs) == 10 and len(set(pairs)) == 10
    assert all(a in effective and b in effective for a, b in pairs)


@pytest.mark.parametrize("kwargs", [
    {"mode": "overexpress"},
    {"mode": "knockdown", "efficiency": 1.0},
    {"mode": "knockdown", "pairs": "some"},
    {"mode": "knockdown", "pairs": [(1, 1)]},
    {"mode": "knockdown", "pair_pool": "best"},
])
def test_invalid_screen_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        simulate_conditions(network(10, 1), **kwargs)


def test_summary_reports_screen_statistics():
    summary = simulate_conditions(network(30, 2), mode="knockdown", pairs=5, seed=0).summary()
    assert summary["n_single"] == 30 and summary["n_pairs"] == 5
    assert 0.0 <= summary["fraction_with_effect"] <= 1.0
    assert 0.0 <= summary["fraction_effects_indirect"] <= 1.0
    assert sum(summary["label_counts"].values()) == 30


# --- export -------------------------------------------------------------------------------------------

def test_screen_exports_to_anndata_with_ground_truth(tmp_path):
    anndata = pytest.importorskip("anndata")
    screen = simulate_conditions(network(12, 2), mode="knockdown", pairs=3, seed=0)
    U = design_matrix(screen)
    assert U.shape == (12, 15) and np.allclose(U[0, 0], -0.8) and np.count_nonzero(U[:, -1]) == 2
    adata = screen_to_anndata(screen, cells_per_condition=5, seed=1)
    assert adata.n_obs == 16 * 5
    assert set(adata.obs["perturbation"]) == {"non-targeting", *screen.condition_names}
    assert adata.obs["n_perturbed_genes"].max() == 2
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    gt = back.uns["ground_truth"]
    assert gt["mode"] == "knockdown"
    assert np.allclose(gt["total_effects_log2"], screen.total_effects)
    assert list(gt["single_labels"]) == screen.labels
    assert len(gt["pair_epistasis"]) == 3


def test_server_exports_a_knockdown_screen_by_default():
    pytest.importorskip("anndata")
    from grn_sim.web import server

    server._execute_simulation({"num_genes": 8, "topology": "power_law", "seed": 7})
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=3, n_pairs=2)
    assert filename.endswith(".h5ad") and "knockdown_2pairs" in filename and len(data) > 0
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=3, perturbation_mode="input")
    assert "knockdown" not in filename


@pytest.mark.parametrize("query", [{"perturbation": ["overexpress"]}, {"pairs": ["-1"]}, {"pairs": ["100000"]}])
def test_server_rejects_bad_perturb_seq_options(query):
    from grn_sim.web import server

    with pytest.raises(ValueError):
        server._parse_perturb_seq_options(query)
