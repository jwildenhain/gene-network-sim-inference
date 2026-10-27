"""Regression tests for the VCC comparative benchmark (grn_sim.benchmarks.vcc_benchmark) and its web endpoint."""

import itertools

import numpy as np
import pytest

from grn_sim.benchmarks import vcc_benchmark as vcc
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.web import server


@pytest.fixture(scope="module")
def benchmark():
    return vcc.run_vcc_comparative_benchmark(num_genes=12, k_av=1.8, seed=2026)


def _network(n=12, seed=2026):
    return GeneNetwork.create(n=n, k_av=1.8, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)


# --- rewiring ---------------------------------------------------------------------------------

def test_rewired_edges_keep_their_kinetics():
    net = _network()
    before = (net.topology.adjacency != 0).sum()
    vcc._rewire_edges(net, 0.45, np.random.default_rng(1))
    adj, p = net.topology.adjacency, net.params

    assert (adj != 0).sum() == before, "rewiring must neither create nor destroy edges"
    for param in (p.Ka, p.Ki, p.n_k, p.n_i):
        assert np.all(param[adj != 0] > 0), "every edge needs non-zero Hill parameters (x**0 == 1 regulates nothing)"
        assert np.all(param[adj == 0] == 0)


def test_zero_rewire_fraction_leaves_network_untouched():
    net = _network()
    adj = net.topology.adjacency.copy()
    vcc._rewire_edges(net, 0.0, np.random.default_rng(1))
    assert np.array_equal(net.topology.adjacency, adj)


# --- virtual single cells ----------------------------------------------------------------------

def test_virtual_cells_follow_the_perturbed_state_of_each_condition():
    exp = PerturbationSimulator(perturbation_factor=1.0, measurement_noise=0.0).run(_network(n=8), seed=3)
    names = [f"Gene_{i + 1:02d}" for i in range(8)]
    cells, meta = vcc._virtual_single_cells(exp, names, seed=5, n_control=0, cells_per_condition=400)

    for j in range(3):
        rows = [i for i, m in enumerate(meta) if m["target"] == names[j]]
        expected = np.maximum(0.2, exp.perturbed_states[:, j] * 60.0)  # column j = state after perturbing gene j
        assert np.allclose(cells[rows].mean(axis=0), expected, rtol=0.1, atol=1.0)


# --- response models ---------------------------------------------------------------------------

SIGN_CHECKED = ["vote", "s2g", "g2s", "ridge_cv", "nir", "dmdgrn"]


def test_response_predictions_are_positively_correlated_with_truth(benchmark):
    profiles = [t for t in benchmark["target_profiles"] if not t["is_ood"]]
    for model in SIGN_CHECKED:
        mean_r = np.mean([t["metrics"][model]["pearson_r"] for t in profiles])
        assert mean_r > 0.05, f"{model}: mean seen-target Pearson r = {mean_r:.3f} (sign convention inverted?)"


def test_unavailable_models_are_reported_not_substituted(benchmark):
    models = {m["key"]: m for m in benchmark["parameters"]["response_models"]}
    unavailable = {k for k, m in models.items() if not m["available"]}
    assert "cpa_lite" in unavailable and models["cpa_lite"]["reason"]
    assert all(m["reason"] for m in models.values() if not m["available"])
    assert not any(m["reason"] for m in models.values() if m["available"])

    for profile in benchmark["target_profiles"]:
        assert not set(profile["predictions"]) & unavailable
        assert not set(profile["metrics"]) & unavailable
        assert set(profile["predictions"]) == set(models) - unavailable

    # No model may silently reuse another model's predictions (CellBox used to be a copy of Vote).
    profiles = benchmark["target_profiles"]
    for a, b in itertools.combinations(profiles[0]["predictions"], 2):
        assert any(t["predictions"][a] != t["predictions"][b] for t in profiles), f"{a} and {b} are identical"


def test_response_operator_rejects_degenerate_fits():
    assert vcc._response_operator(np.zeros((4, 4)), 1.0)[0] is None
    bad = np.eye(4)
    bad[0, 1] = np.nan
    assert vcc._response_operator(bad, 1.0)[0] is None
    op, _ = vcc._response_operator(np.diag([2.0, 4.0]), -1.0)
    assert np.allclose(op, np.diag([-0.5, -0.25]))


def test_ood_models_are_fit_without_the_unseen_experiments(monkeypatch):
    calls = []
    original = vcc._build_runners

    def recording_runners(k_est):
        runners = original(k_est)
        for name, runner in runners.items():
            def fit(X, U, _fit=runner.fit, _name=name, **kw):
                calls.append((_name, np.array(X), np.array(U)))
                return _fit(X, U, **kw)
            runner.fit = fit
        return runners

    monkeypatch.setattr(vcc, "_build_runners", recording_runners)
    res = vcc.run_vcc_comparative_benchmark(num_genes=8, k_av=1.5, seed=42, rewire_fractions=[0.0])

    n_unseen = res["parameters"]["unseen_ood_targets"]
    unseen = slice(8 - n_unseen, 8)
    vote_fits = [(X, U) for name, X, U in calls if name == "vote"]
    assert len(vote_fits) == 2  # full-data fit for edge recovery + held-out fit for response prediction
    full, held_out = vote_fits
    assert np.any(full[0][:, unseen]) and np.any(full[1][:, unseen])
    assert not np.any(held_out[0][:, unseen]) and not np.any(held_out[1][:, unseen])
    assert np.array_equal(held_out[0][:, : 8 - n_unseen], full[0][:, : 8 - n_unseen])


# --- epistasis ---------------------------------------------------------------------------------

def test_epistasis_is_deviation_of_the_simulated_double_perturbation_from_log_additivity():
    sim = PerturbationSimulator(perturbation_factor=1.0, measurement_noise=0.0)
    exp = sim.run(_network(n=6), seed=1)
    eps, unsteady = vcc._dual_perturbation_screen(sim, exp)

    assert np.allclose(eps, eps.T) and np.all(np.diag(eps) == 0) and unsteady == 0
    x_ss = exp.reference_steady_state

    def lfc(u):
        return np.log2(sim.solver.solve_with_diagnostics(exp.network, x0=x_ss.copy(), perturbation=u)[0] / x_ss)

    i, j = 1, 4
    expected = np.linalg.norm(lfc(exp.U[:, i] + exp.U[:, j]) - (lfc(exp.U[:, i]) + lfc(exp.U[:, j])))
    assert eps[i, j] == pytest.approx(expected)


def test_epistasis_screen_does_not_depend_on_measurement_noise():
    quiet = PerturbationSimulator(perturbation_factor=1.0, measurement_noise=0.0)
    noisy = PerturbationSimulator(perturbation_factor=1.0, measurement_noise=0.01)
    net = _network(n=6)
    eps_quiet, _ = vcc._dual_perturbation_screen(quiet, quiet.run(net, seed=1))
    eps_noisy, _ = vcc._dual_perturbation_screen(noisy, noisy.run(net, seed=1))
    # The solver warm-starts from the previous run, so allow its convergence tolerance (~1e-3);
    # measurement noise on the singles used to add a ~0.18 floor.
    assert np.allclose(eps_quiet, eps_noisy, atol=5e-3)


# --- parameter validation / cache --------------------------------------------------------------

@pytest.mark.parametrize("args", [
    (2, 1.8, 0.15), (1000, 1.8, 0.15), (12.5, 1.8, 0.15),
    (12, float("nan"), 0.15), (12, 0.0, 0.15), (12, 50.0, 0.15),
    (12, 1.8, 2.0), (12, 1.8, -1.0), (12, 1.8, float("nan")), (12, 1.8, float("inf")),
])
def test_invalid_parameters_are_rejected(args):
    with pytest.raises(ValueError):
        vcc.validate_vcc_params(*args)
    with pytest.raises(ValueError):
        server._build_vcc_benchmark_payload(args[0], args[1], rewire_fraction=args[2])


def test_valid_parameters_are_accepted():
    vcc.validate_vcc_params(vcc.MIN_GENES, 1.5, 0.0)
    vcc.validate_vcc_params(vcc.MAX_GENES, vcc.MAX_K_AV, 1.0)


def test_benchmark_rejects_invalid_gene_count_directly():
    with pytest.raises(ValueError):
        vcc.run_vcc_comparative_benchmark(num_genes=2)


def test_vcc_cache_is_bounded_and_quantised(monkeypatch):
    runs = []
    monkeypatch.setattr(
        server, "run_vcc_comparative_benchmark",
        lambda **kw: runs.append(kw) or {"status": "success", **kw},
    )
    saved = dict(server.BENCHMARK_CACHE)
    server.BENCHMARK_CACHE.clear()
    try:
        server._build_vcc_benchmark_payload(8, 1.5, seed=1, rewire_fraction=0.15)
        server._build_vcc_benchmark_payload(8, 1.5, seed=1, rewire_fraction=0.1504)
        assert len(runs) == 1, "rewire fractions that round to the same value share a cache entry"

        for seed in range(100, 140):
            server._build_vcc_benchmark_payload(8, 1.5, seed=seed, rewire_fraction=0.15)
        assert len([k for k in server.BENCHMARK_CACHE if k.startswith("vcc_")]) == server.VCC_CACHE_LIMIT
    finally:
        server.BENCHMARK_CACHE.clear()
        server.BENCHMARK_CACHE.update(saved)


# --- single registry -----------------------------------------------------------------------------

def test_algorithm_registry_matches_the_runners_and_drives_the_table(benchmark):
    keys = [k for k, _ in vcc.ALGORITHMS]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(vcc._build_runners(1)) | {"atlasshift"}
    assert [row["algorithm"] for row in benchmark["summary_table"]] == [k.upper() for k in keys]
    assert all(row["paradigm"] for row in benchmark["summary_table"])


def test_response_models_cover_runners_and_payload_order(benchmark):
    assert {m.key for m in vcc.RESPONSE_MODELS} <= set(vcc._build_runners(1))
    assert [m["key"] for m in benchmark["parameters"]["response_models"]] == [m.key for m in vcc.RESPONSE_MODELS]
    assert all(m["label"] for m in benchmark["parameters"]["response_models"])


def test_atlasshift_has_no_ignored_weights_argument():
    import inspect
    from grn_sim.inference.atlasshift import AtlasShift
    assert "weights" not in inspect.signature(AtlasShift).parameters
