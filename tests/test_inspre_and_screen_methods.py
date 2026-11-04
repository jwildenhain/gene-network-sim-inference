"""inspre, calibrated ACE, interventional baselines, bicycle-lite, DOTEARS-lite, consensus and the prior experiment."""

import numpy as np
import pytest

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.evaluation.prior_experiment import make_prior, prior_penalty
from grn_sim.inference import registry
from grn_sim.inference.bicycle_lite import _Problem
from grn_sim.inference.inspre import InspreAlgorithm, ace_from_xu, direct_effects, inspre_solve
from grn_sim.inference.interventional_baselines import (GeneFeatureBaseline, GRNBoostLite, MeanDifference,
                                                        PearsonCorrelation)
from grn_sim.inference.interventional_consensus import consensus, rank_normalise


def linear_screen(n=30, k=3, seed=0, sigma=0.0):
    """Cyclic linear model: column i of X is the response to perturbing gene i (U = -diag(delta))."""
    rng = np.random.default_rng(seed)
    G = (rng.random((n, n)) < k / n) * rng.normal(size=(n, n)) * 0.5
    np.fill_diagonal(G, 0)
    rho = max(abs(np.linalg.eigvals(G)))
    if rho > 0.7:
        G *= 0.7 / rho
    T = np.linalg.inv(np.eye(n) - G)
    d = rng.uniform(0.5, 1.0, n)
    X = (-d[:, None] * T).T + rng.normal(scale=sigma, size=(n, n))
    return G, X, -np.diag(d)


def test_ace_from_xu_normalises_by_own_response():
    G, X, U = linear_screen()
    genes, R, own = ace_from_xu(X, U)
    assert list(genes) == list(range(30))
    assert np.allclose(R[np.arange(30), genes], 1.0)
    assert np.allclose(own, np.diag(X))


def test_inspre_recovers_a_noise_free_cyclic_network():
    G, X, U = linear_screen()
    res = InspreAlgorithm(k_interactions=3).fit(X, U)
    assert compute_aupr(G != 0, res.A_est, ignore_diagonal=True) > 0.99
    assert res.diagnostics["spectral_radius"] < 1


def test_inspre_dense_solution_is_the_matrix_inverse():
    G, X, U = linear_screen()
    _, R, _ = ace_from_xu(X, U)
    V, _, _ = inspre_solve(R, lam=0.0, max_iter=500, tol=1e-9)
    assert np.allclose(direct_effects(V), G, atol=1e-3)


def test_inspre_edge_penalty_moves_the_support_towards_the_prior():
    G, X, U = linear_screen(sigma=0.05, seed=3)
    base = InspreAlgorithm(k_interactions=3, density_factor=1.0).fit(X, U).A_est
    prior = (G != 0)
    guided = InspreAlgorithm(k_interactions=3, density_factor=1.0).fit(X, U, edge_penalty=prior_penalty(prior, 0.95)).A_est
    assert compute_aupr(prior, guided, ignore_diagonal=True) >= compute_aupr(prior, base, ignore_diagonal=True)


def test_unperturbed_genes_get_zero_rows_and_columns():
    G, X, U = linear_screen()
    keep = np.arange(0, 30, 2)
    res = InspreAlgorithm(k_interactions=3).fit(X[:, keep], U[:, keep])
    unp = np.setdiff1d(np.arange(30), keep)
    assert not res.A_est[unp].any() and not res.A_est[:, unp].any()
    assert res.diagnostics["nodes"] == len(keep)


def test_baselines_run_and_have_the_documented_orientation():
    G, X, U = linear_screen()
    for algo in (MeanDifference(3), PearsonCorrelation(3), GeneFeatureBaseline(3)):
        res = algo.fit(X, U)
        assert res.A_est.shape == (30, 30) and not np.diag(res.A_est).any()
    gb = GRNBoostLite(3, n_estimators=10, n_jobs=1).fit(X, U)
    assert gb.A_est.shape == (30, 30)


def test_mean_difference_row_is_the_response_of_the_perturbed_gene():
    G, X, U = linear_screen()
    md = MeanDifference(3).fit(X, U).A_est
    for a in (0, 7, 19):
        expect = X[:, a].copy()
        expect[a] = 0
        assert np.allclose(md[a], expect)


def test_registered_with_the_orientation_they_return():
    keys = registry.keys()
    for k in ("inspre", "mean_difference", "pearson", "gene_features", "grnboost"):
        assert k in keys
    from grn_sim.inference import orientation
    assert not orientation.output_is_target_regulator("inspre", orientation.HILL)
    assert orientation.output_is_target_regulator("grnboost", orientation.HILL)
    algo = registry.build("inspre", registry.BuildContext(k_est=3))
    assert isinstance(algo, InspreAlgorithm)


def test_bicycle_lite_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    p = 5
    G = (rng.random((p, p)) < 0.3) * rng.normal(size=(p, p)) * 0.3
    np.fill_diagonal(G, 0)
    A = rng.normal(size=(p, p))
    prob = _Problem(rng.normal(size=(p, p)), np.full((p, p), 0.3), np.ones(p), A @ A.T / p + np.eye(p), 400, 1.0)
    prob.nodes = np.arange(p)
    ld, lt = np.log(rng.uniform(0.1, 0.5, p)), np.log(rng.uniform(0.1, 0.5, p))
    _, gG, gd, gt = prob.value_grad(G, ld, lt)

    def num(f, x, eps=1e-6):
        g = np.zeros_like(x)
        for i in np.ndindex(x.shape):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (f(xp) - f(xm)) / (2 * eps)
        return g

    ng = num(lambda x: prob.value_grad(x, ld, lt)[0], G)
    np.fill_diagonal(ng, 0)
    assert np.allclose(ng, gG, rtol=1e-4, atol=1e-4)
    assert np.allclose(num(lambda x: prob.value_grad(G, x, lt)[0], ld), gd, rtol=1e-4, atol=1e-4)
    assert np.allclose(num(lambda x: prob.value_grad(G, ld, x)[0], lt), gt, rtol=1e-4, atol=1e-4)


def test_consensus_rank_normalises_scales_away():
    rng = np.random.default_rng(1)
    S = rng.random((8, 8))
    c = consensus({"a": S, "b": 1000 * S})
    assert np.allclose(c, rank_normalise(S))
    rows = np.array([True] * 4 + [False] * 4)
    assert not rank_normalise(S, rows)[4:].any()


def test_make_prior_has_the_requested_sensitivity_and_precision():
    rng = np.random.default_rng(2)
    n = 60
    direct = np.zeros((n, n), dtype=bool)
    idx = rng.choice(n * n, 200, replace=False)
    direct.ravel()[idx] = True
    np.fill_diagonal(direct, False)
    rows = np.ones(n, dtype=bool)
    prior = make_prior(direct, rows, 0.5, 0.25, rng)
    tp = (prior & direct).sum()
    assert abs(tp / direct.sum() - 0.5) < 0.05
    assert abs(tp / prior.sum() - 0.25) < 0.05
    none = make_prior(direct, rows, 0.5, 0.0, rng)
    assert (none & direct).sum() == 0


@pytest.fixture(scope="module")
def small_screen():
    from grn_sim.perturbseq.pipeline import build_network, run_screen
    return run_screen(build_network("replogle_like", 30, seed=1), mode="knockdown", cells_per_condition=40, seed=1)


def test_ace_estimate_detects_on_target_knockdown_and_is_calibrated(small_screen):
    from grn_sim.perturbseq.ace import estimate_ace
    run = small_screen
    ace = estimate_ace(run.adata, gene_names=list(run.screen.gene_names), seed=0)
    assert len(ace.nodes) >= 25
    effective = np.abs(ace.own_z) >= 2
    assert effective.mean() > 0.6
    assert (ace.own[effective] < 0).mean() > 0.9                       # CRISPRi lowers its own gene
    assert 0.5 <= ace.null_scale.min() and ace.null_scale.max() <= 3.0
    # off-target p-values are mostly not significant; the screen has a minority of real effects
    off = np.ones_like(ace.pvalue, dtype=bool)
    off[np.arange(len(ace.nodes)), ace.nodes] = False
    assert (ace.qvalue[off] <= 0.05).mean() < 0.6
    assert ace.cov.shape == (30, 30) and ace.n_cov > 1000
    X, U = ace.to_xu()
    assert X.shape == (30, len(ace.nodes)) and np.count_nonzero(U) == len(ace.nodes)


def test_cell_level_methods_run_on_a_screen(small_screen):
    from grn_sim.evaluation import screen_benchmark as sb
    from grn_sim.inference.bicycle_lite import BicycleLite
    from grn_sim.inference.dotears import DOTEARSLite
    run = small_screen
    ace = sb.screen_inputs(run, "cells", 0).denoised(0.05)
    for res in (InspreAlgorithm(3).fit_ace(ace, alpha=None), BicycleLite(3, max_iter=15).fit_ace(ace),
                DOTEARSLite(3, max_outer=3).fit_adata(run.adata, ace)):
        S = np.abs(res.A_est)
        assert S.shape == (30, 30) and np.isfinite(S).all()
        metrics = sb.evaluate_scores(run.screen, S)
        assert metrics["aupr_direct"] > metrics["base_direct"]         # better than picking pairs at random
    truth = sb.evaluate_scores(run.screen, sb.score_with_algorithm(MeanDifference(3), sb.screen_inputs(run, "truth")))
    assert truth["aupr_total"] > 0.99                                  # mean difference ranks total effects perfectly


def test_ridge_cv_additive_default_includes_the_own_gene_term():
    from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
    G, X, U = linear_screen(n=40, seed=1)
    default = KaggleMultiOutputRidge(k_interactions=3).fit(X, U).A_est.T
    no_self = KaggleMultiOutputRidge(k_interactions=3, allow_self_regulation=False).fit(X, U).A_est.T
    assert compute_aupr(G != 0, default, ignore_diagonal=True) > 0.7
    assert compute_aupr(G != 0, no_self, ignore_diagonal=True) < 0.2
    # the regulator-input model has no self term: the default there is unchanged
    reg = KaggleMultiOutputRidge(k_interactions=3, perturbation_mode="regulator")
    assert reg.allow_self_regulation is None
