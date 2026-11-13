"""The all-targets ridge solver equals one sklearn RidgeCV per target (coefficients, alphas, adjacency)."""
import numpy as np
import pytest

from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge


def _data(n, m, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, m)), np.eye(n)[:, :m] * 2 + 0.1 * rng.normal(size=(n, m))


@pytest.mark.parametrize("n,m", [(30, 30), (40, 25)])
@pytest.mark.parametrize("mode", ["additive", "regulator"])
@pytest.mark.parametrize("self_reg", [False, True])
def test_fast_matches_the_per_target_fit(n, m, mode, self_reg):
    X, U = _data(n, m, seed=n + m)
    kw = dict(perturbation_mode=mode, allow_self_regulation=self_reg, k_interactions=3)
    ref = KaggleMultiOutputRidge(fast=False, n_jobs=1, **kw).fit(X, U)
    fast = KaggleMultiOutputRidge(fast=True, **kw).fit(X, U)
    assert np.allclose(ref.A_est, fast.A_est, atol=1e-8)
    assert np.array_equal(ref.adjacency_pred, fast.adjacency_pred)
    assert ref.diagnostics["mean_alpha"] == fast.diagnostics["mean_alpha"]


def test_options_that_need_per_target_work_use_the_reference_path():
    X, U = _data(20, 20, 1)
    res = KaggleMultiOutputRidge(adaptive_k=True, k_interactions=3, perturbation_mode="additive").fit(X, U)
    assert (res.adjacency_pred != 0).sum() > 0
