"""LLC (mean-shift equations, registered as ``llc_mean``) and DCD-FG-lite (``grn_sim.inference.experimental``, not registered)."""
import numpy as np
import pytest

from grn_sim.inference import registry
from grn_sim.inference.experimental import DCDFGLite
from grn_sim.inference.llc import LLCMean


def test_only_the_cell_level_algorithm_stays_unregistered():
    assert "llc_mean" in registry.keys() and "dcdfg_lite" not in registry.keys()


def _linear_cyclic(n=8, seed=0):
    rng = np.random.default_rng(seed)
    B = np.zeros((n, n))                                          # B[regulator, target], includes a 2-cycle
    for i, j in [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (6, 7), (7, 6), (1, 5)]:
        B[i, j] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.5)
    return B


def test_llc_mean_recovers_a_noise_free_linear_cyclic_model():
    n = 8
    B = _linear_cyclic(n)
    T = np.linalg.inv(np.eye(n) - B.T)                           # shift of every gene per unit intervention on a gene
    X = T.copy()                                                 # column a = response to perturbing gene a
    U = np.eye(n)
    A = LLCMean(ridge=1e-9).fit(X, U).A_est
    assert np.allclose(A, B, atol=1e-3)


def test_llc_mean_scores_true_edges_above_the_rest_with_noisy_responses():
    n = 8
    B = _linear_cyclic(n)
    T = np.linalg.inv(np.eye(n) - B.T) + np.random.default_rng(1).normal(scale=0.02, size=(n, n))
    A = np.abs(LLCMean().fit(T, np.eye(n)).A_est)
    off = ~np.eye(n, dtype=bool)
    assert A[(B != 0) & off].min() > np.median(A[(B == 0) & off])


def test_dcdfg_lite_fits_cells_and_returns_a_finite_near_acyclic_graph():
    anndata = pytest.importorskip("anndata")
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    n, per = 6, 120
    names = [f"G{i}" for i in range(n)]
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 0.9                                         # a chain G0 -> G1 -> ... -> G5
    rows, labels = [], []
    for g in [None] + list(range(n)):
        for _ in range(per):
            x = np.zeros(n)
            for j in range(n):                                    # topological order = index order
                x[j] = rng.normal() if j == g else (x @ W[:, j] + rng.normal(scale=0.5))
                if j == g:
                    x[j] -= 2.0                                   # knockdown: the mechanism is replaced
            rows.append(np.exp(np.clip(x, -3, 3)) * 5.0)
            labels.append("non-targeting" if g is None else names[g])
    ad = anndata.AnnData(np.asarray(rows, dtype=np.float32))
    ad.var_names = names
    ad.obs["perturbation"] = labels
    res = DCDFGLite(rank=4, outer=3, inner=150).fit_adata(ad, names)
    assert np.all(np.isfinite(res.A_est)) and res.A_est.shape == (n, n)
    assert np.allclose(np.diag(res.A_est), 0.0)
    assert res.diagnostics["acyclicity_h"] < 1.0
