"""
Baselines that every perturbation-based network method has to beat before its extra machinery is worth reporting.

On the CausalBench K562 / RPE1 challenge, a plain control-versus-perturbed mean difference tied or beat the elaborate
methods, and neutral benchmarks keep finding that correlation or tree-importance scores match deep and causal ones.
These four are the controls:

* ``MeanDifference``: score(a -> j) = |response of gene j when node a is perturbed| (the unnormalised ACE);
* ``PearsonCorrelation``: |corr| of two genes across the conditions (symmetric, no use of the design);
* ``GRNBoostLite``: per-target gradient-boosted trees on the other genes' responses (GRNBoost2-style importances);
* ``GeneFeatureBaseline``: score(a -> j) = (how much node a moves anything) x (how much gene j moves under anything),
  i.e. what is explained by gene-level hubness alone and not by the specific pair.

All take the repository's (X, U) conditions. Output: ``A_est[regulator, target]`` except ``GRNBoostLite``
(``A_est[target, regulator]``, like GENIE3).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from grn_sim.inference._parallel import default_n_jobs, map_indices
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.inspre import ace_from_xu


def _result(name: str, A: np.ndarray, k: int, t0: float, **diag) -> InferenceResult:
    n = A.shape[0]
    adj = np.zeros((n, n), dtype=int)
    mags = np.abs(A)
    for r in range(n):
        top = np.argsort(-mags[r])[:k]
        adj[r, top[mags[r, top] > 0]] = np.sign(A[r, top[mags[r, top] > 0]]).astype(int)
    return InferenceResult(name, A, adj, time.time() - t0, dict(diag))


def _node_effects(X: np.ndarray, U: np.ndarray):
    """(genes, E, own): E[a, j] = raw response of gene j to node a (genes[a] = the perturbed gene)."""
    genes, R, own = ace_from_xu(X, U)
    E = R * own[:, None]
    return genes, E, own


class MeanDifference(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3, normalise: bool = False):
        super().__init__(name="mean_difference")
        self.k_interactions = k_interactions
        self.normalise = normalise

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        t0 = time.time()
        n = np.asarray(X).shape[0]
        genes, E, own = _node_effects(X, U)
        if self.normalise:
            with np.errstate(divide="ignore", invalid="ignore"):
                E = np.where(own[:, None] != 0, E / own[:, None], 0.0)
        A = np.zeros((n, n))
        A[genes] = E
        np.fill_diagonal(A, 0.0)
        return _result(self.name, A, int(kwargs.get("k", self.k_interactions)), t0, nodes=int(len(genes)))


class PearsonCorrelation(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3):
        super().__init__(name="pearson")
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        t0 = time.time()
        X = np.real(np.asarray(X, dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(X)
        C = np.nan_to_num(C)
        np.fill_diagonal(C, 0.0)
        return _result(self.name, C, int(kwargs.get("k", self.k_interactions)), t0)


class GRNBoostLite(BaseInferenceAlgorithm):
    """Gradient-boosted-tree importances per target (the GRNBoost2 recipe with scikit-learn's boosting)."""

    def __init__(self, k_interactions: int = 3, n_estimators: int = 60, max_features: float = 0.1,
                 seed: int = 0, n_jobs: Optional[int] = None):
        super().__init__(name="grnboost")
        self.k_interactions = k_interactions
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.seed = seed
        self.n_jobs = n_jobs

    def _target(self, payload, i: int) -> np.ndarray:
        X, n_est, mf, seed = payload
        n = X.shape[0]
        reg = np.array([j for j in range(n) if j != i])
        model = GradientBoostingRegressor(n_estimators=n_est, max_depth=3, learning_rate=0.1, subsample=0.9,
                                          max_features=mf, random_state=seed + i)
        model.fit(X[reg].T, X[i])
        imp = model.feature_importances_
        out = np.zeros(n)
        out[reg] = imp / imp.sum() if imp.sum() > 0 else imp
        return out

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        t0 = time.time()
        X = np.real(np.asarray(X, dtype=float))
        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=16)
        rows = map_indices(self._target, (X, self.n_estimators, self.max_features, self.seed), X.shape[0], n_jobs)
        W = np.vstack(rows)                          # W[target, regulator]
        return _result(self.name, W, int(kwargs.get("k", self.k_interactions)), t0)


class GeneFeatureBaseline(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3):
        super().__init__(name="gene_features")
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        t0 = time.time()
        n = np.asarray(X).shape[0]
        genes, E, own = _node_effects(X, U)
        A_abs = np.abs(E)
        a = A_abs.sum(axis=1)                         # how much each node moves anything
        b = np.abs(np.real(np.asarray(X, dtype=float))).sum(axis=1)   # how much each gene moves under anything
        A = np.zeros((n, n))
        A[genes] = a[:, None] * b[None, :]
        np.fill_diagonal(A, 0.0)
        return _result(self.name, A, int(kwargs.get("k", self.k_interactions)), t0)
