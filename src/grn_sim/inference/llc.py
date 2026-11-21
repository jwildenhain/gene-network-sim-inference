"""
LLC, mean-shift part (Hyttinen, Eberhardt, Hoyer 2012: learning linear cyclic causal models with latent variables).

For a linear cyclic model ``x = B^T x + e`` and a single-gene intervention on gene ``a`` the shift of the steady state is
``delta_a = (I - B^T)^-1 e_a s``, hence for every gene ``j != a``: ``delta_a[j] = sum_i B[i, j] delta_a[i]``. The shifts of
the *other* genes under experiment ``a`` are the regressors and the shift of ``j`` the response, so each target is one linear
system with one equation per experiment in which ``j`` was not perturbed. The confounder terms cancel in a shift, which is
what makes the equations valid with latent variables. LLC adds second-moment (covariance) constraints that need many cells
per experiment; they are not used here, so this is the mean-shift subset: with ridge regularisation it is the per-target
regression that NIR and RidgeCV also solve (see ``docs/results/paper_algorithms.md`` for the measured comparison).

``perturbation_mode="regulator"`` is for the Hill simulator's regulator-input perturbation (the perturbation is added to the
effective concentration, so the perturbed gene itself barely responds): the shifts are then those of ``X + U``, as in inspre.

Output: ``A_est[regulator, target]``.
"""
from __future__ import annotations

import time

import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.inspre import ace_from_xu


class LLCMean(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3, ridge: float = 0.1, perturbation_mode: str = "additive"):
        super().__init__(name="llc_mean")
        if perturbation_mode not in ("additive", "regulator"):
            raise ValueError(f"perturbation_mode must be 'additive' or 'regulator', got {perturbation_mode!r}")
        self.k_interactions = k_interactions
        self.ridge = ridge
        self.perturbation_mode = perturbation_mode

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        t0 = time.time()
        n = np.asarray(X).shape[0]
        if self.perturbation_mode == "regulator":
            X = np.real(np.asarray(X, dtype=float)) + np.real(np.asarray(U, dtype=float))
        genes, R, own = ace_from_xu(X, U)
        E = R * own[:, None]                              # E[a, j]: shift of gene j when node a is perturbed
        p = len(genes)
        G = E.T @ E                                       # Gram matrix of all experiments, once; a target drops its own rows
        B = np.zeros((n, n))
        for j in range(n):
            rows = np.flatnonzero(genes != j)
            if len(rows) < 2:
                continue
            others = np.delete(np.arange(n), j)
            Gj, gy = G.copy(), G[:, j].copy()
            for own_row in np.flatnonzero(genes == j):    # the experiment that perturbed j is not an equation for j
                Gj -= np.outer(E[own_row], E[own_row])
                gy -= E[own_row] * E[own_row, j]
            ZtZ = Gj[np.ix_(others, others)]
            lam = self.ridge * np.trace(ZtZ) / max(len(others), 1)
            reg = (lam + 1e-12) * np.eye(len(others))
            B[others, j] = np.linalg.solve(ZtZ + reg, gy[others])
        adj = np.zeros((n, n), dtype=int)
        mags = np.abs(B)
        for r in range(n):
            top = np.argsort(-mags[r])[: self.k_interactions]
            adj[r, top[mags[r, top] > 0]] = np.sign(B[r, top[mags[r, top] > 0]]).astype(int)
        return InferenceResult(self.name, B, adj, time.time() - t0, {"nodes": int(p), "perturbation_mode": self.perturbation_mode})
