"""
bicycle-lite: a steady-state Ornstein-Uhlenbeck likelihood for the direct-effect matrix, in the spirit of bicycle
(Rohbeck et al., CLeaR 2024) and Dictys.

Model. Expression follows  dx = -(I - G^T) x dt + b dt + (perturbation) dt + noise  with direct effects G[a, b] (a -> b).
At steady state

* the mean shift when node a is knocked down is  mu_a = -delta_a T[a, :],  T = (I - G)^-1 (the total effects),
* the cell-to-cell covariance solves the Lyapunov equation  M S + S M^T = D,  M = I - G^T,  D = diag(sigma^2),
  and what is measured adds independent technical noise  diag(tau^2).

The negative log-likelihood is the sum of a weighted least-squares fit of the mean shifts to the ACE estimate (standard
errors as weights, delta_a profiled out in closed form) and a Gaussian likelihood of the pooled covariance, plus an L1
penalty on G. It is minimised with L-BFGS-B over the entries of G (split in positive and negative parts) from an
inspre starting point and with the free entries limited to the strongest candidates.

What this is not. bicycle also models a latent (noise-free) layer with per-cell perturbation states and learns from
single cells directly; this version works with the sufficient statistics (per-node mean shifts and one pooled
covariance), keeps the same perturbation strength for all cells of a node, and shares one covariance between conditions.
The covariance ignores that a knockdown changes the targeted gene's own variance. Needs the ACE estimate to carry the
pooled covariance (``estimate_ace`` provides it); without it only the mean term is used.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.inspre import InspreAlgorithm, ace_from_xu, spectral_radius


class _Problem:
    def __init__(self, lfc: np.ndarray, se: np.ndarray, node_ok: np.ndarray, cov: Optional[np.ndarray], n_cov: int,
                 cov_weight: float):
        self.lfc, self.se2, self.ok = lfc, se ** 2, node_ok
        self.cov, self.n_cov, self.cov_weight = cov, n_cov, cov_weight
        self.p, self.n = lfc.shape
        self.nodes = np.arange(self.p)            # row a of ``lfc`` is the perturbation of gene nodes[a] (square problem: itself)

    def value_grad(self, G: np.ndarray, log_d: np.ndarray, log_tau: np.ndarray):
        n = self.n
        eye = np.eye(n)
        M = eye - G.T
        if np.min(np.linalg.eigvals(M).real) < 0.02:               # unstable feedback: outside the model
            return 1e12, np.zeros_like(G), np.zeros(n), np.zeros(n)
        T = np.linalg.inv(eye - G)                                    # T[a, b] = total effect a -> b
        Tn = T[self.nodes]                                          # (p, n)
        W = self.ok[:, None] / self.se2
        num = np.sum(W * self.lfc * Tn, axis=1)
        den = np.sum(W * Tn * Tn, axis=1) + 1e-12
        delta = -num / den                                          # mu_a = -delta_a T_a
        mu = -delta[:, None] * Tn
        r = self.lfc - mu
        val = 0.5 * np.sum(W * r * r)
        Q = np.zeros((n, n))
        Q[self.nodes] = delta[:, None] * (W * r)                    # dL/dT
        gG = T.T @ Q @ T.T
        g_ld = np.zeros(n)
        g_lt = np.zeros(n)
        if self.cov is not None and self.cov_weight > 0:
            d, tau = np.exp(log_d), np.exp(log_tau)
            S = solve_continuous_lyapunov(M, np.diag(d))
            S = 0.5 * (S + S.T)
            Sig = S + np.diag(tau)
            try:
                L = np.linalg.cholesky(Sig)
            except np.linalg.LinAlgError:
                return 1e12, np.zeros_like(G), np.zeros(n), np.zeros(n)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            Si = np.linalg.inv(Sig)
            c = 0.5 * self.n_cov * self.cov_weight
            val += c * (logdet + np.sum(Si * self.cov))
            Gs = c * (Si - Si @ self.cov @ Si)
            Lam = solve_continuous_lyapunov(M.T, Gs)
            Lam = 0.5 * (Lam + Lam.T)
            gG += 2.0 * S @ Lam
            g_ld = np.diag(Lam) * d
            g_lt = np.diag(Gs) * tau
        np.fill_diagonal(gG, 0.0)
        return float(val), gG, g_ld, g_lt


class BicycleLite(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3, lam: float = 5.0, candidates_per_node: Optional[int] = 12,
                 cov_weight: float = 1.0, max_iter: int = 120, density_factor: float = 2.0):
        super().__init__(name="bicycle_lite")
        self.k_interactions = k_interactions
        self.lam = lam
        self.candidates_per_node = candidates_per_node
        self.cov_weight = cov_weight
        self.max_iter = max_iter
        self.density_factor = density_factor

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """Mean-shift term only (an (X, U) pair carries no covariance); standard errors are taken as equal."""
        genes, R, own = ace_from_xu(X, U)
        n = np.asarray(X).shape[0]
        lfc = R * own[:, None]
        return self._fit(n, genes, lfc, np.ones_like(lfc), own != 0, None, 0, **kwargs)

    def fit_ace(self, ace: Any, alpha: Optional[float] = None, min_own_z: float = 2.0, **kwargs) -> InferenceResult:
        est = ace.denoised(alpha) if alpha is not None else ace
        ok = np.abs(est.own_z) >= min_own_z
        return self._fit(est.lfc.shape[1], est.nodes, est.lfc, est.se, ok, est.cov, est.n_cov, **kwargs)

    def _fit(self, n: int, genes: np.ndarray, lfc: np.ndarray, se: np.ndarray, ok: np.ndarray,
             cov: Optional[np.ndarray], n_cov: int, **kwargs) -> InferenceResult:
        t0 = time.time()
        k = int(kwargs.get("k", self.k_interactions))
        p = len(genes)
        A = np.zeros((n, n))
        diag: Dict[str, Any] = {"nodes": int(p), "uses_covariance": cov is not None}
        if p >= 3:
            # starting point and candidate set from inspre on the same ACE estimate
            ins = InspreAlgorithm(k_interactions=k, density_factor=self.density_factor)
            R = np.zeros_like(lfc)
            own = lfc[np.arange(p), genes]
            R[ok] = lfc[ok] / own[ok, None]
            w = ok.astype(float)
            G0 = ins._solve(n, genes, R, w, k=k, **{}).A_est
            G0n = G0[np.ix_(genes, genes)] if p < n else G0
            score = np.abs(ins._solve(n, genes, R, w, k=k * 4).A_est)[np.ix_(genes, genes)]
            cand = np.zeros((p, p), dtype=bool)
            if self.candidates_per_node is None:         # no restriction: every pair is a free parameter
                cand[:] = True
            else:
                for a in range(p):
                    top = np.argsort(-score[a])[: self.candidates_per_node]
                    cand[a, top[score[a, top] > 0]] = True
            np.fill_diagonal(cand, False)
            cand |= (G0n != 0)
            r0 = spectral_radius(G0n)
            if r0 > 0.9:
                G0n = G0n * (0.9 / r0)
            # restrict the model to the p perturbed genes (square problem)
            lfc_n = lfc[:, genes]
            se_n = se[:, genes]
            cov_n = cov[np.ix_(genes, genes)] if cov is not None else None
            prob = _Problem(lfc_n, se_n, ok.astype(float), cov_n, n_cov, self.cov_weight)
            prob.nodes = np.arange(p)
            m = p * p
            x0 = np.concatenate([np.maximum(G0n, 0).ravel(), np.maximum(-G0n, 0).ravel(),
                                 np.full(p, np.log(0.05)), np.full(p, np.log(0.05))])
            if cov_n is not None:
                dg = np.diag(cov_n)
                x0[2 * m:2 * m + p] = np.log(np.maximum(0.3 * dg, 1e-4))
                x0[2 * m + p:] = np.log(np.maximum(0.7 * dg, 1e-4))
            ub = np.where(np.tile(cand.ravel(), 2), np.inf, 0.0)
            bounds = [(0.0, u) for u in ub] + [(-12.0, 4.0)] * (2 * p)
            lam = self.lam

            def f(x):
                Gp = x[:m].reshape(p, p)
                Gq = x[m:2 * m].reshape(p, p)
                G = Gp - Gq
                val, gG, gld, glt = prob.value_grad(G, x[2 * m:2 * m + p], x[2 * m + p:])
                val += lam * float(np.sum(x[:2 * m]))
                grad = np.concatenate([(gG + lam).ravel(), (-gG + lam).ravel(), gld, glt])
                return val, grad

            res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": self.max_iter, "maxfun": self.max_iter * 2})
            G = res.x[:m].reshape(p, p) - res.x[m:2 * m].reshape(p, p)
            np.fill_diagonal(G, 0.0)
            A[np.ix_(genes, genes)] = G
            diag.update(iterations=int(res.nit), objective=float(res.fun), spectral_radius=spectral_radius(G),
                        edges=int(np.count_nonzero(np.abs(G) > 1e-6)), candidates=int(cand.sum()), converged=bool(res.success))
        adj = np.zeros((n, n), dtype=int)
        for r in range(n):
            mags = np.abs(A[r])
            top = np.argsort(-mags)[:k]
            for c in top:
                if mags[c] > 1e-6:
                    adj[r, c] = int(np.sign(A[r, c]))
        return InferenceResult(self.name, A, adj, time.time() - t0, diag)
