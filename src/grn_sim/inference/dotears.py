"""
DOTEARS-style interventional DAG learning (Squires et al. 2023 build on NOTEARS; Zheng et al. 2018).

A linear structural model  y_j = sum_i W[i, j] y_i + e_j  with a differentiable acyclicity constraint
h(W) = tr(exp(W o W)) - p = 0, as in NOTEARS, with two changes that matter for perturbation screens:

* **hard interventions**: a cell in which gene a is perturbed does not contribute to the equation of gene a (its value
  was set, not caused by the parents), so each gene's squared error is summed only over the cells where it is not
  intervened on. The pooled sufficient statistics (``S_total`` and one per node) make each evaluation independent of the
  number of cells;
* **per-gene noise variances** ``sigma_j^2`` are re-estimated from the residuals, so a small-variance gene is not
  swamped by a noisy one, the failure of plain NOTEARS (whose loss assumes equal variances).

It is a DAG model: feedback loops in the truth cannot be represented and cost it accuracy. Free entries of W are limited
to the strongest inspre candidates (``candidates_per_node``; ``None`` frees every pair), which also keeps the exponential in the constraint cheap
(cubic in the number of perturbed genes: practical to a few hundred). The loss is the mean over cells.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.inspre import InspreAlgorithm
from grn_sim.perturbseq.ace import log_normalise


def dotears_statistics(adata: Any, ace: Any, perturbation_key: str = "perturbation",
                       control_label: str = "non-targeting") -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    ``(S_total, S_node, n_node, N)``: second-moment matrices of the control-centred expression over all cells
    (restricted to the perturbed genes), per node over its own cells, and the cell counts.
    """
    lab = adata.obs[perturbation_key].astype(str).values
    Y = log_normalise(adata.X).astype(np.float64)
    ctrl = lab == control_label
    Y = Y - Y[ctrl].mean(axis=0)
    names = [str(g) for g in ace.gene_names]
    node_names = [names[i] for i in ace.nodes]
    Yn = Y[:, ace.nodes]
    keep = ctrl | np.isin(lab, node_names)
    Yn, lab = Yn[keep], lab[keep]
    S_tot = Yn.T @ Yn
    p = len(node_names)
    S_node = np.zeros((p, p, p))
    n_node = np.zeros(p, dtype=int)
    for a, nm in enumerate(node_names):
        sel = lab == nm
        n_node[a] = int(sel.sum())
        if n_node[a]:
            S_node[a] = Yn[sel].T @ Yn[sel]
    return S_tot, S_node, n_node, int(len(Yn))


class DOTEARSLite(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3, lam: float = 0.02, candidates_per_node: Optional[int] = 12,
                 max_outer: int = 8, h_tol: float = 1e-7, density_factor: float = 2.0):
        super().__init__(name="dotears_lite")
        self.k_interactions = k_interactions
        self.lam = lam
        self.candidates_per_node = candidates_per_node
        self.max_outer = max_outer
        self.h_tol = h_tol
        self.density_factor = density_factor

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        raise NotImplementedError("DOTEARSLite needs cell-level second moments: use fit_adata(adata, ace)")

    def fit_adata(self, adata: Any, ace: Any, alpha: Optional[float] = 0.05, min_own_z: float = 2.0,
                  **kwargs) -> InferenceResult:
        t0 = time.time()
        k = int(kwargs.get("k", self.k_interactions))
        n = ace.lfc.shape[1]
        genes = ace.nodes
        p = len(genes)
        S_tot, S_node, n_node, N = dotears_statistics(adata, ace)
        est = ace.denoised(alpha) if alpha is not None else ace
        ok = np.abs(est.own_z) >= min_own_z
        R = np.zeros_like(est.lfc)
        own = est.own
        R[ok] = est.lfc[ok] / own[ok, None]
        ins = InspreAlgorithm(k_interactions=k, density_factor=self.density_factor)
        score = np.abs(ins._solve(n, genes, R, ok.astype(float), k=k * 4).A_est)[np.ix_(genes, genes)]
        cand = np.zeros((p, p), dtype=bool)
        if self.candidates_per_node is None:             # no restriction: every pair is a free parameter
            cand[:] = True
        else:
            for a in range(p):
                top = np.argsort(-score[a])[: self.candidates_per_node]
                cand[a, top[score[a, top] > 0]] = True
        np.fill_diagonal(cand, False)
        m = p * p
        # per-gene (target) second moments: S_j = S_total - S_node(j), N_j = N - n_node(j); the genes here are the nodes
        S_j = S_tot[None, :, :] - S_node                     # (p, p, p), index = target
        N_j = (N - n_node).astype(float)
        sig2 = np.maximum(np.diag(S_tot) / N, 1e-6)
        lam = self.lam
        W = np.zeros((p, p))
        alpha_l, rho = 0.0, 1.0

        def loss_grad(Wm, s2):
            # target j: ||y_j - W[:, j].y||^2 summed over cells where j is not intervened = S_jj - 2 w.S_j[:, j] + w.S_j w
            SW = np.einsum("jab,bj->ja", S_j, Wm)            # (p, p): S_j w_j per target
            sjj = S_j[np.arange(p), np.arange(p), np.arange(p)]
            col = S_j[np.arange(p), :, np.arange(p)]          # (p targets, p) = S_j[:, j]
            ss = sjj - 2.0 * np.sum(col * Wm.T, axis=1) + np.sum(Wm.T * SW, axis=1)
            val = float(np.sum(ss / (2.0 * s2 * N_j)))
            g = ((SW - col) / (s2 * N_j)[:, None]).T          # d/dW[:, j]
            return val, g, ss / N_j

        for outer in range(self.max_outer):
            def f(x):
                Wm = x[:m].reshape(p, p) - x[m:].reshape(p, p)
                val, g, _ = loss_grad(Wm, sig2)
                E = expm(Wm * Wm)
                h = float(np.trace(E) - p)
                gh = E.T * 2.0 * Wm
                val += lam * float(np.sum(x)) + alpha_l * h + 0.5 * rho * h * h
                gW = g + (alpha_l + rho * h) * gh
                np.fill_diagonal(gW, 0.0)
                return val, np.concatenate([(gW + lam).ravel(), (-gW + lam).ravel()])
            ub = np.where(np.tile(cand.ravel(), 2), np.inf, 0.0)
            x0 = np.concatenate([np.maximum(W, 0).ravel(), np.maximum(-W, 0).ravel()])
            res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, u) for u in ub],
                           options={"maxiter": 200})
            W = res.x[:m].reshape(p, p) - res.x[m:].reshape(p, p)
            _, _, resid = loss_grad(W, sig2)
            sig2 = np.maximum(resid, 1e-6)
            h = float(np.trace(expm(W * W)) - p)
            if h <= self.h_tol:
                break
            alpha_l += rho * h
            rho = min(rho * 10.0, 1e8)
        np.fill_diagonal(W, 0.0)
        A = np.zeros((n, n))
        A[np.ix_(genes, genes)] = W
        adj = np.zeros((n, n), dtype=int)
        for r in range(n):
            mags = np.abs(A[r])
            top = np.argsort(-mags)[:k]
            for c in top:
                if mags[c] > 1e-6:
                    adj[r, c] = int(np.sign(A[r, c]))
        diag: Dict[str, Any] = {"nodes": int(p), "acyclicity_h": h, "outer_rounds": outer + 1,
                                "edges": int(np.count_nonzero(np.abs(W) > 1e-6)), "candidates": int(cand.sum())}
        return InferenceResult(self.name, A, adj, time.time() - t0, diag)
