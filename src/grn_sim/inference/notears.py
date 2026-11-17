"""
NOTEARS: Continuous Optimization for Directed Acyclic Graphs (DAGs).
Based on Zheng et al. (NeurIPS 2018, "DAGs with NO TEARS: Continuous Optimization for Directed Acyclic Graphs").
Enhanced with interventional equilibrium data preparation and soft-acyclicity compensation.
"""

import time
from typing import Optional
import numpy as np
from grn_sim.inference._parallel import report_progress
from scipy.linalg import expm

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


class NOTEARSAlgorithm(BaseInferenceAlgorithm):
    """
    NOTEARS: Differentiable, continuous optimization for directed causal network structure learning.

    Replaces discrete combinatorial DAG search with smooth mathematical optimization:
        min_W  (1 / 2M) * ||(I - W) X + U||_F^2 + lambda * ||W||_1
        s.t.   h(W) = Tr(exp(W o W)) - d = 0

    Solves the continuous constrained formulation using an Augmented Lagrangian method.

    The L1 penalty is relative to the data: ``lambda1=None`` (the default) uses ``lambda_fractions[0]`` times the smallest
    penalty that keeps every weight at zero (the lasso's lambda_max, read off the gradient at W = 0), so it scales with
    whatever units and size the data have. A fixed value cannot do that: 0.02 shrank whole networks to a few edges on the
    mechanistic data, and 0.0005 let the solver overflow (``Tr(exp(W o W))`` becomes inf, the weights NaN) on the
    statistical screens, which returned an empty graph that scored as chance. A fit that overflows, or that comes out
    dense (more than ``max_density`` x N non-zero weights, the sign of a penalty below the sparsity transition), is
    refitted with the next, larger fraction. Pass a number as ``lambda1`` to fix the penalty (no refits).
    """

    def __init__(
        self,
        lambda1: Optional[float] = None,
        max_iter: int = 25,
        h_tol: float = 1e-4,
        rho_max: float = 1000.0,
        w_threshold: float = 0.05,
        k_interactions: Optional[int] = None,
        use_data_prep: bool = True,
        lambda_fractions: tuple = (0.1, 0.2, 0.4, 0.8),
        max_density: float = 5.0,
    ):
        super().__init__(name="notears")
        self.lambda1 = lambda1
        self.lambda_fractions = tuple(lambda_fractions)
        self.max_density = float(max_density)
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.k_interactions = k_interactions
        self.use_data_prep = use_data_prep

    RESCALE_ABOVE = 20.0  # data magnitude beyond which fit() rescales (see there)

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        use_prep = kwargs.get("use_data_prep", self.use_data_prep)
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        w_thresh = float(kwargs.get("w_threshold", self.w_threshold))

        # On engines whose expression spans thousands (linear Jacobian, CellBox) the genes live on wildly different
        # scales and the proximal-gradient solve never leaves W = 0: the result was an empty graph. Standardise each gene
        # (dX_i/s_i, U_i/s_i); the weights then relate as W_ij(scaled) = W_ij s_j / s_i and are mapped back at the end.
        # Ordinary data (spread below RESCALE_ABOVE) is left exactly as it was.
        gene_scale = np.ones(N)
        if M > 1 and float(np.max(np.abs(X_mat - X_mat[:, :1]))) > self.RESCALE_ABOVE:
            spread = np.std(X_mat, axis=1)
            gene_scale = np.where(spread > 1e-9, spread, 1.0)
            X_mat = X_mat / gene_scale[:, None]
            U_mat = U_mat / gene_scale[:, None]

        if use_prep:
            # 1. Data Preparation: Baseline shift Delta X = X - x0 & interventional alignment
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            Delta_X = prep.Delta_X
            C = (1.0 / max(1, M)) * (Delta_X @ Delta_X.T)
            B = (1.0 / max(1, M)) * (Delta_X + U_mat) @ Delta_X.T
            norm_C = float(np.linalg.norm(C, 2)) + 1e-4
            eff_rho_max = min(self.rho_max, 1000.0)
            lambda_max = float(np.max(np.abs(B - np.diag(np.diag(B))))) if N > 1 else 0.0

            def _grad(W, rho, alpha):
                # Interventional equilibrium gradient: nabla_W F(W) = W @ C - B
                grad_F = W @ C - B
                M_sq = np.clip(W * W, 0.0, 10.0)
                E = expm(M_sq)
                h_val = float(np.trace(E) - N)
                grad_h = 2.0 * (E.T * W)
                grad_tot = grad_F + (alpha + rho * h_val) * grad_h
                return np.clip(grad_tot, -10.0, 10.0), h_val
        else:
            # Classical observational SEM formulation
            X_centered = X_mat - np.mean(X_mat, axis=1, keepdims=True)
            C = (1.0 / max(1, M)) * (X_centered @ X_centered.T)
            norm_C = float(np.linalg.norm(C, 2)) + 1e-4
            eff_rho_max = self.rho_max
            lambda_max = float(np.max(np.abs(C - np.diag(np.diag(C))))) if N > 1 else 0.0

            def _grad(W, rho, alpha):
                grad_F = C @ W - C
                M_sq = np.clip(W * W, 0.0, 10.0)
                E = expm(M_sq)
                h_val = float(np.trace(E) - N)
                grad_h = 2.0 * (E.T * W)
                grad_tot = grad_F + (alpha + rho * h_val) * grad_h
                return np.clip(grad_tot, -10.0, 10.0), h_val

        def _h(W):
            """Acyclicity constraint: Tr(exp(W o W)) - N."""
            M_sq = np.clip(W * W, 0.0, 10.0)
            E = expm(M_sq)
            return float(np.trace(E) - N)

        step_size = 0.05 / norm_C

        def _solve(lam: float, base: float, span: float):
            """One augmented-Lagrangian fit at penalty ``lam`` (progress base..base+span); (W, converged-finite)."""
            W = np.zeros((N, N), dtype=float)
            rho = 1.0
            alpha = 0.0
            h_prev = np.inf
            for outer in range(self.max_iter):
                report_progress(base + span * outer / self.max_iter)      # upper bound: it stops early once h(W) is small
                # Gradient descent inner loop
                for _ in range(15):
                    grad_W, h_val = _grad(W, rho, alpha)
                    np.fill_diagonal(grad_W, 0.0)

                    # Proximal gradient step for L1 regularization
                    W_step = W - step_size * grad_W
                    lam_step = lam * step_size
                    W = np.sign(W_step) * np.maximum(0.0, np.abs(W_step) - lam_step)
                    W = np.clip(W, -3.0, 3.0)
                    np.fill_diagonal(W, 0.0)

                # Check constraint violation
                h_val = _h(W)
                if not np.isfinite(h_val) or not np.all(np.isfinite(W)):
                    return W, False                         # overflow: Tr(exp(W o W)) left the float range
                if abs(h_val) <= self.h_tol or rho >= eff_rho_max:
                    break
                if h_val > 0.25 * h_prev:
                    rho = min(eff_rho_max, rho * 2.0)
                h_prev = h_val
                alpha += rho * h_val
            return W, True

        if self.lambda1 is not None:
            lambdas = [float(self.lambda1)]                 # a fixed penalty: one fit, taken as it is
        else:
            lambdas = [f * lambda_max for f in self.lambda_fractions] if lambda_max > 0 else [0.0]
        max_edges = max(50, int(self.max_density * N))
        tries = []
        W = np.zeros((N, N), dtype=float)
        lam_used = lambdas[-1]
        for fit_no, lam in enumerate(lambdas):
            W_try, ok = _solve(lam, fit_no / len(lambdas), 1.0 / len(lambdas))     # the path is an upper bound too
            report_progress((fit_no + 1) / len(lambdas))
            n_edges = int(np.count_nonzero(np.abs(W_try) > 1e-6)) if ok else -1
            tries.append({"lambda1": float(lam), "finite": bool(ok), "nonzero": n_edges})
            if ok and (self.lambda1 is not None or n_edges <= max_edges):
                W, lam_used = W_try, lam
                break
            if ok:
                W, lam_used = W_try, lam                     # dense: keep it only if no larger penalty does better
        else:
            if not any(t["finite"] for t in tries):
                W = np.zeros((N, N), dtype=float)            # every fit overflowed: an honest empty graph
        diverged = not any(t["finite"] for t in tries)

        np.fill_diagonal(W, 0.0)
        A_est = W * (gene_scale[:, None] / gene_scale[None, :])  # back to the original units

        # Discretize into signed adjacency matrix
        adj_pred = np.zeros((N, N), dtype=int)
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = max(w_thresh, np.mean(scores) * 0.75) if np.any(scores > 0) else w_thresh
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="notears",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "h_violation": float(_h(W)),
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
                "use_data_prep": use_prep,
                "lambda1": float(lam_used),
                "lambda_max": float(lambda_max),
                "fits": tries,
                "diverged": bool(diverged),
            },
        )
