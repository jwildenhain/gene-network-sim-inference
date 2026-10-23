"""
NOTEARS: Continuous Optimization for Directed Acyclic Graphs (DAGs).
Based on Zheng et al. (NeurIPS 2018, "DAGs with NO TEARS: Continuous Optimization for Directed Acyclic Graphs").
Enhanced with interventional equilibrium data preparation and soft-acyclicity compensation.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
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
    """

    def __init__(
        self,
        lambda1: float = 0.02,
        max_iter: int = 25,
        h_tol: float = 1e-4,
        rho_max: float = 1000.0,
        w_threshold: float = 0.05,
        k_interactions: Optional[int] = None,
        use_data_prep: bool = True,
    ):
        super().__init__(name="notears")
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.k_interactions = k_interactions
        self.use_data_prep = use_data_prep

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

        if use_prep:
            # 1. Data Preparation: Baseline shift Delta X = X - x0 & interventional alignment
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            Delta_X = prep.Delta_X
            C = (1.0 / max(1, M)) * (Delta_X @ Delta_X.T)
            B = (1.0 / max(1, M)) * (Delta_X + U_mat) @ Delta_X.T
            norm_C = float(np.linalg.norm(C, 2)) + 1e-4
            eff_rho_max = min(self.rho_max, 1000.0)

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

        # Augmented Lagrangian initialization
        W = np.zeros((N, N), dtype=float)
        rho = 1.0
        alpha = 0.0
        step_size = 0.05 / norm_C

        h_prev = np.inf
        for _ in range(self.max_iter):
            # Gradient descent inner loop
            for _ in range(15):
                grad_W, h_val = _grad(W, rho, alpha)
                np.fill_diagonal(grad_W, 0.0)

                # Proximal gradient step for L1 regularization
                W_step = W - step_size * grad_W
                lam_step = self.lambda1 * step_size
                W = np.sign(W_step) * np.maximum(0.0, np.abs(W_step) - lam_step)
                W = np.clip(W, -3.0, 3.0)
                np.fill_diagonal(W, 0.0)

            # Check constraint violation
            h_val = _h(W)
            if abs(h_val) <= self.h_tol or rho >= eff_rho_max:
                break
            if h_val > 0.25 * h_prev:
                rho = min(eff_rho_max, rho * 2.0)
            h_prev = h_val
            alpha += rho * h_val

        np.fill_diagonal(W, 0.0)
        A_est = W.copy()

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
            },
        )
