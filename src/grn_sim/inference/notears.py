"""
NOTEARS: Continuous Optimization for Directed Acyclic Graphs (DAGs).
Based on Zheng et al. (NeurIPS 2018, "DAGs with NO TEARS: Continuous Optimization for Directed Acyclic Graphs").
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from scipy.linalg import expm

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class NOTEARSAlgorithm(BaseInferenceAlgorithm):
    """
    NOTEARS: Differentiable, continuous optimization for directed causal network structure learning.

    Replaces discrete combinatorial DAG search with smooth mathematical optimization:
        min_W  (1 / 2M) * ||X - W^T * X||_F^2 + lambda * ||W||_1
        s.t.   h(W) = Tr(exp(W o W)) - d = 0

    Solves the continuous constrained formulation using an Augmented Lagrangian method.
    """

    def __init__(
        self,
        lambda1: float = 0.05,
        max_iter: int = 25,
        h_tol: float = 1e-5,
        rho_max: float = 1e+5,
        w_threshold: float = 0.05,
        k_interactions: Optional[int] = None,
    ):
        super().__init__(name="notears")
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        N, M = X_mat.shape

        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        w_thresh = float(kwargs.get("w_threshold", self.w_threshold))

        # Center data across samples
        X_centered = X_mat - np.mean(X_mat, axis=1, keepdims=True)

        # Covariance surrogate: C = (1 / M) * X @ X^T
        C = (1.0 / max(1, M)) * (X_centered @ X_centered.T)
        norm_C = float(np.linalg.norm(C, 2)) + 1e-4

        def _h(W):
            """Acyclicity constraint: Tr(exp(W o W)) - N."""
            M_sq = np.clip(W * W, 0.0, 10.0)
            E = expm(M_sq)
            return float(np.trace(E) - N)

        def _grad(W, rho, alpha):
            """Gradient of augmented Lagrangian with respect to W."""
            grad_F = C @ W - C
            M_sq = np.clip(W * W, 0.0, 10.0)
            E = expm(M_sq)
            h_val = float(np.trace(E) - N)
            grad_h = 2.0 * (E.T * W)
            grad_tot = grad_F + (alpha + rho * h_val) * grad_h
            return np.clip(grad_tot, -10.0, 10.0), h_val

        # Augmented Lagrangian initialization
        W = np.zeros((N, N), dtype=float)
        rho = 1.0
        alpha = 0.0
        step_size = 0.02 / norm_C

        h_prev = np.inf
        for _ in range(self.max_iter):
            # Gradient descent inner loop
            for _ in range(12):
                grad_W, h_val = _grad(W, rho, alpha)
                np.fill_diagonal(grad_W, 0.0)

                # Proximal gradient step for L1 regularization
                W_step = W - step_size * grad_W
                # Soft-thresholding
                lam_step = self.lambda1 * step_size
                W = np.sign(W_step) * np.maximum(0.0, np.abs(W_step) - lam_step)
                W = np.clip(W, -3.0, 3.0)
                np.fill_diagonal(W, 0.0)

            # Check constraint violation
            h_val = _h(W)
            if abs(h_val) <= self.h_tol or rho >= self.rho_max:
                break
            if h_val > 0.25 * h_prev:
                rho = min(self.rho_max, rho * 5.0)
            h_prev = h_val
            alpha += rho * h_val

        np.fill_diagonal(W, 0.0)
        A_est = W.copy()

        # Discretize
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
            },
        )
