"""
FAST algorithm for reverse engineering gene networks.
Implements the SVD decomposition and sparse row truncation method of Farina et al.
matching fast_modified.m from the thesis.
"""

import time
from typing import Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class FastSVD(BaseInferenceAlgorithm):
    """
    FAST algorithm (Farina et al.).
    Computes initial dense interaction matrix A_0 using truncated SVD,
    selects top K connections with largest absolute magnitude per row,
    and recalculates optimal sparse coefficients using Moore-Penrose pseudo-inverse.
    """

    def __init__(
        self,
        k_interactions: int = 2,
        threshold: float = 1e-4,
    ):
        super().__init__(name="fast")
        self.k_interactions = k_interactions
        self.threshold = threshold

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Solves A * X = -U using SVD and sparse thresholding.
        """
        start_time = time.time()
        N, M = X.shape
        K = kwargs.get("k", self.k_interactions)
        K = max(1, min(K, N))

        # 1. Singular Value Decomposition
        U_svd, S_svd, Vt_svd = np.linalg.svd(X, full_matrices=True)
        V_svd = Vt_svd.T

        # Truncate rank N-1 matching fast_modified.m
        rank = max(1, min(N - 1, len(S_svd) - 1))
        Ur = U_svd[:, :rank]
        Sr = np.diag(S_svd[:rank])
        Vr = V_svd[:, :rank]

        # Initial estimate A_0 = (X - U) * Vr * inv(Sr) * Ur^T - I
        inv_Sr = np.diag(1.0 / np.maximum(S_svd[:rank], 1e-12))
        A_0 = (X - U) @ Vr @ inv_Sr @ Ur.T - np.eye(N)

        # 2. Select top K elements per row with largest magnitude
        A_est = np.zeros((N, N), dtype=float)

        for i in range(N):
            row_abs = np.abs(A_0[i, :])
            # Indices of K largest entries in row i
            top_k_indices = np.argsort(row_abs)[-K:]
            top_k_indices = np.sort(top_k_indices)

            # Sub-matrix of selected rows from X
            x_sel = X[top_k_indices, :]  # (K, M)
            pinv_x = np.linalg.pinv(x_sel)  # (M, K)

            # Re-estimate sparse coefficients: a_i = -u_i * pinv(x_sel)
            u_row = U[i, :]
            coeffs = -u_row @ pinv_x  # (K,)

            for idx, c in zip(top_k_indices, coeffs):
                if abs(c) > self.threshold:
                    A_est[i, idx] = c

        adj = np.zeros((N, N), dtype=int)
        adj[A_est > self.threshold] = 1
        adj[A_est < -self.threshold] = -1

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
            diagnostics={"k": K},
        )
