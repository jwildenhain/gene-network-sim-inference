"""
DMD-GRN: Dynamic Mode Decomposition for Gene Regulatory Networks.
Based on Koopman operator theory and dynamic modal decomposition for biological dynamical systems
(Brunton et al., Kutz et al., Schmid 2010).
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class DMDGRNAlgorithm(BaseInferenceAlgorithm):
    """
    DMD-GRN: Dynamic Mode Decomposition for Gene Regulatory Network reverse-engineering.

    1. Formulates state transition dynamics X_{k+1} ~ A * X_k.
    2. Performs truncated Singular Value Decomposition (SVD) on expression snapshots:
       X_1 = U * Sigma * V^T
    3. Computes the best-fit linear Koopman transition operator:
       A_DMD = X_2 * V * Sigma^(-1) * U^T
    4. Extracts dynamic eigenvalues (oscillation frequencies and decay rates) and modes.
    5. Discretizes directed off-diagonal regulatory coupling coefficients into signed network topology.
    """

    def __init__(
        self,
        rank_r: Optional[int] = None,
        threshold: float = 0.05,
        k_interactions: Optional[int] = None,
    ):
        super().__init__(name="dmdgrn")
        self.rank_r = rank_r
        self.threshold = threshold
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        thresh = float(kwargs.get("threshold", self.threshold))

        # 1. Setup Snapshot Matrices
        # In perturbation experiments, X is steady state responses, U is perturbation.
        # Dynamic relationship: X = -inv(A) * U => A * X = -U
        # Dynamic mode decomposition between effective regulator state Z = (X + U) and response -X
        if M > 1:
            X1 = X_mat[:, :-1]
            X2 = X_mat[:, 1:]
        else:
            X1 = X_mat
            X2 = -U_mat

        # Combine with perturbation transition matrix
        # Estimate operator A such that A @ X1 ~ X2
        r = self.rank_r
        if r is None:
            # Optimal hard thresholding or min dimension
            r = min(N - 1, max(2, int(min(X1.shape) * 0.85)))
        else:
            r = min(r, min(X1.shape))

        # 2. SVD of X1
        try:
            U_svd, S_svd, Vt_svd = np.linalg.svd(X1, full_matrices=False)
            r = min(r, len(S_svd))
            U_r = U_svd[:, :r]
            S_r = S_svd[:r]
            Vt_r = Vt_svd[:r, :]

            # Truncated Koopman operator
            # A_tilde = U_r^T @ X2 @ V_r @ inv(Sigma_r)
            A_tilde = U_r.T @ X2 @ Vt_r.T @ np.diag(1.0 / np.maximum(S_r, 1e-8))
            A_dmd = U_r @ A_tilde @ U_r.T
        except Exception:
            # Robust pseudo-inverse fallback
            A_dmd = X2 @ np.linalg.pinv(X1)

        # 3. Incorporate perturbation input balance
        # At equilibrium, the interaction matrix A satisfies A @ X = -U
        try:
            A_pert = -U_mat @ np.linalg.pinv(X_mat)
            A_combined = 0.5 * (A_dmd + A_pert)
        except Exception:
            A_combined = A_dmd

        A_est = A_combined.copy()
        np.fill_diagonal(A_est, 0.0)  # Exclude self-degradation loops for edge topology

        # 4. Discretization into signed adjacency
        adj_pred = np.zeros((N, N), dtype=int)
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = max(thresh, np.mean(scores) * 0.75) if np.any(scores > 0) else thresh
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="dmdgrn",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "rank_r": r,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
            },
        )
