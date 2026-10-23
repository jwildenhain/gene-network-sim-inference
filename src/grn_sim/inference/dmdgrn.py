"""
DMD-GRN: Dynamic Mode Decomposition for Gene Regulatory Networks.
Based on Koopman operator theory and dynamic modal decomposition for biological dynamical systems
(Brunton et al., Kutz et al., Schmid 2010).
Enhanced with Controlled DMD (DMDc) data preparation and baseline perturbation response scaling.
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


class DMDGRNAlgorithm(BaseInferenceAlgorithm):
    """
    DMD-GRN: Dynamic Mode Decomposition for Gene Regulatory Network reverse-engineering.

    1. Formulates state transition dynamics using Controlled DMD (DMDc):
       Delta X ~ -inv(A) * U
    2. Performs truncated Singular Value Decomposition (SVD) on expression shifts:
       Delta X = V_r * Sigma_r * W_r^T
    3. Computes the best-fit linear Koopman transition operator:
       A_DMD = -U * W_r * Sigma_r^(-1) * V_r^T
    4. Discretizes directed off-diagonal regulatory coupling coefficients into signed network topology.
    """

    def __init__(
        self,
        rank_r: Optional[int] = None,
        threshold: float = 0.05,
        k_interactions: Optional[int] = None,
        use_data_prep: bool = True,
    ):
        super().__init__(name="dmdgrn")
        self.rank_r = rank_r
        self.threshold = threshold
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
        thresh = float(kwargs.get("threshold", self.threshold))

        if use_prep:
            # 1. Data Preparation: Controlled perturbation shifts Delta X = X - x0
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            Delta_X = prep.Delta_X

            # Determine truncation rank r
            r = self.rank_r
            if r is None:
                r = min(N - 1, max(2, int(min(Delta_X.shape) * 0.85)))
            else:
                r = min(r, min(Delta_X.shape))

            # Truncated SVD of Delta X
            try:
                U_svd, S_svd, Vt_svd = np.linalg.svd(Delta_X, full_matrices=False)
                r = min(r, len(S_svd))
                U_r = U_svd[:, :r]
                S_r = S_svd[:r]
                Vt_r = Vt_svd[:r, :]

                # Controlled DMD inversion: A_dmd = -U_mat @ Vt_r.T @ inv(S_r) @ U_r.T
                S_inv = np.diag(1.0 / np.maximum(S_r, 1e-4))
                A_dmd = -U_mat @ Vt_r.T @ S_inv @ U_r.T
            except Exception:
                A_dmd = -U_mat @ np.linalg.pinv(Delta_X)

            A_est = A_dmd.copy()
        else:
            # Classical unadjusted consecutive snapshot slicing
            if M > 1:
                X1 = X_mat[:, :-1]
                X2 = X_mat[:, 1:]
            else:
                X1 = X_mat
                X2 = -U_mat

            r = self.rank_r or min(N - 1, max(2, int(min(X1.shape) * 0.85)))
            try:
                U_svd, S_svd, Vt_svd = np.linalg.svd(X1, full_matrices=False)
                r = min(r, len(S_svd))
                U_r = U_svd[:, :r]
                S_r = S_svd[:r]
                Vt_r = Vt_svd[:r, :]
                A_tilde = U_r.T @ X2 @ Vt_r.T @ np.diag(1.0 / np.maximum(S_r, 1e-8))
                A_dmd = U_r @ A_tilde @ U_r.T
            except Exception:
                A_dmd = X2 @ np.linalg.pinv(X1)

            try:
                A_pert = -U_mat @ np.linalg.pinv(X_mat)
                A_est = 0.5 * (A_dmd + A_pert)
            except Exception:
                A_est = A_dmd

        np.fill_diagonal(A_est, 0.0)  # Exclude self-degradation loops for edge topology

        # Discretization into signed adjacency
        adj_pred = np.zeros((N, N), dtype=int)
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_indices = np.argsort(scores)[::-1][:K]
                for j in top_indices:
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
                "use_data_prep": use_prep,
            },
        )
