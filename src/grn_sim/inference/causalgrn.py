"""
CausalGRN: Causal Discovery from Interventional Perturbation Screens.
Based on causal discovery principles for single-cell CRISPR/perturbation screens
(do-calculus, partial correlation skeleton discovery, and interventional edge orientation).
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class CausalGRNAlgorithm(BaseInferenceAlgorithm):
    """
    CausalGRN: Deciphering direct causal gene regulatory interactions from perturbation screens.

    1. Infers undirected regulatory skeleton via first-order partial correlation conditioning,
       removing spurious correlations driven by strong third-party master regulators.
    2. Exploits experimental intervention vectors (do-calculus) to direct edges j -> i:
       exogenous perturbation at gene j identifies causal downstream propagation to target i.
    3. Calculates interventional regression coefficients to determine sign (+ activation, - inhibition).
    """

    def __init__(
        self,
        pcorr_threshold: float = 0.08,
        k_interactions: Optional[int] = None,
        alpha: float = 0.05,
    ):
        super().__init__(name="causalgrn")
        self.pcorr_threshold = pcorr_threshold
        self.k_interactions = k_interactions
        self.alpha = alpha

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        thresh = float(kwargs.get("pcorr_threshold", self.pcorr_threshold))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        # 1. Compute zero-order correlation matrix
        corr = np.corrcoef(X_mat)
        np.nan_to_num(corr, copy=False, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # 2. Compute first-order partial correlation matrix:
        # r_{ij|k} = (r_ij - r_ik * r_jk) / sqrt((1 - r_ik^2) * (1 - r_jk^2))
        pcorr_mat = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = corr[i, j]
                # Find most influential conditioning variable k
                min_pcorr = abs(r_ij)
                best_sign = np.sign(r_ij)
                for k in range(N):
                    if k == i or k == j:
                        continue
                    r_ik = corr[i, k]
                    r_jk = corr[j, k]
                    denom = np.sqrt(max(1e-9, (1.0 - r_ik ** 2) * (1.0 - r_jk ** 2)))
                    pcorr_val = (r_ij - r_ik * r_jk) / denom
                    if abs(pcorr_val) < min_pcorr:
                        min_pcorr = abs(pcorr_val)
                        best_sign = np.sign(pcorr_val)

                pcorr_mat[i, j] = best_sign * min_pcorr
                pcorr_mat[j, i] = best_sign * min_pcorr

        # 3. Interventional Directionality from Perturbation Matrix U
        # For condition m where gene j is perturbed (U[j, m] != 0), response in gene i indicates j -> i
        causal_influence = np.zeros((N, N), dtype=float)
        pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
        try:
            # Linear causal effect: Delta X ~ B * U => B = Delta X @ pinv(U)
            B_effect = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
        except Exception:
            B_effect = np.zeros((N, N))

        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Combine partial correlation skeleton with directed interventional effect
                direct_skel = abs(pcorr_mat[i, j])
                interv_eff = B_effect[i, j] if abs(B_effect[i, j]) > 1e-4 else pcorr_mat[i, j]
                sign_eff = np.sign(interv_eff) if abs(interv_eff) > 1e-5 else np.sign(corr[i, j])

                # Weight: product of partial correlation confidence and interventional sensitivity
                A_est[i, j] = sign_eff * (0.6 * direct_skel + 0.4 * min(1.0, abs(interv_eff)))

        # 4. Discretization
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = max(thresh, np.mean(scores) * 0.8)
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="causalgrn",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "pcorr_threshold": thresh,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
            },
        )
