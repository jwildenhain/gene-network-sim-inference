"""
ARACNE: Algorithm for the Reconstruction of Accurate Cellular Networks.
Based on Margolin et al. (Nature Protocols 2006, "ARACNE: An Algorithm for the
Reconstruction of Gene Regulatory Networks in a Mammalian Cellular Context") and
Basso et al. (Nature Genetics 2005).
Enhanced with Ledoit-Wolf covariance shrinkage and interventional asymmetry compensation.
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


class AracneAlgorithm(BaseInferenceAlgorithm):
    """
    ARACNE: Mutual Information with Data Processing Inequality (DPI) pruning.
    
    1. Computes pairwise Mutual Information matrix I(X_i, X_j) using regularized Gaussian copula.
    2. Filters insignificant edges below an adaptive or specified MI threshold.
    3. Applies Data Processing Inequality (DPI) on every triplet (i, j, k):
       If I(i, j) < (1 - eps) * min(I(i, k), I(j, k)), edge (i, j) is removed as an indirect cascade.
    4. Causal direction is assigned via experimental perturbation response asymmetry,
       with sign determined by the empirical expression correlation.
    """

    def __init__(
        self,
        eps: float = 0.10,
        mi_threshold: Optional[float] = None,
        k_interactions: Optional[int] = None,
        perturbation_mode: str = "additive",
        use_data_prep: bool = True,
    ):
        super().__init__(name="arachne")
        self.eps = eps
        self.mi_threshold = mi_threshold
        self.k_interactions = k_interactions
        self.perturbation_mode = perturbation_mode
        self.use_data_prep = use_data_prep

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        use_prep = kwargs.get("use_data_prep", self.use_data_prep)
        eps = float(kwargs.get("eps", self.eps))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        if use_prep:
            # 1. Data Preparation: Baseline shift Delta X & shrunk covariance
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            corr = prep.corr_shrunk
            S_dir = prep.sensitivity_mat
        else:
            # Raw sample correlation
            corr = np.corrcoef(X_mat)
            np.nan_to_num(corr, copy=False, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            try:
                pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
                S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
            except Exception:
                S_dir = -X_mat[:, :N] if M >= N else np.zeros((N, N))

        # 2. Estimate Mutual Information matrix using Gaussian copula estimator:
        # I(X_i; X_j) = -0.5 * ln(1 - r_ij^2)
        r_sq = np.clip(corr ** 2, 0.0, 0.999)
        MI = -0.5 * np.log(1.0 - r_sq)
        np.fill_diagonal(MI, 0.0)

        # 3. Determine significance threshold
        if self.mi_threshold is not None:
            thresh = float(self.mi_threshold)
        else:
            upper_vals = MI[np.triu_indices(N, k=1)]
            if len(upper_vals) > 0 and np.std(upper_vals) > 1e-6:
                thresh = float(np.mean(upper_vals) + 0.15 * np.std(upper_vals))
            else:
                thresh = 0.05

        MI_filtered = np.where(MI >= thresh, MI, 0.0)

        # 4. Data Processing Inequality (DPI) Triplet Pruning
        # For every triplet (i, j, k), if I(i, j) < (1 - eps) * min(I(i, k), I(k, j)), eliminate edge (i, j)
        DPI_pruned = MI_filtered.copy()
        for i in range(N):
            for j in range(i + 1, N):
                if DPI_pruned[i, j] == 0:
                    continue
                w_ij = DPI_pruned[i, j]
                for k in range(N):
                    if k == i or k == j:
                        continue
                    w_ik = DPI_pruned[i, k]
                    w_jk = DPI_pruned[j, k]
                    if w_ik > 0 and w_jk > 0:
                        min_alt = min(w_ik, w_jk)
                        if w_ij < (1.0 - eps) * min_alt:
                            DPI_pruned[i, j] = 0.0
                            DPI_pruned[j, i] = 0.0
                            break

        # 5. Direct Causal Orientation using Perturbation Response Asymmetry
        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if DPI_pruned[i, j] > 0:
                    asym = abs(S_dir[i, j])
                    sign_val = np.sign(S_dir[i, j]) if abs(S_dir[i, j]) > 1e-4 else np.sign(corr[i, j])
                    if sign_val == 0:
                        sign_val = 1.0
                    weight = DPI_pruned[i, j] * (1.0 + asym)
                    A_est[i, j] = sign_val * weight

        # 6. Apply top-K per gene or thresholding
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_indices = np.argsort(scores)[::-1][:K]
                for j in top_indices:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = np.mean(scores[scores > 0]) * 0.7 if np.any(scores > 0) else 0.0
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="arachne",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "eps": eps,
                "mi_threshold": thresh,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
                "use_data_prep": use_prep,
            },
        )
