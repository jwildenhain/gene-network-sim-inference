"""
Network Deconvolution (ND): Distinguishing Direct Dependencies from Indirect Transitive Cascades.
Based on Feizi et al. (Nature Biotechnology 2013, "Network deconvolution as a general method
to distinguish direct dependencies in networks").
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class NetworkDeconvolution(BaseInferenceAlgorithm):
    """
    Recovers direct causal regulatory interactions by analytically deconvolving
    indirect transitive correlations from perturbation response data:
        G_dir = G_obs * (I + G_obs)^(-1)
    """

    def __init__(
        self,
        beta: float = 0.85,
        threshold: float = 0.08,
        k_interactions: Optional[int] = None,
    ):
        super().__init__(name="netdeconv")
        self.beta = beta
        self.threshold = threshold
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        beta = float(kwargs.get("beta", self.beta))
        thresh = float(kwargs.get("threshold", self.threshold))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        # 1. Compute observed response association matrix
        # Combine correlation and directed perturbation sensitivity
        corr = np.corrcoef(X_mat)
        np.nan_to_num(corr, copy=False, nan=0.0)

        # Directed perturbation influence: sensitivity S = -X @ pinv(U)
        pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
        S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)

        # Symmetrize base dependency for deconvolution
        G_obs = 0.5 * (corr + corr.T)
        np.fill_diagonal(G_obs, 0.0)

        # 2. Eigen-decomposition for Network Deconvolution
        eigvals, eigvecs = np.linalg.eigh(G_obs)

        # Scale eigenvalues so that max absolute eigenvalue is beta < 1
        max_eig = np.max(np.abs(eigvals))
        if max_eig > 1e-12:
            scaled_eigvals = (eigvals / max_eig) * beta
        else:
            scaled_eigvals = eigvals

        # Deconvolve: lambda_dir = lambda_obs / (1 + lambda_obs)
        deconv_eigvals = scaled_eigvals / (1.0 + np.maximum(scaled_eigvals, -0.999))

        # Reconstruct deconvolved matrix
        G_dir = eigvecs @ np.diag(deconv_eigvals) @ eigvecs.T
        np.fill_diagonal(G_dir, 0.0)

        # 3. Impose directed causal asymmetry using perturbation sensitivity S_dir
        # If regulator j causes response in i, A_est[i, j] takes sign of correlation and magnitude from G_dir
        A_est = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j:
                    weight = G_dir[i, j]
                    # Direct asymmetry from perturbation response
                    asym_factor = 1.0 + np.tanh(S_dir[i, j] - S_dir[j, i])
                    A_est[i, j] = weight * asym_factor

        np.fill_diagonal(A_est, 0.0)
        # Orientation: use true_matrix[regulator, target] (rows = regulators)
        A_est = A_est.T.copy()

        # Discretize
        adj_pred = np.zeros((N, N), dtype=int)
        if K is not None:
            for i in range(N):
                mags = np.abs(A_est[i, :])
                top_idx = np.argsort(mags)[-K:]
                for reg in top_idx:
                    if mags[reg] >= thresh:
                        adj_pred[i, reg] = int(np.sign(A_est[i, reg]))
        else:
            max_mag = np.max(np.abs(A_est)) if np.max(np.abs(A_est)) > 0 else 1.0
            norm_A = A_est / max_mag
            mask = np.abs(norm_A) >= thresh
            adj_pred[mask] = np.sign(norm_A[mask]).astype(int)

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="netdeconv",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=elapsed,
            diagnostics={"beta": beta, "max_eigval": float(max_eig)},
        )
