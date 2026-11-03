"""
CausalGRN: Causal Discovery from Interventional Perturbation Screens.
Based on causal discovery principles for single-cell CRISPR/perturbation screens
(do-calculus, partial correlation skeleton discovery, and interventional edge orientation).
Enhanced with baseline response shift extraction and regularized correlation conditioning.
"""

import time
from typing import Optional
import numpy as np

from grn_sim.inference._parallel import default_n_jobs, map_indices
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


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
        use_data_prep: bool = True,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(name="causalgrn")
        self.n_jobs = n_jobs
        self.pcorr_threshold = pcorr_threshold
        self.k_interactions = k_interactions
        self.alpha = alpha
        self.use_data_prep = use_data_prep

    @staticmethod
    def _pcorr_row(corr: np.ndarray, i: int) -> np.ndarray:
        """Signed minimum |partial correlation| of gene ``i`` with each gene j > i (first minimum over k wins, and it
        must beat the marginal |r_ij| strictly)."""
        N = corr.shape[0]
        js = np.arange(i + 1, N)
        rows = np.arange(js.size)
        r_ij = corr[i, js]
        r_ik = corr[i, :]
        r_jk = corr[js, :]
        denom = np.sqrt(np.maximum(1e-6, (1.0 - r_ik ** 2)[None, :] * (1.0 - r_jk ** 2)))
        val = (r_ij[:, None] - r_ik[None, :] * r_jk) / denom
        mag = np.abs(val)
        mag[np.isnan(mag)] = np.inf
        mag[:, i] = np.inf                  # k == i
        mag[rows, js] = np.inf              # k == j
        k_best = np.argmin(mag, axis=1)
        min_k = mag[rows, k_best]
        better = min_k < np.abs(r_ij)
        min_pcorr = np.where(better, min_k, np.abs(r_ij))
        best_sign = np.where(better, np.sign(val[rows, k_best]), np.sign(r_ij))
        return best_sign * min_pcorr

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        use_prep = kwargs.get("use_data_prep", self.use_data_prep)
        thresh = float(kwargs.get("pcorr_threshold", self.pcorr_threshold))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        if use_prep:
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            corr = prep.corr_shrunk
            B_effect = prep.sensitivity_mat
        else:
            corr = np.corrcoef(X_mat)
            np.nan_to_num(corr, copy=False, nan=0.0)
            np.fill_diagonal(corr, 1.0)
            pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
            try:
                B_effect = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
            except Exception:
                B_effect = np.zeros((N, N))

        # 2. Compute first-order partial correlation matrix:
        # r_{ij|k} = (r_ij - r_ik * r_jk) / sqrt((1 - r_ik^2) * (1 - r_jk^2))
        # For each pair the smallest |partial correlation| over all conditioning genes k, one row i at a time.
        pcorr_mat = np.zeros((N, N), dtype=float)
        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=16)
        for i, vals in enumerate(map_indices(self._pcorr_row, corr, N - 1, n_jobs)):
            pcorr_mat[i, i + 1:] = vals
            pcorr_mat[i + 1:, i] = vals

        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        direct_skel = np.abs(pcorr_mat)
        interv_eff = np.where(np.abs(B_effect) > 1e-4, B_effect, pcorr_mat)
        sign_eff = np.where(np.abs(interv_eff) > 1e-5, np.sign(interv_eff), np.sign(corr))
        sign_eff = np.where(sign_eff == 0, 1.0, sign_eff)
        A_est = sign_eff * (0.6 * direct_skel + 0.4 * np.minimum(1.0, np.abs(interv_eff)))
        np.fill_diagonal(A_est, 0.0)

        # 4. Discretization
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = max(thresh, np.mean(scores) * 0.75) if np.any(scores > 0) else thresh
                keep = (scores > row_thresh) & (scores > 1e-4)
                adj_pred[i, keep] = np.sign(A_est[i, keep]).astype(int)

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="causalgrn",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "pcorr_threshold": thresh,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
                "use_data_prep": use_prep,
            },
        )
