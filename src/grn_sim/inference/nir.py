"""
Network Identification by Multiple Regression (NIR) algorithm.
Implements Gardner et al. (Science 2003) and nir_v3.m with forward-topD greedy search
and ridge regression.
"""

import time
import itertools
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class NIRAlgorithm(BaseInferenceAlgorithm):
    """
    Network Identification by Multiple Regression (NIR).
    Finds a fixed number K of regulatory connections per gene by minimizing
    the sum of squared errors (SSE) using multiple linear regression and optional ridge penalty.
    """

    def __init__(
        self,
        k_interactions: int = 2,
        top_d: int = 10,
        ridge: float = 0.0,
        cond_max: float = 1000.0,
        use_perturbation_values: bool = True,
        threshold: float = 1e-4,
    ):
        super().__init__(name="nir")
        self.k_interactions = k_interactions
        self.top_d = top_d
        self.ridge = ridge
        self.cond_max = cond_max
        self.use_perturbation_values = use_perturbation_values
        self.threshold = threshold

    def _fit_single_row(
        self,
        i: int,
        X: np.ndarray,
        P: np.ndarray,
        K: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solves row i using forward top-D greedy search matching nir_v3.m.
        """
        N, M = X.shape
        u = P[i, :].copy()
        Xtmp = X.copy()
        nopert = 0

        # Check if gene i was perturbed
        if not np.any(np.abs(u) > 1e-12):
            # Extract row i as target response
            u = -Xtmp[i, :].copy()
            # Remove gene i from predictors
            keep_indices = [idx for idx in range(N) if idx != i]
            Xtmp = Xtmp[keep_indices, :]
            nopert = 1

        Ntmp = Xtmp.shape[0]
        Ktmp = max(1, min(K - nopert, Ntmp))

        # Forward top-D search
        top_d_combos: List[List[int]] = []
        n_coeff = 0

        while n_coeff < Ktmp:
            if n_coeff == 0:
                candidates = [[c] for c in range(Ntmp)]
            else:
                candidates = []
                for prev in top_d_combos:
                    for cand in range(Ntmp):
                        if cand not in prev:
                            candidates.append(prev + [cand])

            # Eliminate duplicates by sorted tuple
            unique_candidates = [list(c) for c in {tuple(sorted(c)): c for c in candidates}.values()]

            combo_sse = []
            combo_coeffs = []

            for combo in unique_candidates:
                Z = Xtmp[combo, :]  # (len(combo), M)
                ZZt = Z @ Z.T
                cond_val = np.linalg.cond(ZZt) if ZZt.size > 0 else 1.0

                if cond_val <= self.cond_max:
                    reg = self.ridge * np.eye(len(combo))
                    # a = -u * Z^T * (Z * Z^T + reg)^{-1}
                    try:
                        inv_mat = np.linalg.inv(ZZt + reg)
                        a_cand = -u @ Z.T @ inv_mat
                        sse = float(np.sum((u + a_cand @ Z) ** 2))
                    except np.linalg.LinAlgError:
                        a_cand = np.zeros(len(combo))
                        sse = np.inf
                else:
                    a_cand = np.zeros(len(combo))
                    sse = np.inf

                combo_sse.append(sse)
                combo_coeffs.append(a_cand)

            sorted_indices = np.argsort(combo_sse)
            d_keep = min(self.top_d, len(sorted_indices))
            top_d_combos = [unique_candidates[sorted_indices[d]] for d in range(d_keep)]

            if n_coeff == Ktmp - 1:
                best_idx = sorted_indices[0]
                best_combo = unique_candidates[best_idx]
                best_coeffs = combo_coeffs[best_idx]
                best_sse = combo_sse[best_idx]

            n_coeff += 1

        # Adjust indices if row was extracted
        final_indices = np.array(best_combo, dtype=int)
        if nopert:
            # Shift indices back
            final_indices = np.array([idx if idx < i else idx + 1 for idx in final_indices])

        return final_indices, best_coeffs, best_sse

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Fits regulatory interaction matrix using NIR.
        """
        start_time = time.time()
        N, M = X.shape
        K = kwargs.get("k", self.k_interactions)
        # Cap K at N - 1
        K = min(K, N)

        A_est = np.zeros((N, N), dtype=float)
        diagnostics = {"sse": []}

        for i in range(N):
            indices, coeffs, sse = self._fit_single_row(i, X, U, K)
            for idx, c in zip(indices, coeffs):
                if abs(c) > self.threshold:
                    A_est[i, idx] = c
            diagnostics["sse"].append(sse)

        adj = np.zeros((N, N), dtype=int)
        adj[A_est > self.threshold] = 1
        adj[A_est < -self.threshold] = -1

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
            diagnostics=diagnostics,
        )
