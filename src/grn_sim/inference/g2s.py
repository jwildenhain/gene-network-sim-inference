"""
General-to-Specific (G2S) reverse engineering algorithm.
Implements the backward elimination basis pruning technique with AIC minimization
from thesis Chapter 3 and gen2spec.m.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class GeneralToSpecific(BaseInferenceAlgorithm):
    """
    General-to-Specific (G2S) algorithm based on Crampin et al.
    Starts with a full set of basis functions and systematically removes
    the interaction causing the least increase in residual error until the
    cost function AIC^(k) is minimised.
    """

    def __init__(
        self,
        criterion: str = "aic",
        min_interactions: int = 1,
        threshold: float = 1e-4,
    ):
        super().__init__(name="g2s")
        self.criterion = criterion.lower()
        self.min_interactions = min_interactions
        self.threshold = threshold

    def _penalty(self, E: np.ndarray, k: int, n_samples: int) -> float:
        """Computes AIC cost function matching thesis: sum(E^2) + k."""
        rss = float(np.sum(E ** 2))
        if self.criterion == "mdl":
            mse = max(rss / n_samples, 1e-12)
            return n_samples * np.log(mse) + k * np.log(n_samples)
        else:
            return rss + float(k)

    def _drop_another(
        self,
        y: np.ndarray,
        X: np.ndarray,
        current_ind: List[int],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Tests dropping each basis in current_ind, returns new indices without the worst basis.
        """
        k = len(current_ind)
        candidate_mse = np.zeros(k, dtype=float)

        for i in range(k):
            # Subset without i
            subset = [current_ind[idx] for idx in range(k) if idx != i]
            if len(subset) == 0:
                candidate_mse[i] = np.inf
                continue
            Xb = X[subset, :].T  # (n_samples, len(subset))
            a_sub = np.linalg.pinv(Xb) @ y
            resid = (Xb @ a_sub) - y
            candidate_mse[i] = float(np.mean(resid ** 2))

        # Best subset is the one with minimum error after dropping
        min_drop_idx = int(np.argmin(candidate_mse))
        new_ind = [current_ind[idx] for idx in range(k) if idx != min_drop_idx]

        # Compute residuals of new set
        X_new = X[new_ind, :].T
        a_new = np.linalg.pinv(X_new) @ y
        final_E = (X_new @ a_new) - y
        return new_ind, final_E

    def _fit_single_target(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits sparse linear model for a single target gene via G2S backward elimination.
        """
        n_features, n_samples = X.shape
        ind = list(range(n_features))
        k = len(ind)

        # Initial full model
        Xb = X[ind, :].T
        alpha = np.linalg.pinv(Xb) @ y
        E = (Xb @ alpha) - y
        s_current = self._penalty(E, k, n_samples)
        s_old = s_current + 1.0

        opt_ind = list(ind)
        opt_cost = s_current

        while k > self.min_interactions and s_current < s_old:
            opt_ind = list(ind)
            opt_cost = s_current
            s_old = s_current

            ind, E = self._drop_another(y, X, ind)
            k = len(ind)
            s_current = self._penalty(E, k, n_samples)

        # Refit optimal model
        if len(opt_ind) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        X_opt = X[opt_ind, :].T
        final_a = np.linalg.pinv(X_opt) @ y
        return np.array(opt_ind, dtype=int), final_a

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Infers regulatory matrix row by row using G2S.
        """
        start_time = time.time()
        n = X.shape[0]
        A_est = np.zeros((n, n), dtype=float)

        for i in range(n):
            y = -U[i, :]
            if np.all(np.abs(y) < 1e-12):
                y = -X[i, :]

            indices, coeffs = self._fit_single_target(y, X)
            for idx, c in zip(indices, coeffs):
                if abs(c) > self.threshold:
                    A_est[i, idx] = c

        adj = np.zeros((n, n), dtype=int)
        adj[A_est > self.threshold] = 1
        adj[A_est < -self.threshold] = -1

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
            diagnostics={"criterion": self.criterion},
        )
