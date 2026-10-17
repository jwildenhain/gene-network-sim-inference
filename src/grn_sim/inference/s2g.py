"""
Simple-to-General (S2G) reverse engineering algorithm.
Implements the forward-backward basis selection with Akaike Information Criterion (AIC)
and Minimum Description Length (MDL) penalty from thesis Chapter 3 and getvar.m.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class SimpleToGeneral(BaseInferenceAlgorithm):
    """
    Simple-to-General (S2G) algorithm based on Crampin et al. / Judd & Mees.
    Iteratively adds basis functions with maximal projection of residual error,
    removes negligible basis functions, and stops when AIC is minimized.
    """

    def __init__(
        self,
        criterion: str = "aic",  # "aic" or "mdl"
        max_interactions: Optional[int] = None,
        normalize_bases: bool = True,
        threshold: float = 1e-4,
    ):
        super().__init__(name="s2g")
        self.criterion = criterion.lower()
        self.max_interactions = max_interactions
        self.normalize_bases = normalize_bases
        self.threshold = threshold

    def _penalty(self, E: np.ndarray, k: int, n_samples: int) -> float:
        """Computes cost function for residual vector E and k parameters."""
        rss = float(np.sum(E ** 2))
        if self.criterion == "mdl":
            # N * log(mean(E^2)) + k * log(N)
            mse = max(rss / n_samples, 1e-12)
            return n_samples * np.log(mse) + k * np.log(n_samples)
        else:
            # Thesis Eq 3.14: AIC^K = (1 / n) * E^T * E + K
            return (rss / n_samples) + float(k)

    def _fit_single_target(
        self,
        y: np.ndarray,      # (m,) target values (-u_i)
        X: np.ndarray,      # (n, m) design matrix (features = genes)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits sparse linear regression for a single gene using simple-to-general selection.
        Returns:
            (selected_indices, coefficients)
        """
        n_features, n_samples = X.shape
        V = X.copy()

        # Normalize bases if requested
        if self.normalize_bases:
            norms = np.sqrt(np.sum(V ** 2, axis=1, keepdims=True))
            norms[norms < 1e-12] = 1.0
            V = V / norms

        max_k = self.max_interactions or n_features

        # Step 1: Select first basis vector with maximum absolute projection
        # V is (n_features, n_samples), y is (n_samples,)
        mu = np.abs(V @ y)
        first_idx = int(np.argmax(mu))
        current_indices = [first_idx]

        # Initial least squares on first basis
        Vb = V[current_indices, :].T  # (n_samples, 1)
        a = np.linalg.pinv(Vb) @ y
        E = (Vb @ a) - y
        s_current = self._penalty(E, 1, n_samples)

        history_indices = [list(current_indices)]
        history_costs = [s_current]

        # Step 2: Add and prune basis functions while cost improves
        k = 1
        while k < max_k:
            s_old = s_current
            # Search for best candidate to add and candidate to remove
            iin = -1
            iout = -2
            loop_guard = 0
            temp_ind = list(current_indices)

            while iin != iout and loop_guard < n_features * 2:
                loop_guard += 1
                # Project residuals to identify next iin
                mu = np.abs(V @ E)
                # Zero out projection of already selected indices to avoid duplicates
                for idx in temp_ind:
                    mu[idx] = -1.0
                if np.all(mu < 0):
                    break
                iin = int(np.argmax(mu))
                candidate_ind = temp_ind + [iin]

                # Update basis set and identify iout with lowest absolute coefficient
                Vb = V[candidate_ind, :].T
                a_cand = np.linalg.pinv(Vb) @ y
                amin_idx = int(np.argmin(np.abs(a_cand)))
                iout = candidate_ind[amin_idx]

                if iout != iin:
                    # Replace candidate
                    temp_ind[amin_idx] = iin
                    Vk = V[temp_ind, :].T
                    ak = np.linalg.pinv(Vk) @ y
                    E = (Vk @ ak) - y
                else:
                    yh = Vb @ a_cand
                    E = yh - y
                    temp_ind = candidate_ind

            current_indices = temp_ind
            k = len(current_indices)
            s_current = self._penalty(E, k, n_samples)

            history_indices.append(list(current_indices))
            history_costs.append(s_current)

            # Stopping condition: if cost did not improve, terminate
            if s_current >= s_old:
                break

        # Best set of indices across history
        best_step = int(np.argmin(history_costs))
        best_indices = history_indices[best_step]

        if len(best_indices) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Refit on original un-normalized X
        Xb = X[best_indices, :].T  # (n_samples, k)
        final_a = np.linalg.pinv(Xb) @ y
        return np.array(best_indices, dtype=int), final_a

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Fits the entire interaction matrix row by row.
        Solves A * X = -U  =>  a_i * X = -u_i
        """
        start_time = time.time()
        n = X.shape[0]
        A_est = np.zeros((n, n), dtype=float)

        for i in range(n):
            # Target is -u_i (the i-th row of perturbation matrix)
            y = -U[i, :]
            # If all zeros (e.g. unperturbed or implicit), fallback to -X[i, :]
            if np.all(np.abs(y) < 1e-12):
                y = -X[i, :]

            indices, coeffs = self._fit_single_target(y, X)
            for idx, c in zip(indices, coeffs):
                if abs(c) > self.threshold:
                    A_est[i, idx] = c

        # Adjacency: sign of estimated coefficients
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
