"""
Simple-to-General (S2G) reverse engineering algorithm.
Implements the exact MATLAB s2g.m and getvar.m forward-backward basis selection
with Minimum Description Length (MDL) / AIC log penalty from thesis Chapter 3.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class SimpleToGeneral(BaseInferenceAlgorithm):
    """
    Simple-to-General (S2G) algorithm matching s2g.m and getvar.m (Crampin et al. / Judd & Mees).
    Iteratively adds basis functions from columns of X with maximal projection of residual error,
    swaps out negligible basis functions via getanother(), and stops when the MDL/AIC log penalty
    C(k) = N * log(mean(E^2)) + k * log(N) stops improving.
    """

    def __init__(
        self,
        criterion: str = "mdl",  # "mdl" (exact getvar.m line 96) or "aic"
        max_interactions: Optional[int] = None,
        normalize_bases: bool = True,
        threshold: float = 0.0,
        matlab_mode: bool = True,
        penalty_weight: float = 1.0,
        min_r2_improvement: float = 0.0,
    ):
        super().__init__(name="s2g")
        self.criterion = criterion.lower()
        self.max_interactions = max_interactions
        self.normalize_bases = normalize_bases
        self.threshold = threshold
        self.matlab_mode = matlab_mode
        self.penalty_weight = penalty_weight
        self.min_r2_improvement = min_r2_improvement

    def _penalty(self, E: np.ndarray, k: int, penalty_weight: float = 1.0) -> float:
        """
        Computes cost function matching getvar.m lines 93-96 with optional penalty weighting:
            function C = penalty(E, k)
            N = length(E);
            S = mean(E.^2);
            C = N*log(S) + (k)*log(N);
        """
        n_len = len(E)
        s_mse = float(np.mean(E ** 2))
        eff_k = float(k * penalty_weight)
        if self.criterion == "aic":
            return float(np.sum(E ** 2)) + eff_k
        s_safe = max(s_mse, 1e-300)
        return float(n_len * np.log(s_safe) + eff_k * np.log(n_len))

    def _getanother(
        self,
        y: np.ndarray,
        V: np.ndarray,
        ind: List[int],
        E: np.ndarray,
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Exact port of getanother(y, V, ind, E) in getvar.m lines 49-89.
        """
        k = len(ind)
        n_cols = V.shape[1]
        iin = -1
        iout = -2
        ind_ext = list(ind) + [0]
        a = np.zeros(k + 1, dtype=float)

        max_iter = max(n_cols * 4, 20)
        it = 0
        while iout != iin and it < max_iter:
            it += 1
            # Project to identify next iin: mu = abs(V' * E)
            mu = np.abs(V.T @ E)
            if it > n_cols:
                # Prevent infinite 2-cycle due to floating-point ties
                for existing in ind_ext[:k]:
                    mu[existing] = -1.0
            iin = int(np.argmax(mu))
            ind_ext[k] = iin

            # Update basis set
            Vb = V[:, ind_ext]

            # Identify iout with lowest absolute coefficient
            a = np.linalg.pinv(Vb) @ y
            aa = np.abs(a)
            imin = int(np.argmin(aa))
            iout = ind_ext[imin]

            # Remove basis if iout ~= iin
            if iout != iin:
                ind_ext[imin] = iin
                Vk = V[:, ind_ext[:k]]
                ak = np.linalg.pinv(Vk) @ y
                yh = Vk @ ak
                E = yh - y
            else:
                yh = Vb @ a
                E = yh - y

        return ind_ext, a, E

    def _getvar(
        self,
        y: np.ndarray,
        X: np.ndarray,
        penalty_weight: float = 1.0,
        min_r2_improvement: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact port of [ind, a, S] = getvar(y, X) in getvar.m lines 1-47.
        """
        N, M = X.shape
        if self.normalize_bases:
            xnorm = np.sqrt(np.sum(X ** 2, axis=0))
            xnorm_safe = np.where(xnorm > 1e-15, xnorm, 1.0)
            V = X / xnorm_safe[None, :]
        else:
            V = X.copy()

        max_k = min(self.max_interactions or M, M)

        # Calculate first basis vector (getvar.m lines 23-33)
        k = 1
        mu = np.abs(V.T @ y)
        iin = int(np.argmax(mu))
        ind = [iin]
        Vb = V[:, ind]
        a = np.linalg.pinv(Vb) @ y
        yh = Vb @ a
        E = yh - y
        total_var = float(np.sum(y ** 2))
        s_k = self._penalty(E, 1, penalty_weight=penalty_weight)
        s_old = s_k + 1.0

        while s_k < s_old and k <= max_k:
            s_prev = s_k
            cand_ind, cand_a, cand_E = self._getanother(y, V, ind, E)
            if min_r2_improvement > 0.0 and total_var > 1e-15:
                old_sse = float(np.sum(E ** 2))
                new_sse = float(np.sum(cand_E ** 2))
                delta_r2 = (old_sse - new_sse) / total_var
                if delta_r2 < min_r2_improvement:
                    break
            ind, a, E = cand_ind, cand_a, cand_E
            k += 1
            s_k = self._penalty(E, k, penalty_weight=penalty_weight)
            s_old = s_prev

        # ind = ind(1:end-1) (getvar.m line 44)
        ind = ind[:-1]

        Xb = X[:, ind]
        final_a = np.linalg.pinv(Xb) @ y
        return np.array(ind, dtype=int), final_a

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Fits the interaction matrix matching s2g.m / s2g_and_g2s.m:
            Z = X;
            for i=1:N
                [ind0, a0, S0] = getvar(U(i,:)', Z);
                A_sg(ind0, i) = a0;
            end
        """
        start_time = time.time()
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)
        n = X_f.shape[0]
        A_est = np.zeros((n, n), dtype=float)

        penalty_w = float(kwargs.get("penalty_weight", self.penalty_weight))
        min_r2 = float(kwargs.get("min_r2_improvement", self.min_r2_improvement))

        for i in range(n):
            y = U_f[i, :].copy()
            if np.all(np.abs(y) < 1e-15):
                y = X_f[i, :].copy()

            if self.matlab_mode:
                indices, coeffs = self._getvar(y, X_f, penalty_weight=penalty_w, min_r2_improvement=min_r2)
                for idx, c in zip(indices, coeffs):
                    if abs(c) > self.threshold:
                        A_est[idx, i] = c
            else:
                indices, coeffs = self._getvar(-y, X_f.T, penalty_weight=penalty_w, min_r2_improvement=min_r2)
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
            diagnostics={"criterion": self.criterion, "matlab_mode": self.matlab_mode},
        )
