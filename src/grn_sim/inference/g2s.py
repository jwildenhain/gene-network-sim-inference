"""
General-to-Specific (G2S) reverse engineering algorithm.
Implements the exact MATLAB g2s.m and gen2spec.m backward elimination basis pruning
technique with AIC minimization from thesis Chapter 3.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class GeneralToSpecific(BaseInferenceAlgorithm):
    """
    General-to-Specific (G2S) algorithm matching g2s.m and gen2spec.m (Crampin et al.).
    Starts with the full set of basis functions (columns of X) and systematically removes
    the interaction causing the least increase in mean squared error via dropanother()
    until the AIC cost function Caic = sum(E^2) + k stops decreasing.
    """

    def __init__(
        self,
        criterion: str = "aic",
        min_interactions: int = 1,
        threshold: float = 0.0,
        matlab_mode: bool = True,
        fast_loo_threshold: int = 25,
        normalized_penalty: bool = False,
    ):
        super().__init__(name="g2s")
        self.criterion = criterion.lower()
        self.min_interactions = min_interactions
        self.threshold = threshold
        self.matlab_mode = matlab_mode
        self.fast_loo_threshold = fast_loo_threshold
        self.normalized_penalty = normalized_penalty

    def _penalty(self, E: np.ndarray, k: int, normalized: bool = False) -> float:
        """
        Computes AIC cost function matching gen2spec.m lines 73-75:
            function C = penalty(E, k)
            Caic = sum(E.^2) + k;
            C = Caic;
        Supports optional scale-normalized AIC/MDL for variance invariance.
        """
        rss = float(np.sum(E ** 2))
        if self.criterion == "mdl":
            n_len = len(E)
            mse = max(rss / n_len, 1e-300)
            return float(n_len * np.log(mse) + k * np.log(n_len))
        elif self.criterion == "aic_scaled" or normalized or self.normalized_penalty:
            n_len = len(E)
            mse = max(rss / n_len, 1e-300)
            return float(n_len * np.log(mse) + 2.0 * k)
        return rss + float(k)

    def _dropanother_exact(
        self,
        y: np.ndarray,
        V: np.ndarray,
        ind: List[int],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Exact port of dropanother(y, V, ind) in gen2spec.m lines 53-70:
            for i=1:k
               ind0 = ind([1:i-1 i+1:k]);
               Vb = V(:,ind0);
               a = pinv(Vb)*y;
               yh = Vb*a;
               e = yh - y;
               E0(i) = mean(e.^2);
            end
            [E0min, imin] = min(E0);
            ind1 = ind([1:imin-1 imin+1:k]);
        """
        k = len(ind)
        e0 = np.zeros(k, dtype=float)
        for i in range(k):
            ind0 = ind[:i] + ind[i + 1:]
            Vb = V[:, ind0]
            a = np.linalg.pinv(Vb) @ y
            yh = Vb @ a
            e = yh - y
            e0[i] = float(np.mean(e ** 2))

        imin = int(np.argmin(e0))
        ind1 = ind[:imin] + ind[imin + 1:]
        Vb1 = V[:, ind1]
        a1 = np.linalg.pinv(Vb1) @ y
        yh1 = Vb1 @ a1
        E1 = yh1 - y
        return ind1, E1

    def _gen2spec(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact port of [ind, a] = gen2spec(y, X) in gen2spec.m lines 1-50.
        Uses exact pinv loop for N <= fast_loo_threshold and Sherman-Morrison LOO
        acceleration for large N.
        """
        N, M = X.shape
        V = X
        k = M
        ind = list(range(M))
        use_fast = M > self.fast_loo_threshold

        Vb = V[:, ind]
        Pinv = np.linalg.pinv(Vb)
        alpha = Pinv @ y
        yh = Vb @ alpha
        E = yh - y
        s_k = self._penalty(E, k)
        s_old = s_k + 1.0

        indopt = list(ind)
        while k > self.min_interactions and s_k < s_old:
            indopt = list(ind)
            s_prev = s_k
            if not use_fast:
                ind, E = self._dropanother_exact(y, V, ind)
            else:
                row_norms_sq = np.sum(Pinv ** 2, axis=1)
                base_sse = float(np.sum(E ** 2))
                e0 = (base_sse + (alpha ** 2) / np.maximum(row_norms_sq, 1e-30)) / N
                imin = int(np.argmin(e0))
                ind = ind[:imin] + ind[imin + 1:]
                Vb = V[:, ind]
                Pinv = np.linalg.pinv(Vb)
                alpha = Pinv @ y
                E = (Vb @ alpha) - y
            k -= 1
            s_k = self._penalty(E, k)
            s_old = s_prev

        Xb = X[:, indopt]
        final_a = np.linalg.pinv(Xb) @ y
        return np.array(indopt, dtype=int), final_a

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Infers regulatory matrix matching g2s.m / s2g_and_g2s.m:
            Z = X;
            for i=1:N
                [ind1, a1] = gen2spec(U(i,:)', Z);
                A_gs(ind1, i) = a1;
            end
        """
        start_time = time.time()
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)
        n = X_f.shape[0]
        A_est = np.zeros((n, n), dtype=float)

        for i in range(n):
            y = U_f[i, :].copy()
            if np.all(np.abs(y) < 1e-15):
                y = X_f[i, :].copy()

            if self.matlab_mode:
                indices, coeffs = self._gen2spec(y, X_f)
                for idx, c in zip(indices, coeffs):
                    if abs(c) > self.threshold:
                        A_est[idx, i] = c
            else:
                indices, coeffs = self._gen2spec(-y, X_f.T)
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
