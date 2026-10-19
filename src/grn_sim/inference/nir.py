"""
Network Identification by Multiple Regression (NIR) algorithm.
Implements Gardner et al. (Science 2003) and nir_v3.m with forward-topD greedy search
and ridge regression.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class NIRAlgorithm(BaseInferenceAlgorithm):
    """
    Network Identification by Multiple Regression (NIR) matching nir_v3.m:
    Finds a fixed number K of regulatory connections per gene by minimizing
    the sum of squared errors (SSE) using forward-topD greedy search and
    multiple linear regression.
    """

    def __init__(
        self,
        k_interactions: int = 2,
        top_d: Optional[int] = None,
        ridge: float = 0.0,
        cond_max: float = 1000.0,
        use_perturbation_values: bool = True,
        threshold: float = 0.0,
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
        D: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Solves row i using forward top-D greedy search matching nir_v3.m lines 49-130.
        """
        N, M = X.shape
        u = P[i, :].copy()
        Xtmp = X.copy()
        nopert = False

        if not self.use_perturbation_values:
            keep_cols = [c for c in range(M) if abs(u[c]) <= 1e-15]
            Xtmp = X[:, keep_cols]
            u = np.zeros(len(keep_cols), dtype=float)

        # If gene i is not perturbed, extract row i and use as perturbation (nir_v3.m lines 69-75)
        if not np.any(np.abs(u) > 1e-15):
            u = -Xtmp[i, :].copy()
            keep_rows = [r for r in range(N) if r != i]
            Xtmp = Xtmp[keep_rows, :]
            nopert = True

        Ntmp = Xtmp.shape[0]
        Ktmp = max(1, min(K - (1 if nopert else 0), Ntmp))
        D_eff = max(1, min(D, Ntmp))

        topD: Optional[np.ndarray] = None
        ncoeff = 0
        best_combo = np.array([], dtype=int)
        best_coeffs = np.array([], dtype=float)
        best_sse = np.inf

        while ncoeff < Ktmp:
            nnewcombo = Ntmp - ncoeff
            if ncoeff == 0:
                subN = np.arange(Ntmp, dtype=int)[:, None]
            else:
                assert topD is not None
                d_cur = topD.shape[0]
                subN = np.zeros((d_cur * nnewcombo, ncoeff + 1), dtype=int)
                all_idx = np.arange(Ntmp, dtype=int)
                for ii in range(d_cur):
                    rem = np.setdiff1d(all_idx, topD[ii, :], assume_unique=False)
                    start_r = ii * nnewcombo
                    end_r = (ii + 1) * nnewcombo
                    subN[start_r:end_r, :ncoeff] = topD[ii, :]
                    subN[start_r:end_r, ncoeff] = rem

            combinations = subN.shape[0]
            aest = np.zeros((combinations, ncoeff + 1), dtype=float)
            SSEK_tmp = np.full(combinations, np.inf, dtype=float)
            reg_eye = self.ridge * np.eye(ncoeff + 1)

            for j in range(combinations):
                Z = Xtmp[subN[j, :], :]
                ZZt = Z @ Z.T
                cond_val = np.linalg.cond(ZZt) if ZZt.size > 1 else (1.0 if ZZt[0, 0] > 0 else np.inf)
                if cond_val <= self.cond_max:
                    try:
                        aest[j, :] = -u @ Z.T @ np.linalg.inv(ZZt + reg_eye)
                        SSEK_tmp[j] = float(np.sum((u + aest[j, :] @ Z) ** 2))
                    except np.linalg.LinAlgError:
                        pass

            # Sort errors (mergesort preserves MATLAB's stable tie-breaking order)
            srtidxSSE = np.argsort(SSEK_tmp, kind="mergesort")
            d_keep = min(D_eff, combinations)
            topDidx = srtidxSSE[:d_keep]
            topD = subN[topDidx, :]

            if ncoeff == Ktmp - 1:
                best_combo = topD[0, :].copy()
                best_coeffs = aest[topDidx[0], :].copy()
                best_sse = float(SSEK_tmp[topDidx[0]])

            ncoeff += 1

        if nopert:
            best_combo = np.where(best_combo >= i, best_combo + 1, best_combo)

        return best_combo, best_coeffs, best_sse, nopert

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Fits regulatory interaction matrix using NIR matching nir_v3.m.
        """
        start_time = time.time()
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)
        N, M = X_f.shape
        K = int(kwargs.get("k", self.k_interactions))
        K = max(1, min(K, N))
        D = int(kwargs.get("top_d", self.top_d if self.top_d is not None else min(N, 10)))

        A_est = np.zeros((N, N), dtype=float)
        diagnostics = {"sse": [], "r2": []}

        for i in range(N):
            indices, coeffs, sse, nopert = self._fit_single_row(i, X_f, U_f, K, D)
            if nopert:
                A_est[i, i] = -1.0
            for idx, c in zip(indices, coeffs):
                if abs(c) > self.threshold:
                    A_est[i, idx] = c
            sse0 = float(np.sum(U_f[i, :] ** 2))
            r2 = 1.0 - (sse / sse0) if sse0 > 1e-15 else 0.0
            diagnostics["sse"].append(sse)
            diagnostics["r2"].append(r2)

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
