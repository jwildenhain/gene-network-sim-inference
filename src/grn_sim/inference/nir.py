"""
Network Identification by Multiple Regression (NIR) algorithm.
Implements Gardner et al. (Science 2003) and nir_v3.m with forward-topD greedy search
and ridge regression.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from grn_sim.inference._parallel import default_n_jobs, map_indices
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult, regulator_input_fit


class NIRAlgorithm(BaseInferenceAlgorithm):
    """
    Network Identification by Multiple Regression (NIR) matching nir_v3.m:
    Finds a fixed number K of regulatory connections per gene by minimizing
    the sum of squared errors (SSE) using forward-topD greedy search and
    multiple linear regression.
    Supports adaptive model-order selection (BIC) and robust regularized solving.
    """

    def __init__(
        self,
        k_interactions: int = 2,
        top_d: Optional[int] = None,
        ridge: float = 0.0,
        cond_max: float = 1000.0,
        use_perturbation_values: bool = True,
        threshold: float = 0.0,
        adaptive_k: bool = False,
        perturbation_mode: str = "additive",
        n_jobs: Optional[int] = None,
    ):
        super().__init__(name="nir")
        self.n_jobs = n_jobs
        self.k_interactions = k_interactions
        self.top_d = top_d
        self.ridge = ridge
        self.cond_max = cond_max
        self.use_perturbation_values = use_perturbation_values
        self.threshold = threshold
        self.adaptive_k = adaptive_k
        self.perturbation_mode = perturbation_mode.lower().strip()

    def _fit_single_row(
        self,
        i: int,
        X: np.ndarray,
        P: np.ndarray,
        K: int,
        D: int,
        adaptive_k: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Solves row i using forward top-D greedy search matching nir_v3.m lines 49-130.
        Supports optional BIC model-order selection when adaptive_k=True.
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
        level_candidates = []

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

            self._score_combinations(Xtmp, u, subN, reg_eye, adaptive_k, aest, SSEK_tmp)

            # Sort errors (mergesort preserves MATLAB's stable tie-breaking order)
            srtidxSSE = np.argsort(SSEK_tmp, kind="mergesort")
            d_keep = min(D_eff, combinations)
            topDidx = srtidxSSE[:d_keep]
            topD = subN[topDidx, :]

            curr_sse = float(SSEK_tmp[topDidx[0]])
            curr_combo = topD[0, :].copy()
            curr_coeffs = aest[topDidx[0], :].copy()
            n_obs = max(1, len(u))
            bic = n_obs * np.log(max(curr_sse, 1e-12) / n_obs) + (ncoeff + 1) * np.log(n_obs)
            level_candidates.append((curr_combo, curr_coeffs, curr_sse, bic))

            if ncoeff == Ktmp - 1:
                best_combo = curr_combo
                best_coeffs = curr_coeffs
                best_sse = curr_sse

            ncoeff += 1

        if adaptive_k and level_candidates:
            # Pick candidate minimizing BIC
            best_by_bic = min(level_candidates, key=lambda c: c[3])
            best_combo, best_coeffs, best_sse, _ = best_by_bic

        if nopert:
            best_combo = np.where(best_combo >= i, best_combo + 1, best_combo)

        return best_combo, best_coeffs, best_sse, nopert

    def _fit_row(self, payload, i: int):
        X_f, U_f, K, D, adaptive = payload
        return self._fit_single_row(i, X_f, U_f, K, D, adaptive_k=adaptive)

    def _score_combinations(self, Xtmp, u, subN, reg_eye, adaptive_k, aest, SSEK_tmp, chunk: int = 1024) -> None:
        """
        Least-squares fit of every candidate regressor set in ``subN`` (rows), in batches: coefficients go to ``aest``
        and the squared error to ``SSEK_tmp``. Same rule as the one-set-at-a-time loop of nir_v3.m: well-conditioned
        sets use the inverse, ill-conditioned ones the pseudo-inverse (when ridge > 0 or adaptive_k), else skipped.
        """
        combinations, k1 = subN.shape
        use_pinv = self.ridge > 0 or adaptive_k
        for start in range(0, combinations, chunk):
            idx = slice(start, min(start + chunk, combinations))
            Z = Xtmp[subN[idx, :], :]                                  # (c, k1, M)
            ZZt = Z @ Z.transpose(0, 2, 1)                             # (c, k1, k1)
            uZ = np.einsum("m,ckm->ck", -u, Z)                         # -u @ Z.T per set
            if k1 > 1:
                cond = np.linalg.cond(ZZt)
            else:
                cond = np.where(ZZt[:, 0, 0] > 0, 1.0, np.inf)
            good = np.flatnonzero(cond <= self.cond_max)                # includes nothing for nan
            bad = np.flatnonzero(~(cond <= self.cond_max))
            for sel, solver in ((good, np.linalg.inv), (bad, np.linalg.pinv if use_pinv else None)):
                if sel.size == 0 or solver is None:
                    continue
                try:
                    inv = solver(ZZt[sel] + reg_eye)
                    coef = np.einsum("ck,ckl->cl", uZ[sel], inv)
                    ok = np.arange(sel.size)
                except (np.linalg.LinAlgError, ValueError):             # a singular set: go one at a time
                    coef = np.zeros((sel.size, k1))
                    ok_list = []
                    for q, c in enumerate(sel):
                        try:
                            coef[q] = uZ[c] @ solver(ZZt[c] + reg_eye)
                            ok_list.append(q)
                        except Exception:
                            pass
                    ok = np.array(ok_list, dtype=int)
                if ok.size == 0:
                    continue
                rows = start + sel[ok]
                aest[rows, :] = coef[ok]
                resid = u[None, :] + np.einsum("ck,ckm->cm", coef[ok], Z[sel[ok]])
                SSEK_tmp[rows] = np.sum(resid ** 2, axis=1)

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Fits regulatory interaction matrix using NIR matching nir_v3.m.
        """
        start_time = time.time()
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)

        pert_mode = kwargs.get("perturbation_mode", self.perturbation_mode).lower().strip()
        if pert_mode == "regulator":
            return regulator_input_fit(self, X_f, U_f, **kwargs)

        N, M = X_f.shape
        K = int(kwargs.get("k", self.k_interactions))
        K = max(1, min(K, N))
        D = int(kwargs.get("top_d", self.top_d if self.top_d is not None else min(N, 10)))
        adaptive = bool(kwargs.get("adaptive_k", self.adaptive_k))

        A_est = np.zeros((N, N), dtype=float)
        diagnostics: Dict[str, List[float]] = {"sse": [], "r2": []}

        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=16)
        fits = map_indices(self._fit_row, (X_f, U_f, K, D, adaptive), N, n_jobs)
        for i, (indices, coeffs, sse, nopert) in enumerate(fits):
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
