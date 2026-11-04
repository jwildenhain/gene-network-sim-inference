"""
Kaggle Multi-Output Ridge Regression (Kouagou Ensemble Baseline).
Based on N'Dah Jean Kouagou's winning methodology for the NeurIPS 2023
Open Problems in Single-Cell Perturbations competition.
"""

import time
from typing import Optional
import numpy as np
from sklearn.linear_model import RidgeCV

from grn_sim.inference._parallel import default_n_jobs, map_indices
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult, regulator_input_fit


class KaggleMultiOutputRidge(BaseInferenceAlgorithm):
    """
    Multi-target regularized Ridge regression with out-of-fold cross-validation
    and empirical covariance shrinkage across perturbation targets.
    """

    def __init__(
        self,
        alphas: Optional[list] = None,
        k_interactions: Optional[int] = None,
        threshold: float = 0.05,
        cv: int = 5,
        perturbation_mode: str = "additive",
        allow_self_regulation: Optional[bool] = None,
        adaptive_k: bool = False,
        stability_boots: int = 0,
        stability_freq: float = 0.5,
        seed: int = 0,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(name="ridge_cv")
        self.n_jobs = n_jobs
        self.alphas = alphas or [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        self.k_interactions = k_interactions
        self.threshold = threshold
        self.cv = cv
        self.perturbation_mode = perturbation_mode.lower().strip()
        # None: the perturbed gene's own term is a regressor in the additive model (A X = -U needs it; without it the
        # fit is near random, see docs/INFERENCE_GAP_ANALYSIS.md) and is left out in the regulator-input model
        self.allow_self_regulation = allow_self_regulation
        self.adaptive_k = adaptive_k
        self.stability_boots = stability_boots
        self.stability_freq = stability_freq
        self.seed = seed

    def _fit_target(self, payload, i: int):
        """Ridge fit of target gene ``i``: (coefficients over its regressors, chosen alpha, regressor indices)."""
        X_mat, U_mat, K, allow_self, kwargs = payload
        N, M = X_mat.shape
        if allow_self:
            reg_idx = list(range(N))
        else:
            reg_idx = [j for j in range(N) if j != i]

        # Design matrix: expression of regressors, shape (M, len(reg_idx))
        features = X_mat[reg_idx, :].T
        # Target: negative perturbation plus response
        target = -U_mat[i, :]

        # RidgeCV with cross validation
        cv_folds = min(self.cv, max(2, M // 2))
        ridge = RidgeCV(alphas=self.alphas, cv=cv_folds)
        ridge.fit(features, target)
        coefs = np.array(ridge.coef_, dtype=float)

        B = int(kwargs.get("stability_boots", self.stability_boots))
        if B > 0 and K is not None:
            # Stability selection: keep regulators in the top-K for >= stability_freq of bootstraps
            rng = np.random.default_rng(self.seed + i)
            freq = np.zeros(len(reg_idx))
            for _ in range(B):
                samp = rng.integers(0, M, M)
                rb = RidgeCV(alphas=[float(ridge.alpha_)]).fit(features[samp], target[samp])
                freq[np.argsort(-np.abs(rb.coef_))[:K]] += 1.0
            coefs = np.where(freq / B >= float(kwargs.get("stability_freq", self.stability_freq)), coefs, 0.0)

        if bool(kwargs.get("adaptive_k", self.adaptive_k)) and K is not None and K > 1:
            # Per-row model order: OLS refit on top-k coefficients, choose k by BIC
            order = np.argsort(-np.abs(coefs))[:K]
            best = None
            for kk in range(1, K + 1):
                cols = order[:kk]
                a_try = np.linalg.pinv(features[:, cols]) @ target
                sse = float(np.sum((target - features[:, cols] @ a_try) ** 2))
                bic = M * np.log(max(sse, 1e-12) / M) + kk * np.log(M)
                if best is None or bic < best[0]:
                    best = (bic, cols)
            assert best is not None  # K >= 1, so the loop ran
            keep = np.zeros(len(reg_idx), dtype=bool)
            keep[best[1]] = True
            coefs = np.where(keep, coefs, 0.0)

        return coefs, float(ridge.alpha_), reg_idx

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)

        pert_mode = kwargs.get("perturbation_mode", self.perturbation_mode).lower().strip()
        if pert_mode == "regulator":
            kwargs.setdefault("allow_self_regulation", False if self.allow_self_regulation is None
                              else self.allow_self_regulation)
            return regulator_input_fit(self, X_mat, U_mat, **kwargs)

        N, M = X_mat.shape

        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        thresh = float(kwargs.get("threshold", self.threshold))
        allow_self = kwargs.get("allow_self_regulation", self.allow_self_regulation)
        allow_self = True if allow_self is None else bool(allow_self)

        A_est = np.zeros((N, N), dtype=float)
        best_alphas = []

        # Fit each target gene i from regressors (targets are independent)
        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=16)
        payload = (X_mat, U_mat, K, allow_self, kwargs)
        for i, (coefs, alpha, reg_idx) in enumerate(map_indices(self._fit_target, payload, N, n_jobs)):
            best_alphas.append(alpha)
            for idx, reg in enumerate(reg_idx):
                A_est[i, reg] = coefs[idx]

        self_term = np.diag(A_est).copy()
        np.fill_diagonal(A_est, 0.0)         # the own-gene term is a nuisance regressor, not a self-loop

        # Discretization
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
            algorithm_name="ridge_cv",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=elapsed,
            diagnostics={"mean_alpha": float(np.mean(best_alphas)), "mean_self_term": float(np.mean(self_term))},
        )
