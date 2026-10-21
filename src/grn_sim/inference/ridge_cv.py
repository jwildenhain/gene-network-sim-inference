"""
Kaggle Multi-Output Ridge Regression (Kouagou Ensemble Baseline).
Based on N'Dah Jean Kouagou's winning methodology for the NeurIPS 2023
Open Problems in Single-Cell Perturbations competition.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import RidgeCV

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


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
        allow_self_regulation: bool = False,
    ):
        super().__init__(name="ridge_cv")
        self.alphas = alphas or [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
        self.k_interactions = k_interactions
        self.threshold = threshold
        self.cv = cv
        self.perturbation_mode = perturbation_mode.lower().strip()
        self.allow_self_regulation = allow_self_regulation

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)

        pert_mode = kwargs.get("perturbation_mode", self.perturbation_mode).lower().strip()
        if pert_mode == "regulator":
            U_mat = -U_mat
        elif pert_mode == "auto":
            if np.all(np.diag(U_mat) > 0) and np.median(np.diag(X_mat)) > 0:
                U_mat = -U_mat

        N, M = X_mat.shape

        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        thresh = float(kwargs.get("threshold", self.threshold))
        allow_self = bool(kwargs.get("allow_self_regulation", self.allow_self_regulation))

        A_est = np.zeros((N, N), dtype=float)
        best_alphas = []

        # Fit each target gene i from regressors
        for i in range(N):
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

            best_alphas.append(float(ridge.alpha_))
            for idx, reg in enumerate(reg_idx):
                A_est[i, reg] = ridge.coef_[idx]

        if not allow_self:
            np.fill_diagonal(A_est, 0.0)

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
            diagnostics={"mean_alpha": float(np.mean(best_alphas))},
        )
