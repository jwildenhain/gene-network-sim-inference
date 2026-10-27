"""
GENIE3: GEne Network Inference with Ensemble of trees.
The winning algorithm of the DREAM4 and DREAM5 GRN Reconstruction Challenges
(Huynh-Thu et al., PLoS ONE 2010).
"""

import time
from typing import Optional
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class GENIE3Algorithm(BaseInferenceAlgorithm):
    """
    Decomposes the network inference problem into N independent feature-selection
    regression tasks using ensembles of randomized decision trees.
    Assigns interaction sign based on empirical correlation between regulator and target.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        tree_type: str = "extra_trees",
        k_interactions: Optional[int] = None,
        threshold: float = 0.05,
        seed: int = 42,
    ):
        super().__init__(name="genie3")
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.k_interactions = k_interactions
        self.threshold = threshold
        self.seed = seed

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        N, M = X_mat.shape

        n_trees = int(kwargs.get("n_estimators", self.n_estimators))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))
        thresh = float(kwargs.get("threshold", self.threshold))
        seed = int(kwargs.get("seed", self.seed))

        # Weight matrix: W[i, j] is regulator j -> target i
        W = np.zeros((N, N), dtype=float)
        # Sign matrix based on pairwise correlation
        corr = np.corrcoef(X_mat)
        np.nan_to_num(corr, copy=False, nan=0.0)

        # Regress each target gene i against all candidate regulators j != i
        for i in range(N):
            target = X_mat[i, :]
            # Exclude self-loop from candidate regulators
            reg_idx = [j for j in range(N) if j != i]
            features = X_mat[reg_idx, :].T  # Shape (M, N-1)

            if self.tree_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=n_trees,
                    max_features="sqrt",
                    random_state=seed + i,
                    n_jobs=1,
                )
            else:
                model = ExtraTreesRegressor(
                    n_estimators=n_trees,
                    max_features="sqrt",
                    random_state=seed + i,
                    n_jobs=1,
                )

            model.fit(features, target)
            importances = model.feature_importances_

            # Normalize importances
            sum_imp = np.sum(importances)
            if sum_imp > 0:
                importances = importances / sum_imp

            for idx, reg in enumerate(reg_idx):
                sign_val = 1 if corr[reg, i] >= 0 else -1
                W[i, reg] = importances[idx] * sign_val

        np.fill_diagonal(W, 0.0)

        # Discretization
        adj_pred = np.zeros((N, N), dtype=int)
        if K is not None:
            # Top-K regulators per target gene
            for i in range(N):
                mags = np.abs(W[i, :])
                top_idx = np.argsort(mags)[-K:]
                for reg in top_idx:
                    if mags[reg] >= thresh:
                        adj_pred[i, reg] = int(np.sign(W[i, reg]))
        else:
            # Global thresholding
            max_mag = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1.0
            norm_W = W / max_mag
            mask = np.abs(norm_W) >= thresh
            adj_pred[mask] = np.sign(norm_W[mask]).astype(int)

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="genie3",
            A_est=W,
            adjacency_pred=adj_pred,
            time_elapsed=elapsed,
            diagnostics={"n_trees": n_trees, "tree_type": self.tree_type},
        )
