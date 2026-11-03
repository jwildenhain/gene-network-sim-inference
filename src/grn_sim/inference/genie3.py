"""
GENIE3: GEne Network Inference with Ensemble of trees.
The winning algorithm of the DREAM4 and DREAM5 GRN Reconstruction Challenges
(Huynh-Thu et al., PLoS ONE 2010).
"""

import time
from typing import Optional
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from grn_sim.inference._parallel import default_n_jobs, map_indices
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
        n_jobs: Optional[int] = None,
    ):
        super().__init__(name="genie3")
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.tree_type = tree_type
        self.k_interactions = k_interactions
        self.threshold = threshold
        self.seed = seed

    def _target_importances(self, payload, i: int) -> np.ndarray:
        """Normalised feature importances of target ``i`` over the other genes (in index order)."""
        X_mat, n_trees, seed = payload
        N = X_mat.shape[0]
        target = X_mat[i, :]
        reg_idx = [j for j in range(N) if j != i]   # no self-loop candidates
        features = X_mat[reg_idx, :].T              # (M, N-1)
        cls = RandomForestRegressor if self.tree_type == "random_forest" else ExtraTreesRegressor
        model = cls(n_estimators=n_trees, max_features="sqrt", random_state=seed + i, n_jobs=1)
        model.fit(features, target)
        importances = model.feature_importances_
        total = np.sum(importances)
        return importances / total if total > 0 else importances

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

        # Regress each target gene i against all candidate regulators j != i (targets are independent)
        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=16)
        rows = map_indices(self._target_importances, (X_mat, n_trees, seed), N, n_jobs)
        for i, importances in enumerate(rows):
            reg_idx = [j for j in range(N) if j != i]
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
