"""
By-Chance baseline algorithm.
Randomly infers edges with specified connectivity and activator/inhibitor ratio,
establishing the statistical null hypothesis benchmark from the thesis.
"""

import time
from typing import Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class ByChance(BaseInferenceAlgorithm):
    """
    Random null benchmark matching by_chance.m from the thesis.
    Randomly connects nodes with given connectivity and activator/inhibitor ratio r.
    """

    def __init__(
        self,
        k_interactions: int = 2,
        ratio: float = 0.5,
        seed: Optional[int] = None,
    ):
        super().__init__(name="by_chance")
        self.k_interactions = k_interactions
        self.ratio = ratio
        self.seed = seed

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        N, M = X.shape
        k_tot = kwargs.get("total_edges", self.k_interactions * N)
        rng = np.random.default_rng(kwargs.get("seed", self.seed))

        adj = np.zeros((N, N), dtype=int)
        A_est = np.zeros((N, N), dtype=float)

        all_pairs = [(i, j) for i in range(N) for j in range(N)]
        chosen_indices = rng.choice(len(all_pairs), size=min(k_tot, len(all_pairs)), replace=False)

        for idx in chosen_indices:
            i, j = all_pairs[idx]
            sign_edge = 1 if rng.uniform(0.0, 1.0) < self.ratio else -1
            adj[i, j] = sign_edge
            A_est[i, j] = float(sign_edge)

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
        )
