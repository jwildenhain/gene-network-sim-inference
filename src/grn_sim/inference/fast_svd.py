"""
FAST algorithm for reverse engineering gene networks.
Implements the SVD decomposition and sparse row truncation method of Farina et al.
matching fast.m and fast_modified.m from the thesis.
"""

import time
from typing import Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class FastSVD(BaseInferenceAlgorithm):
    """
    FAST algorithm (Farina et al.) matching fast_modified.m and fast.m:
    1. [U, S, V] = svd(x)
    2. Ur = U(:, 1:N-1), Sr = S(1:N-1, 1:N-1), Vr = V(:, 1:N-1)
    3. A_0 = (x - u) * Vr * inv(Sr) * Ur'
    4. Select top K largest |A_0(i, :)| per row and refit via -u(i, :) * pinv(x_row_sel)
    5. Apply self-regulation diagonal correction (+ eye(N)) when N == 9 (SOS benchmark)
       or when add_diagonal_identity=True (matching fast.m line 85 / new_stats.m line 52).
    """

    def __init__(
        self,
        k_interactions: int = 2,
        threshold: float = 0.0,
        add_diagonal_identity: Optional[bool] = None,
    ):
        super().__init__(name="fast")
        self.k_interactions = k_interactions
        self.threshold = threshold
        self.add_diagonal_identity = add_diagonal_identity

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Solves A * X = -U using truncated SVD and sparse Moore-Penrose refitting
        matching fast_modified.m lines 35-63.
        """
        start_time = time.time()
        x = np.real(X).astype(float)
        u = np.real(U).astype(float)
        N, M = x.shape
        K = int(kwargs.get("k", self.k_interactions))
        K = max(1, min(K, N))

        # 1. Singular Value Decomposition: [U, S, V] = svd(x)
        U_svd, S_svd, Vt_svd = np.linalg.svd(x, full_matrices=True)
        V_svd = Vt_svd.T

        # Truncate rank N-1 matching fast_modified.m lines 36-38
        rank = max(1, min(N - 1, len(S_svd) - 1))
        Ur = U_svd[:, :rank]
        Vr = V_svd[:, :rank]
        inv_Sr = np.diag(1.0 / np.maximum(S_svd[:rank], 1e-15))

        do_diag = (N == 9) if self.add_diagonal_identity is None else self.add_diagonal_identity

        # Initial estimate matching fast.m lines 40-41 / fast_modified.m line 40
        A_0 = (x - u) @ Vr @ inv_Sr @ Ur.T
        if do_diag:
            A_0 = A_0 - np.eye(N)

        # 2. Select top K elements per row with largest magnitude (fast_modified.m lines 46-63)
        ind = np.argsort(np.abs(A_0), axis=1, kind="mergesort")
        ind_M = np.sort(ind[:, N - K:N], axis=1)

        A_est = np.zeros((N, N), dtype=float)
        for i in range(N):
            sel_cols = ind_M[i, :]
            x_row_sel = x[sel_cols, :]
            a_est_m = -u[i, :] @ np.linalg.pinv(x_row_sel)
            A_est[i, sel_cols] = a_est_m

        # Self-regulation diagonal correction (fast.m line 85 / new_stats.m line 52)
        if do_diag:
            A_est = A_est + np.eye(N)

        adj = np.zeros((N, N), dtype=int)
        adj[A_est > self.threshold] = 1
        adj[A_est < -self.threshold] = -1

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
            diagnostics={"k": K, "add_diagonal_identity": do_diag},
        )
