"""
SFINN-style scoring: perturbation sensitivity weighted by a shared-factor neighborhood association.
Inspired by SFINN (Briefings in Bioinformatics 2024, "Shared factor neighborhood and integrated neural network
for gene regulatory network inference"); this is a deterministic scoring rule, not a neural network.
"""

import time
from typing import Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


class SFINNAlgorithm(BaseInferenceAlgorithm):
    """
    Scores every (target i, regulator j) pair as

        score[i, j] = |S[i, j]| * (1 + |G[i, j]|),

    where S is the perturbation sensitivity matrix (``-X pinv(U)``: the response of gene i to perturbing gene j)
    and G is a gene-gene association computed in a shared-factor space: the conditions are smoothed over their
    nearest neighbours (a cosine kernel between condition profiles, row-normalised) before the genes are
    correlated. The interventional term carries the causal direction and sign; the shared-factor term favours
    pairs that also co-vary once the condition-level noise is averaged out. Smoothing S itself over gene
    neighbourhoods was tried and lowered F1, so it is not done.

    The result is A[target, regulator] (rows are targets), like ``S``. Edges are the top-K regulators per target
    (K = ``k_interactions``) or, without K, the entries above 0.8 times the row mean.

    ``embed_dim``, ``hidden_dim`` and ``seed`` belong to the earlier randomly initialised network and are ignored;
    they stay so existing callers keep working.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        k_interactions: Optional[int] = None,
        seed: int = 42,
        use_data_prep: bool = True,
    ):
        super().__init__(name="sfinn")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.k_interactions = k_interactions
        self.seed = seed
        self.use_data_prep = use_data_prep

    @staticmethod
    def _shared_factor_association(X_feats: np.ndarray) -> np.ndarray:
        """Gene-gene cosine association after smoothing the conditions over their positive-similarity neighbours."""
        col_norms = np.linalg.norm(X_feats, axis=0, keepdims=True) + 1e-8
        X_normed = X_feats / col_norms
        K_cond = np.maximum(0.0, X_normed.T @ X_normed)                  # (M, M) condition kernel
        K_norm = K_cond / (np.sum(K_cond, axis=1, keepdims=True) + 1e-8)
        F_shared = X_feats @ K_norm
        gene_cov = F_shared @ F_shared.T
        d = np.diag(gene_cov)
        G = gene_cov / (np.sqrt(np.outer(d, d)) + 1e-8)
        np.fill_diagonal(G, 0.0)
        return G

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        use_prep = kwargs.get("use_data_prep", self.use_data_prep)
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        partial = M != N or bool(np.any(np.all(np.abs(U_mat) < 1e-15, axis=0)))
        if partial:
            # Partial screen (fewer perturbation columns than genes): the sensitivity of every gene to each perturbed gene is
            # the column of X divided by that gene's own perturbation; columns of genes that were never perturbed stay 0.
            from grn_sim.inference.partial import square_subproblem
            P, cols, _, _ = square_subproblem(X_mat, U_mat)
            u = np.where(np.abs(U_mat[P, cols]) < 1e-12, 1e-12, U_mat[P, cols])
            S_dir = np.zeros((N, N))
            S_dir[:, P] = -X_mat[:, cols] / u[None, :]                   # = -X pinv(U) for a diagonal U
            X_feats = X_mat - np.median(X_mat, axis=1, keepdims=True)
        elif use_prep:
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            X_feats = prep.Delta_X
            S_dir = prep.sensitivity_mat
        else:
            X_feats = X_mat
            try:
                pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
                S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
            except Exception:
                S_dir = -X_mat[:, :N] if M >= N else np.zeros((N, N))

        G_shared = self._shared_factor_association(X_feats)

        scores = np.abs(S_dir) * (1.0 + np.abs(G_shared))
        np.fill_diagonal(scores, 0.0)
        sign = np.where(np.abs(S_dir) > 1e-4, np.sign(S_dir), np.sign(G_shared))
        sign = np.where(sign == 0, 1.0, sign)
        A_est = sign * scores

        adj_pred = np.zeros((N, N), dtype=int)
        for i in range(N):
            row = scores[i, :]
            if K is not None:
                keep = np.argsort(row)[::-1][:K]
                keep = keep[row[keep] > 1e-4]
            else:
                row_thresh = np.mean(row) * 0.8 if np.any(row > 0) else 0.1
                keep = np.flatnonzero((row > row_thresh) & (row > 1e-4))
            adj_pred[i, keep] = np.sign(A_est[i, keep]).astype(int)

        return InferenceResult(
            algorithm_name="sfinn",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time.time() - start_time,
            diagnostics={
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
                "use_data_prep": use_prep,
                "partial_screen": partial,
            },
        )
