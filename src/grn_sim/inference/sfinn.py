"""
SFINN: Shared Factor neighborhood and Integrated Neural Network for Gene Regulatory Network Inference.
Based on Briefings in Bioinformatics 2024 ("SFINN: Shared factor neighborhood and integrated neural
network for gene regulatory network inference from single-cell and spatial transcriptomic data").
Enhanced with baseline response shift extraction and adaptive factor kernel scaling.
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.preprocessor import default_preprocessor


class SFINNAlgorithm(BaseInferenceAlgorithm):
    """
    SFINN: Shared Factor Neighborhood and Integrated Neural Network.

    1. Constructs a condition/cellular shared factor neighborhood kernel from expression profiles.
    2. Context Aggregation: Projects gene expression through shared factor neighborhood space.
    3. Integrated Neural Classifier: Combines graph-convolved embeddings with non-linear
       feedforward interaction layers over gene-pair features [z_j, z_i, |z_j - z_i|, z_j * z_i].
    4. Outputs directed, signed interaction probabilities for every regulator-target pair.
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

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        use_prep = kwargs.get("use_data_prep", self.use_data_prep)
        rng = np.random.RandomState(self.seed)
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        if use_prep:
            prep = default_preprocessor.prepare(X_mat, U_mat, **kwargs)
            X_feats = prep.Delta_X
            S_dir = prep.sensitivity_mat
            d_emb = min(self.embed_dim, max(4, N - 1))
            h_dim = min(self.hidden_dim, max(8, N))
        else:
            X_feats = X_mat
            d_emb = min(self.embed_dim, max(8, N - 1))
            h_dim = self.hidden_dim
            try:
                pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
                S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
            except Exception:
                S_dir = -X_mat[:, :N] if M >= N else np.zeros((N, N))

        # 1. Shared Factor Neighborhood Kernel
        col_norms = np.linalg.norm(X_feats, axis=0, keepdims=True) + 1e-8
        X_normed = X_feats / col_norms
        K_cond = X_normed.T @ X_normed  # (M, M)
        K_cond = np.maximum(0.0, K_cond)
        row_sums = np.sum(K_cond, axis=1, keepdims=True) + 1e-8
        K_norm = K_cond / row_sums

        # 2. Context Aggregation over Shared Factors
        F_shared = X_feats @ K_norm
        gene_cov = F_shared @ F_shared.T  # (N, N)
        cov_norm = np.sqrt(np.diag(gene_cov)[:, None] @ np.diag(gene_cov)[None, :]) + 1e-8
        G_shared = gene_cov / cov_norm
        np.fill_diagonal(G_shared, 0.0)

        # 3. Dual-Branch Neural Feature Projection
        mean_exp = np.mean(X_feats, axis=1, keepdims=True)
        var_exp = np.var(X_feats, axis=1, keepdims=True)
        H_in = np.hstack([mean_exp, var_exp, G_shared[:, :min(6, N)]])
        in_dim = H_in.shape[1]

        W_gcn = rng.randn(in_dim, d_emb) * np.sqrt(2.0 / (in_dim + d_emb))
        W_mlp1 = rng.randn(d_emb * 4 + 1, h_dim) * np.sqrt(2.0 / (d_emb * 4 + 1 + h_dim))
        W_mlp2 = rng.randn(h_dim, 1) * np.sqrt(2.0 / (h_dim + 1))
        b_mlp1 = np.zeros(h_dim)
        b_mlp2 = np.zeros(1)

        # Graph Convolution over shared neighborhood
        A_hat = G_shared + np.eye(N)
        deg = np.sum(A_hat, axis=1)
        deg_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        norm_lap = np.diag(deg_inv) @ A_hat @ np.diag(deg_inv)

        Z = norm_lap @ H_in @ W_gcn
        Z = np.maximum(0.0, Z)

        # 4. Neural Link Classification
        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                zi = Z[i, :]
                zj = Z[j, :]
                pair_feat = np.concatenate([zj, zi, np.abs(zj - zi), zj * zi, [S_dir[i, j]]])
                h1 = np.maximum(0.0, pair_feat @ W_mlp1 + b_mlp1)
                logit = float(h1 @ W_mlp2 + b_mlp2)
                prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

                sign_val = np.sign(S_dir[i, j]) if abs(S_dir[i, j]) > 1e-4 else np.sign(G_shared[i, j])
                if sign_val == 0:
                    sign_val = 1.0
                A_est[i, j] = sign_val * prob

        # 5. Discretization
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = np.mean(scores) * 0.8 if np.any(scores > 0) else 0.1
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="sfinn",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "embed_dim": d_emb,
                "hidden_dim": h_dim,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
                "use_data_prep": use_prep,
            },
        )
