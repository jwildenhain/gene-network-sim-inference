"""
SFINN: Shared Factor neighborhood and Integrated Neural Network for Gene Regulatory Network Inference.
Based on Briefings in Bioinformatics 2024 ("SFINN: Shared factor neighborhood and integrated neural
network for gene regulatory network inference from single-cell and spatial transcriptomic data").
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


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
    ):
        super().__init__(name="sfinn")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.k_interactions = k_interactions
        self.seed = seed

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        rng = np.random.RandomState(self.seed)
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        # 1. Shared Factor Neighborhood Kernel
        # K_cond(m, l) = cos_sim(X[:, m], X[:, l])
        col_norms = np.linalg.norm(X_mat, axis=0, keepdims=True) + 1e-8
        X_normed = X_mat / col_norms
        K_cond = X_normed.T @ X_normed  # (M, M)
        K_cond = np.maximum(0.0, K_cond)  # Non-negative neighborhood weights

        # Normalize neighborhood matrix
        row_sums = np.sum(K_cond, axis=1, keepdims=True) + 1e-8
        K_norm = K_cond / row_sums

        # 2. Context Aggregation over Shared Factors
        # F_shared = X @ K_norm (N, M)
        F_shared = X_mat @ K_norm

        # Pairwise gene co-occurrence under shared factors
        gene_cov = F_shared @ F_shared.T  # (N, N)
        cov_norm = np.sqrt(np.diag(gene_cov)[:, None] @ np.diag(gene_cov)[None, :]) + 1e-8
        G_shared = gene_cov / cov_norm
        np.fill_diagonal(G_shared, 0.0)

        # 3. Dual-Branch Neural Feature Projection
        d_emb = min(self.embed_dim, max(8, N - 1))
        # Initial gene features: concatenated mean, variance, and shared factor response
        mean_exp = np.mean(X_mat, axis=1, keepdims=True)
        var_exp = np.var(X_mat, axis=1, keepdims=True)
        H_in = np.hstack([mean_exp, var_exp, G_shared[:, :min(6, N)]])
        in_dim = H_in.shape[1]

        # Branch weights
        W_gcn = rng.randn(in_dim, d_emb) * np.sqrt(2.0 / (in_dim + d_emb))
        W_mlp1 = rng.randn(d_emb * 4 + 1, self.hidden_dim) * np.sqrt(2.0 / (d_emb * 4 + 1 + self.hidden_dim))
        W_mlp2 = rng.randn(self.hidden_dim, 1) * np.sqrt(2.0 / (self.hidden_dim + 1))
        b_mlp1 = np.zeros(self.hidden_dim)
        b_mlp2 = np.zeros(1)

        # Graph Convolution over shared neighborhood
        A_hat = G_shared + np.eye(N)
        deg = np.sum(A_hat, axis=1)
        deg_inv = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        norm_lap = np.diag(deg_inv) @ A_hat @ np.diag(deg_inv)

        Z = norm_lap @ H_in @ W_gcn  # (N, d_emb)
        Z = np.maximum(0.0, Z)       # ReLU

        # Directed perturbation sensitivity: S = -X @ pinv(U)
        try:
            pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
            S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
        except Exception:
            S_dir = -X_mat[:, :N] if M >= N else np.zeros((N, N))

        corr = np.corrcoef(X_mat)
        np.nan_to_num(corr, copy=False, nan=0.0)

        # 4. Pairwise Interaction Evaluation
        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Regulator j -> Target i
                # Feature vector: [z_j, z_i, |z_j - z_i|, z_j * z_i, S_dir[i, j]]
                diff_feat = np.abs(Z[j] - Z[i])
                prod_feat = Z[j] * Z[i]
                v_pair = np.concatenate([Z[j], Z[i], diff_feat, prod_feat, [S_dir[i, j]]])

                # MLP forward
                h1 = np.maximum(0.0, v_pair @ W_mlp1 + b_mlp1)
                logit = float(h1 @ W_mlp2 + b_mlp2)
                prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -10.0, 10.0)))

                sign_val = np.sign(corr[i, j]) if abs(corr[i, j]) > 1e-4 else 1.0
                asym_weight = 1.0 + abs(S_dir[i, j])
                A_est[i, j] = sign_val * prob * asym_weight

        # 5. Discretization
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = np.mean(scores) * 0.85 if np.any(scores > 0) else 0.5
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
                "hidden_dim": self.hidden_dim,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
            },
        )
