"""
DeepRIG: Graph Autoencoder-based Model for Regulatory Networks Inference among Genes.
Based on PLOS Computational Biology 2024 ("DeepRIG: A deep learning-based model for regulatory
networks inference among genes using graph autoencoders").
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class DeepRIGAlgorithm(BaseInferenceAlgorithm):
    """
    DeepRIG: Graph Autoencoder (GAE) with Graph Convolutional Network (GCN) encoder.

    1. Constructs a prior co-expression graph A_prior from normalized expression profiles.
    2. GCN Encoder: Multi-layer graph convolution propagates global topological features:
       H^(1) = ReLU(D^(-1/2) * (A_prior + I) * D^(-1/2) * X * W_1)
       Z = D^(-1/2) * (A_prior + I) * D^(-1/2) * H^(1) * W_2
    3. Link Decoder: Inner-product scoring function S_hat_ij = sigmoid(z_i^T * z_j).
    4. Directed Causal Propagation: Combines latent link likelihood with asymmetric
       perturbation sensitivity to infer directed, signed regulatory topology.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        prior_threshold: float = 0.15,
        k_interactions: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(name="deeprig")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.prior_threshold = prior_threshold
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

        # 1. Prior Co-expression Graph Construction
        corr = np.corrcoef(X_mat)
        np.nan_to_num(corr, copy=False, nan=0.0)
        np.fill_diagonal(corr, 0.0)

        prior_thresh = float(kwargs.get("prior_threshold", self.prior_threshold))
        A_prior = np.where(np.abs(corr) >= prior_thresh, np.abs(corr), 0.0)

        # 2. Symmetric Normalized Laplacian: D^(-1/2) * (A + I) * D^(-1/2)
        A_hat = A_prior + np.eye(N)
        deg = np.sum(A_hat, axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
        norm_laplacian = np.diag(deg_inv_sqrt) @ A_hat @ np.diag(deg_inv_sqrt)

        # 3. GCN Encoder Layer 1 & Layer 2
        # Input features H0: normalized expression profiles (N, M)
        feat_mean = np.mean(X_mat, axis=1, keepdims=True)
        feat_std = np.std(X_mat, axis=1, keepdims=True) + 1e-6
        H0 = (X_mat - feat_mean) / feat_std

        # Layer 1 weights (M -> hidden_dim)
        h_dim = min(self.hidden_dim, max(8, M))
        l_dim = min(self.latent_dim, h_dim)
        W1 = rng.randn(M, h_dim) * np.sqrt(2.0 / (M + h_dim))
        W2 = rng.randn(h_dim, l_dim) * np.sqrt(2.0 / (h_dim + l_dim))

        # Forward pass: GCN
        H1 = norm_laplacian @ H0 @ W1
        H1 = np.maximum(0.0, H1)  # ReLU
        Z = norm_laplacian @ H1 @ W2  # Latent representation (N, l_dim)

        # Normalize latent embeddings
        z_norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Z_unit = Z / z_norm

        # 4. Decoder: Inner product link prediction matrix
        # Score_ij = sigmoid(z_i @ z_j^T)
        logits = Z_unit @ Z_unit.T
        link_scores = 1.0 / (1.0 + np.exp(-np.clip(logits * 3.0, -10.0, 10.0)))
        np.fill_diagonal(link_scores, 0.0)

        # 5. Direct Causal Asymmetry from Perturbations
        # Sensitivity S = -X @ pinv(U)
        try:
            pinv_U = np.linalg.pinv(U_mat) if np.any(U_mat) else np.eye(M)
            S_dir = -X_mat @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -X_mat @ np.eye(N)
        except Exception:
            S_dir = -X_mat[:, :N] if M >= N else np.zeros((N, N))

        A_est = np.zeros((N, N), dtype=float)
        adj_pred = np.zeros((N, N), dtype=int)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Regulator j -> Target i
                prob = link_scores[i, j]
                sign_val = np.sign(corr[i, j]) if abs(corr[i, j]) > 1e-4 else 1.0
                asym_weight = 1.0 + min(2.0, abs(S_dir[i, j]))
                A_est[i, j] = sign_val * prob * asym_weight

        # 6. Discretization
        for i in range(N):
            scores = np.abs(A_est[i, :])
            if K is not None:
                top_idx = np.argsort(scores)[::-1][:K]
                for j in top_idx:
                    if scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))
            else:
                row_thresh = np.mean(scores) * 0.9 if np.any(scores > 0) else 0.5
                for j in range(N):
                    if scores[j] > row_thresh and scores[j] > 1e-4:
                        adj_pred[i, j] = int(np.sign(A_est[i, j]))

        time_elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="deeprig",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "latent_dim": l_dim,
                "hidden_dim": h_dim,
                "total_reconstructed_edges": int(np.sum(adj_pred != 0)),
            },
        )
