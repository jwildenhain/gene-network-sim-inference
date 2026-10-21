"""
CPA-Lite: Compositional Perturbation Autoencoder with Latent Space Vector Arithmetic.
Based on the compositional perturbation architecture by Fabian Theis's lab
(Mol Syst Biol 2023, "Predicting cellular responses to complex perturbations").
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CPALite(BaseInferenceAlgorithm):
    """
    Learns a low-dimensional latent space bottleneck where perturbation effects
    are represented as additive latent vectors:
        z = Encoder(x)
        x_pred = Decoder(z_ctrl + Delta z_pert)
    Infers gene-gene interaction influence via the reconstructed sensitivity Jacobian.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        epochs: int = 120,
        lr: float = 0.01,
        threshold: float = 0.05,
        k_interactions: Optional[int] = None,
    ):
        super().__init__(name="cpa_lite")
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.k_interactions = k_interactions

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        d = min(self.latent_dim, max(2, N // 2))
        thresh = float(kwargs.get("threshold", self.threshold))
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            K = max(1, min(int(K), N - 1))

        if HAS_TORCH:
            A_est, diagnostics = self._fit_torch(X_mat, U_mat, d)
        else:
            A_est, diagnostics = self._fit_linear_pca(X_mat, U_mat, d)

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
            algorithm_name="cpa_lite",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=elapsed,
            diagnostics=diagnostics,
        )

    def _fit_torch(self, X_mat: np.ndarray, U_mat: np.ndarray, d: int) -> tuple:
        N, M = X_mat.shape
        device = torch.device("cpu")

        # Linear Autoencoder with perturbation injection in latent space
        X_t = torch.tensor(X_mat.T, dtype=torch.float32, device=device)  # (M, N)
        U_t = torch.tensor(U_mat.T, dtype=torch.float32, device=device)  # (M, N)

        # Encoder: N -> d, Decoder: d -> N, Perturbation projector: N -> d
        encoder = nn.Linear(N, d, bias=False)
        decoder = nn.Linear(d, N, bias=False)
        pert_proj = nn.Linear(N, d, bias=False)

        params = list(encoder.parameters()) + list(decoder.parameters()) + list(pert_proj.parameters())
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=1e-4)

        for ep in range(self.epochs):
            optimizer.zero_grad()
            z_basal = encoder(X_t)
            delta_z = pert_proj(U_t)
            z_pert = z_basal + delta_z
            X_recon = decoder(z_pert)

            loss = torch.mean((X_recon - X_t) ** 2)
            loss.backward()
            optimizer.step()

        # Effective network sensitivity Jacobian: W = Decoder.weight @ Encoder.weight
        with torch.no_grad():
            W = torch.matmul(decoder.weight, encoder.weight).cpu().numpy()

        np.fill_diagonal(W, 0.0)
        return W, {"latent_dim": d, "final_loss": round(float(loss.item()), 5), "backend": "pytorch"}

    def _fit_linear_pca(self, X_mat: np.ndarray, U_mat: np.ndarray, d: int) -> tuple:
        # SVD-based linear latent representation
        U_svd, S, Vt = np.linalg.svd(X_mat, full_matrices=False)
        Ud = U_svd[:, :d]  # Decoder basis (N, d)
        Vd = Ud.T          # Encoder projection (d, N)

        # Sensitivity matrix W = Ud @ Vd - I
        W = Ud @ Vd - np.eye(X_mat.shape[0])
        np.fill_diagonal(W, 0.0)
        return W, {"latent_dim": d, "backend": "svd_pca"}
