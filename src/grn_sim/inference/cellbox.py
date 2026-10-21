"""
CellBox: Continuous Neural ODE Relaxation with L1 Sparsity for Gene Network Inference.
Based on the neural ordinary differential equation formulation by Chris Sander's lab
(Cell Systems 2020, "Predicting cell dynamics through machine learning of interaction networks").
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult, regulator_input_fit

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CellBoxNeuralODE(BaseInferenceAlgorithm):
    """
    Infers regulatory network interaction matrix W by fitting a continuous
    non-linear neural ODE relaxation to steady-state perturbation data:
        dx_i / dt = phi( sum_j W_ij * x_j + u_i ) - alpha_i * x_i
    At steady state:
        x_i = (1 / alpha_i) * phi( sum_j W_ij * x_j + u_i )
    Trained with L1 Lasso sparsity penalty on W and L2 regularization.
    """

    def __init__(
        self,
        l1_reg: float = 0.003,
        l2_reg: float = 0.001,
        epochs: int = 150,
        lr: float = 0.05,
        threshold: float = 0.02,
        envelope: str = "tanh",
        adaptive_fallback: bool = True,
        perturbation_mode: str = "additive",
    ):
        super().__init__(name="cellbox")
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.envelope = envelope
        self.adaptive_fallback = adaptive_fallback
        self.perturbation_mode = perturbation_mode.lower().strip()

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        if kwargs.get("perturbation_mode", self.perturbation_mode).lower().strip() == "regulator":
            return regulator_input_fit(self, X_mat, U_mat, **kwargs)
        N, M = X_mat.shape

        l1 = float(kwargs.get("l1_reg", self.l1_reg))
        l2 = float(kwargs.get("l2_reg", self.l2_reg))
        thresh = float(kwargs.get("threshold", self.threshold))
        max_epochs = int(kwargs.get("epochs", self.epochs))

        if HAS_TORCH:
            A_est, diagnostics = self._fit_torch(X_mat, U_mat, l1, l2, max_epochs)
        else:
            A_est, diagnostics = self._fit_scipy(X_mat, U_mat, l1, l2)

        # Remove diagonal self-loops
        np.fill_diagonal(A_est, 0.0)

        # Discretize: sign of estimated coefficients exceeding threshold
        adj_pred = np.zeros((N, N), dtype=int)
        mask = np.abs(A_est) >= thresh
        if np.sum(mask) == 0 and self.adaptive_fallback and np.any(A_est != 0):
            # Adaptive fallback: select top 20% non-zero weights
            nonzero_mags = np.abs(A_est[A_est != 0])
            cutoff = float(np.percentile(nonzero_mags, 80))
            mask = np.abs(A_est) >= max(cutoff, 1e-4)
        adj_pred[mask] = np.sign(A_est[mask]).astype(int)

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name="cellbox",
            A_est=A_est,
            adjacency_pred=adj_pred,
            time_elapsed=elapsed,
            diagnostics=diagnostics,
        )

    def _fit_torch(
        self,
        X_mat: np.ndarray,
        U_mat: np.ndarray,
        l1: float,
        l2: float,
        epochs: int,
    ) -> tuple:
        N, M = X_mat.shape
        device = torch.device("cpu")

        # Convert to tensors: shape (M, N) for batch of experiments
        X_t = torch.tensor(X_mat.T, dtype=torch.float32, device=device)
        U_t = torch.tensor(U_mat.T, dtype=torch.float32, device=device)

        # Learnable interaction matrix W (N, N) and decay alpha (N,)
        # Initialize W with small values and alpha near 1.0
        W = nn.Parameter(torch.randn(N, N, device=device) * 0.01)
        log_alpha = nn.Parameter(torch.zeros(N, device=device))

        optimizer = optim.Adam([W, log_alpha], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        best_W = np.zeros((N, N))

        for ep in range(epochs):
            optimizer.zero_grad()
            alpha = torch.exp(log_alpha) + 1e-4

            # Steady-state prediction: X_pred = (1 / alpha) * phi(X @ W.T + U)
            activation = torch.matmul(X_t, W.t()) + U_t
            if self.envelope == "tanh":
                phi_act = torch.tanh(activation)
            else:
                phi_act = torch.sigmoid(activation) - 0.5

            X_pred = phi_act / alpha

            mse_loss = torch.mean((X_pred - X_t) ** 2)
            # L1 penalty on off-diagonal W
            diag_mask = torch.eye(N, device=device).bool()
            off_diag_W = W.masked_select(~diag_mask)
            l1_loss = l1 * torch.sum(torch.abs(off_diag_W))
            l2_loss = l2 * torch.sum(W ** 2)

            loss = mse_loss + l1_loss + l2_loss
            loss.backward()

            # Zero out diagonal gradients
            with torch.no_grad():
                W.grad.fill_diagonal_(0.0)

            optimizer.step()
            scheduler.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_W = W.detach().cpu().numpy().copy()

        return best_W, {"final_loss": round(best_loss, 5), "backend": "pytorch"}

    def _fit_scipy(self, X_mat: np.ndarray, U_mat: np.ndarray, l1: float, l2: float) -> tuple:
        from scipy.optimize import minimize
        N, M = X_mat.shape

        def loss_fn(params):
            W = params[:N*N].reshape(N, N)
            np.fill_diagonal(W, 0.0)
            alpha = np.exp(params[N*N:]) + 1e-4

            activation = X_mat.T @ W.T + U_mat.T
            phi_act = np.tanh(activation)
            X_pred = phi_act / alpha

            mse = np.mean((X_pred - X_mat.T) ** 2)
            l1_term = l1 * np.sum(np.abs(W))
            l2_term = l2 * np.sum(W ** 2)
            return mse + l1_term + l2_term

        init_params = np.zeros(N * N + N)
        res = minimize(loss_fn, init_params, method="L-BFGS-B", options={"maxiter": 80})
        best_W = res.x[:N*N].reshape(N, N)
        np.fill_diagonal(best_W, 0.0)
        return best_W, {"final_loss": round(float(res.fun), 5), "backend": "scipy"}
