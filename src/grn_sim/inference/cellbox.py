"""
CellBox: Continuous Neural ODE Relaxation with L1 Sparsity for Gene Network Inference.
Based on the neural ordinary differential equation formulation by Chris Sander's lab
(Cell Systems 2020, "Predicting cell dynamics through machine learning of interaction networks").
"""

import importlib.util
import os
import time
from typing import Optional

import numpy as np
from grn_sim.inference._parallel import report_progress

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult, regulator_input_fit

# PyTorch is optional and slow to import: only check that it is installed here, import it when a model is fitted.
HAS_TORCH = importlib.util.find_spec("torch") is not None

DEVICES = ("auto", "cpu", "cuda")
GPU_MIN_GENES = 300          # below this the CPU is as fast: CUDA's start-up (about 1 s per process) would dominate


def resolve_device(requested: Optional[str], n_genes: int) -> str:
    """
    ``"cpu"`` or ``"cuda"`` for a fit on ``n_genes`` genes. ``requested`` is ``"auto"`` (the default: the GPU when one is
    usable and the network has at least ``GPU_MIN_GENES`` genes), ``"cpu"`` or ``"cuda"`` (raises when no GPU is usable).
    ``None`` reads the ``GRN_SIM_DEVICE`` environment variable (default ``auto``), so a deployment can choose without code.
    """
    choice = (requested or os.environ.get("GRN_SIM_DEVICE") or "auto").lower().strip()
    if choice == "gpu":
        choice = "cuda"
    if choice not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, got {choice!r}")
    if choice == "cpu":
        return "cpu"
    import torch
    usable = torch.cuda.is_available()
    if choice == "cuda":
        if not usable:
            raise RuntimeError("device='cuda' was requested but PyTorch reports no usable GPU")
        return "cuda"
    return "cuda" if usable and n_genes >= GPU_MIN_GENES else "cpu"


PENALTY_REFERENCE_GENES = 100
LARGE_NETWORK_EPOCHS = 150


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
        device: Optional[str] = None,
        auto_scale: bool = True,
        size_scaled_penalty: bool = True,
        k_interactions: Optional[int] = None,
        regulator_direct: bool = True,
    ):
        super().__init__(name="cellbox")
        self.k_interactions = k_interactions
        # regulator-input data (X = cF(X + U)): fit X ~ phi(W (X + U)) / alpha directly instead of CellBox's additive u
        self.regulator_direct = regulator_direct
        self.device = device
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.lr = lr
        self.threshold = threshold
        self.envelope = envelope
        self.adaptive_fallback = adaptive_fallback
        self.perturbation_mode = perturbation_mode.lower().strip()
        # The tanh envelope assumes responses of order 1. Log fold changes of a screen are ~0.1: the network then sits in the
        # linear part of tanh and learns nothing useful (AUPR at the random level on 100-500 genes). Scaling X and U by
        # the same factor (1 / std of X) keeps the model (W is fitted to the scaled data) and fixes it.
        self.auto_scale = auto_scale
        # The fit error is a mean over entries but the L1/L2 penalties are sums over the N^2 weights, so the penalty's weight
        # grows with N^2: tuned near 100 genes, at 200 genes it shrinks every weight to ~0 (AUPR 0.41 -> 0.08 on simulated
        # screens). Above PENALTY_REFERENCE_GENES the penalties are scaled by (reference / N)^2 and the optimiser gets at least
        # LARGE_NETWORK_EPOCHS steps (AUPR at 200 genes back to 0.32-0.49); at or below the reference nothing changes.
        self.size_scaled_penalty = size_scaled_penalty

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        mode = kwargs.get("perturbation_mode", self.perturbation_mode).lower().strip()
        direct = mode == "regulator" and HAS_TORCH and bool(kwargs.get("regulator_direct", self.regulator_direct))
        if mode == "regulator" and not direct:
            return regulator_input_fit(self, X_mat, U_mat, **kwargs)
        N, M = X_mat.shape
        # direct regulator fit: the regulators act at their effective concentration Z = X + U, the model has no additive term
        Z_mat = X_mat + U_mat if direct else None
        if bool(kwargs.get("auto_scale", self.auto_scale)):
            sd = float(np.std(X_mat))
            if sd > 1e-12:
                X_mat, U_mat = X_mat / sd, U_mat / sd
                if direct:
                    Z_mat = Z_mat / sd

        l1 = float(kwargs.get("l1_reg", self.l1_reg))
        l2 = float(kwargs.get("l2_reg", self.l2_reg))
        thresh = float(kwargs.get("threshold", self.threshold))
        max_epochs = int(kwargs.get("epochs", self.epochs))
        if bool(kwargs.get("size_scaled_penalty", self.size_scaled_penalty)) and N > PENALTY_REFERENCE_GENES:
            factor = (PENALTY_REFERENCE_GENES / N) ** 2
            l1, l2 = l1 * factor, l2 * factor
            max_epochs = max(max_epochs, LARGE_NETWORK_EPOCHS)

        try:
            if not HAS_TORCH:
                raise ImportError("PyTorch is not installed")
            requested = (kwargs.get("device", self.device) or os.environ.get("GRN_SIM_DEVICE") or "auto").lower().strip()
            device = resolve_device(requested, N)
            fit_in, fit_u, fit_target = (Z_mat, np.zeros_like(X_mat), X_mat) if direct else (X_mat, U_mat, None)
            try:
                A_est, diagnostics = self._fit_torch(fit_in, fit_u, l1, l2, max_epochs, device, target=fit_target)
            except RuntimeError as exc:
                if device != "cuda" or requested in ("cuda", "gpu"):
                    raise                                                   # an explicit GPU request fails loudly
                A_est, diagnostics = self._fit_torch(fit_in, fit_u, l1, l2, max_epochs, "cpu", target=fit_target)
                diagnostics["gpu_fallback"] = str(exc)[:120]                # auto mode: the CPU finishes the job
        except ImportError:  # not installed, or installed but not importable: use the SciPy optimiser
            if N * N > 40000:
                raise RuntimeError(f"CellBox without PyTorch uses finite-difference gradients over N*N parameters: "
                                   f"{N} genes is too many (install torch, or use at most 200 genes)") from None
            A_est, diagnostics = self._fit_scipy(X_mat, U_mat, l1, l2)

        # Remove diagonal self-loops
        np.fill_diagonal(A_est, 0.0)

        # Discretize: sign of estimated coefficients exceeding threshold
        adj_pred = np.zeros((N, N), dtype=int)
        K = kwargs.get("k", self.k_interactions)
        if K is not None:
            # K strongest regulators per target (rows of A_est), as the other algorithms do when K is known; the
            # absolute threshold below left only about 40 edges for 600 true ones at 200 genes
            K = max(1, min(int(K), N - 1))
            for i in range(N):
                mags = np.abs(A_est[i, :])
                for reg in np.argsort(mags)[-K:]:
                    if mags[reg] > 0:
                        adj_pred[i, reg] = int(np.sign(A_est[i, reg]))
            mask = adj_pred != 0
        else:
            mask = np.abs(A_est) >= thresh
        if K is None and np.sum(mask) == 0 and self.adaptive_fallback and np.any(A_est != 0):
            # Adaptive fallback: select top 20% non-zero weights
            nonzero_mags = np.abs(A_est[A_est != 0])
            cutoff = float(np.percentile(nonzero_mags, 80))
            mask = np.abs(A_est) >= max(cutoff, 1e-4)
        if K is None:
            adj_pred[mask] = np.sign(A_est[mask]).astype(int)

        if direct:       # rows were targets; the regulator-input fits return T[regulator, target] (regulator_input_fit)
            A_est, adj_pred = A_est.T.copy(), adj_pred.T.copy()
            diagnostics["perturbation_mode"] = "regulator"
            diagnostics["regulator_direct"] = True

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
        device_name: str = "cpu",
        target: Optional[np.ndarray] = None,
    ) -> tuple:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        N, M = X_mat.shape
        device = torch.device(device_name)

        # Convert to tensors: shape (M, N) for batch of experiments
        X_t = torch.tensor(X_mat.T, dtype=torch.float32, device=device)
        U_t = torch.tensor(U_mat.T, dtype=torch.float32, device=device)
        # the response to fit: the steady state itself, or (regulator-input fit) X when X_mat holds the effective Z = X + U
        Y_t = X_t if target is None else torch.tensor(target.T, dtype=torch.float32, device=device)

        # Learnable interaction matrix W (N, N) and decay alpha (N,)
        # Initialize W with small values and alpha near 1.0
        # drawn on the CPU and moved, so a seeded run starts from the same weights on either device
        W = nn.Parameter((torch.randn(N, N) * 0.01).to(device))
        log_alpha = nn.Parameter(torch.zeros(N, device=device))

        optimizer = optim.Adam([W, log_alpha], lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        best_W = None

        for _ep in range(epochs):
            report_progress(_ep / epochs)
            optimizer.zero_grad()
            alpha = torch.exp(log_alpha) + 1e-4

            # Steady-state prediction: X_pred = (1 / alpha) * phi(X @ W.T + U)
            activation = torch.matmul(X_t, W.t()) + U_t
            if self.envelope == "tanh":
                phi_act = torch.tanh(activation)
            else:
                phi_act = torch.sigmoid(activation) - 0.5

            X_pred = phi_act / alpha

            mse_loss = torch.mean((X_pred - Y_t) ** 2)
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

            loss_value = loss.item()
            if loss_value < best_loss:
                best_loss = loss_value
                best_W = W.detach().clone()                                 # stays on the device until the end

        best = np.zeros((N, N)) if best_W is None else best_W.cpu().numpy()
        return best, {"final_loss": round(best_loss, 5), "backend": "pytorch", "device": device_name}

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
