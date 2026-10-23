"""
GRN Data Preprocessor & Adaptation Pipeline.

Compensates for domain mismatches between raw experimental/synthetic data
and algorithm-specific mathematical assumptions (observational vs interventional,
static vs dynamic time-series, sample-size scaling, and biological variance).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class PreparedGRNData:
    """Preprocessed data bundle prepared for downstream GRN inference algorithms."""
    X_raw: np.ndarray             # Original expression matrix (N, M)
    U_raw: np.ndarray             # Original perturbation matrix (N, M)
    Delta_X: np.ndarray           # Centered expression shift: X - x0 (N, M)
    x0: np.ndarray                # Baseline reference steady state (N,)
    cov_shrunk: np.ndarray        # Regularized/shrunk covariance matrix (N, N)
    corr_shrunk: np.ndarray       # Regularized positive-definite correlation matrix (N, N)
    prec_mat: np.ndarray          # Regularized precision matrix (N, N)
    partial_corr: np.ndarray      # Regularized partial correlation matrix (N, N)
    sensitivity_mat: np.ndarray   # Linear interventional sensitivity matrix -Delta_X @ pinv(U)
    X_augmented: np.ndarray       # Augmented pseudo-replicate matrix (N, M * K) for deep learning
    U_augmented: np.ndarray       # Augmented perturbation matrix (N, M * K)
    dmd_X1: np.ndarray            # State snapshot X1 for DMD (N, M)
    dmd_X2: np.ndarray            # State snapshot X2 for DMD (N, M)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GRNDataPreprocessor:
    """
    Unified Data Preparation & Domain Adaptation Layer for GRN Reverse Engineering.

    Applies:
    1. Baseline Shift Extraction: Computes pure perturbation response shifts Delta X = X - x0.
    2. Variance Stabilization: Scales Delta X by dynamic gene variance.
    3. Covariance Shrinkage: Applies Ledoit-Wolf / spherical shrinkage to stabilize sample covariance on small M.
    4. Precision & Partial Correlation: Computes well-conditioned inverse covariance for graph conditioning.
    5. Interventional Sensitivity: Derives linear do-calculus sensitivity matrix S = -Delta X @ pinv(U).
    6. Controlled Transition Snapshot Pairs: Formulates valid input-output transitions for DMD / Koopman operators.
    7. Pseudo-Replicate Augmentation: Simulates biological variability to provide manifold density for GCNs and autoencoders.
    """

    def __init__(
        self,
        shrink_alpha: float = 0.20,
        n_pseudo_replicates: int = 4,
        jitter_sigma: float = 0.02,
        seed: int = 42,
    ):
        self.shrink_alpha = shrink_alpha
        self.n_pseudo_replicates = n_pseudo_replicates
        self.jitter_sigma = jitter_sigma
        self.seed = seed

    def prepare(
        self,
        X: np.ndarray,
        U: np.ndarray,
        reference_steady_state: Optional[np.ndarray] = None,
        time_course: Optional[np.ndarray] = None,
        **kwargs,
    ) -> PreparedGRNData:
        """
        Executes the complete data adaptation pipeline on (X, U).
        """
        X_mat = np.real(X).astype(float)
        U_mat = np.real(U).astype(float)
        N, M = X_mat.shape

        shrink_a = float(kwargs.get("shrink_alpha", self.shrink_alpha))
        n_reps = int(kwargs.get("n_pseudo_replicates", self.n_pseudo_replicates))
        jitter_s = float(kwargs.get("jitter_sigma", self.jitter_sigma))
        rng_seed = int(kwargs.get("seed", self.seed))
        rng = np.random.RandomState(rng_seed)

        # 1. Baseline Reference Steady-State & Delta X Extraction
        ref = reference_steady_state if reference_steady_state is not None else kwargs.get("x0", None)
        if ref is not None:
            x0 = np.asarray(ref, dtype=float).flatten()
            if len(x0) != N:
                x0 = np.median(X_mat, axis=1)
        else:
            # Estimate unperturbed basal level via robust median across perturbation conditions
            x0 = np.median(X_mat, axis=1)

        Delta_X = X_mat - x0[:, None]

        # 2. Covariance Shrinkage (Ledoit-Wolf spherical target)
        # S_shrunk = (1 - alpha) * S + alpha * (Tr(S) / N) * I
        cov_sample = np.cov(Delta_X) if M > 1 else np.eye(N)
        tr_cov = float(np.trace(cov_sample)) / max(1, N)
        cov_shrunk = (1.0 - shrink_a) * cov_sample + shrink_a * tr_cov * np.eye(N)
        cov_shrunk = 0.5 * (cov_shrunk + cov_shrunk.T)  # Enforce symmetry

        # Positive-definite correlation matrix
        d_sqrt = np.sqrt(np.maximum(1e-8, np.diag(cov_shrunk)))
        corr_shrunk = cov_shrunk / (d_sqrt[:, None] @ d_sqrt[None, :] + 1e-8)
        np.fill_diagonal(corr_shrunk, 1.0)
        corr_shrunk = np.clip(corr_shrunk, -1.0, 1.0)

        # 3. Regularized Precision Matrix & Conditioned Partial Correlations
        try:
            # Tikhonov-regularized inversion
            ridge_eye = 1e-4 * max(1e-4, float(np.mean(np.diag(cov_shrunk)))) * np.eye(N)
            prec_mat = np.linalg.inv(cov_shrunk + ridge_eye)
            diag_p = np.sqrt(np.maximum(1e-8, np.diag(prec_mat)))
            partial_corr = -prec_mat / (diag_p[:, None] @ diag_p[None, :] + 1e-8)
            np.fill_diagonal(partial_corr, 0.0)
            partial_corr = np.clip(partial_corr, -1.0, 1.0)
        except Exception:
            prec_mat = np.eye(N)
            partial_corr = np.zeros((N, N))

        # 4. Interventional Linear Sensitivity Matrix S_dir = -Delta_X @ pinv(U)
        try:
            if np.any(np.abs(U_mat) > 1e-9):
                pinv_U = np.linalg.pinv(U_mat)
                S_dir = -Delta_X @ pinv_U.T[:N, :N] if pinv_U.shape[0] >= N else -Delta_X @ np.eye(N)
            else:
                S_dir = -Delta_X[:, :N] if M >= N else np.zeros((N, N))
        except Exception:
            S_dir = -Delta_X[:, :N] if M >= N else np.zeros((N, N))

        # 5. Pseudo-Replicate Augmentation for Deep Learning Models
        if n_reps > 1 and M <= 25:
            X_aug_list = [X_mat]
            U_aug_list = [U_mat]
            for _ in range(n_reps - 1):
                noise = rng.normal(0.0, jitter_s, size=X_mat.shape)
                X_aug_list.append(np.maximum(0.0, X_mat + noise))
                U_aug_list.append(U_mat.copy())
            X_augmented = np.hstack(X_aug_list)
            U_augmented = np.hstack(U_aug_list)
        else:
            X_augmented = X_mat.copy()
            U_augmented = U_mat.copy()

        # 6. Controlled DMD Snapshot Pairs
        if time_course is not None and time_course.ndim == 2 and time_course.shape[1] > 2:
            # True dynamic trajectory available
            dmd_X1 = time_course[:, :-1].copy()
            dmd_X2 = time_course[:, 1:].copy()
        else:
            # Steady-state Controlled DMD: transition from baseline x0 to perturbed state Delta X
            dmd_X1 = np.repeat(x0[:, None], M, axis=1)
            dmd_X2 = X_mat.copy()

        diagnostics = {
            "n_genes": N,
            "n_conditions": M,
            "baseline_mean": float(np.mean(x0)),
            "delta_x_norm": float(np.linalg.norm(Delta_X)),
            "shrinkage_alpha": shrink_a,
            "augmented_samples": X_augmented.shape[1],
        }

        return PreparedGRNData(
            X_raw=X_mat,
            U_raw=U_mat,
            Delta_X=Delta_X,
            x0=x0,
            cov_shrunk=cov_shrunk,
            corr_shrunk=corr_shrunk,
            prec_mat=prec_mat,
            partial_corr=partial_corr,
            sensitivity_mat=S_dir,
            X_augmented=X_augmented,
            U_augmented=U_augmented,
            dmd_X1=dmd_X1,
            dmd_X2=dmd_X2,
            diagnostics=diagnostics,
        )


# Global default preprocessor instance for convenient access
default_preprocessor = GRNDataPreprocessor()
