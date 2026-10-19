"""
AtlasShift: Empirical Cross-Atlas Perturbation Transfer Model
Faithfully ported and adapted from Arc Institute Virtual Cell Challenge 2026 (VCC 2026):
https://github.com/kaipengm2/Virtual-Cell-Challenge-2026 (Public Leaderboard #82, Score: 0.15456)
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class AtlasShift(BaseInferenceAlgorithm):
    """
    AtlasShift Empirical Transfer Model (Virtual Cell Challenge 2026).

    In contrast to mechanistic ODE models (e.g. S2G/G2S), AtlasShift performs
    data-driven cross-context transfer by weighting empirical differential
    expression vectors across source atlases, applying promoter-neighbor priors,
    and generating single-cell count distributions via dual-moment fitting.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        amplitude: float = 1.0,
        clip: float = 4.0,
        space: str = "log2fc",
        de_threshold: float = 0.25,
        k_top_edges: Optional[int] = None,
    ):
        super().__init__(name="atlasshift")
        self.weights = weights
        self.amplitude = amplitude
        self.clip = clip
        self.space = space
        self.de_threshold = de_threshold
        self.k_top_edges = k_top_edges
        self.sources: List[Dict[str, Any]] = []

    def add_source(
        self,
        name: str,
        X: np.ndarray,
        U: np.ndarray,
        weight: float = 1.0,
        ctrl_probability: Optional[np.ndarray] = None,
    ) -> None:
        """Adds a source perturbation dataset (e.g. K562, HCT116, HEK293T, H1, CD4)."""
        self.sources.append({
            "name": name,
            "X": np.asarray(X, dtype=float),
            "U": np.asarray(U, dtype=float),
            "weight": float(weight),
            "ctrl": ctrl_probability,
        })

    def fuse_source_effects(
        self,
        X_sources: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Fuses multiple source perturbation response matrices using weighted averaging,
        mirroring `fuse_source_centered` in Virtual-Cell-Challenge-2026/model.py.
        """
        if not X_sources:
            raise ValueError("No source matrices provided for fusion.")
        if weights is None:
            weights = [1.0] * len(X_sources)
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= np.maximum(weights_arr.sum(), 1e-12)

        fused = np.zeros_like(X_sources[0], dtype=float)
        for X_s, w in zip(X_sources, weights_arr):
            fused += w * np.asarray(X_s, dtype=float)
        return fused

    def desired_mean(
        self,
        control_probability: np.ndarray,
        effects: np.ndarray,
        amplitude: Optional[float] = None,
        clip: Optional[float] = None,
        space: Optional[str] = None,
    ) -> np.ndarray:
        """
        Calculates post-perturbation target probability distribution,
        mirroring `desired_mean` in Virtual-Cell-Challenge-2026/model.py:
            desired = control_probability * 2^(amplitude * effects)
        """
        amp = amplitude if amplitude is not None else self.amplitude
        clp = clip if clip is not None else self.clip
        sp = space if space is not None else self.space

        p_ctrl = np.asarray(control_probability, dtype=float)
        eff = np.clip(amp * np.asarray(effects, dtype=float), -clp, clp)

        if sp == "log2fc":
            desired = p_ctrl * np.exp2(eff)
        elif sp == "bulk_delta":
            desired = np.expm1(np.maximum(np.log1p(50000.0 * p_ctrl) + eff, 0.0))
        else:
            # Linear relative ratio fallback (phpIoRN ratio convention: X_ij = (x_i - x_ctrl)/x_ctrl)
            desired = np.maximum(p_ctrl * (1.0 + eff), 1e-12)

        total = desired.sum(axis=-1, keepdims=True)
        return desired / np.maximum(total, 1e-12)

    def dual_moment_counts(
        self,
        template: np.ndarray,
        probability: np.ndarray,
        depths: np.ndarray,
        seed: int = 42,
        iterations: int = 40,
        tolerance: float = 0.001,
    ) -> np.ndarray:
        """
        Emits integer single-cell count matrices matching mean CPM and per-cell sequencing depths,
        mirroring `dual_moment_counts` in Virtual-Cell-Challenge-2026/model.py.
        """
        template = np.asarray(template, dtype=float).copy()
        p = np.asarray(probability, dtype=float)
        raw_depths = np.asarray(depths, dtype=int)
        n_cells, n_genes = template.shape

        if p.shape[0] != n_genes:
            raise ValueError(f"Probability dimension {p.shape[0]} != genes {n_genes}")

        # Normalize probability
        p = p / np.maximum(p.sum(), 1e-12)
        z = raw_depths / np.maximum(raw_depths.mean(), 1e-12)
        desired = n_cells * p

        # Normalize template rows
        row_sums = np.maximum(template.sum(axis=1, keepdims=True), 1e-30)
        x = template / row_sums
        x = 0.99 * x + 0.01 * p[None, :]

        # Iterative moment-matching tilt
        for step in range(iterations):
            col = np.maximum(x.sum(axis=0), 1e-30)
            x *= (desired / col)
            first = z @ x
            mean = np.divide(first, desired, out=np.ones_like(first), where=desired > 0)
            second = (z * z) @ x
            var = np.maximum(
                np.divide(second, desired, out=np.zeros_like(second), where=desired > 0) - mean * mean,
                0.0,
            )
            tilt = np.divide(1.0 - mean, var, out=np.zeros_like(mean), where=var > 1e-12)
            tilt = np.clip(tilt, -0.85, 0.85)
            x *= (1.0 + tilt[None, :] * (z[:, None] - mean[None, :]))
            x = np.maximum(x, 1e-30)
            x /= x.sum(axis=1, keepdims=True)

            if step % 5 == 4:
                cpm_err = float(np.abs(x.mean(axis=0) - p).sum())
                if cpm_err < tolerance:
                    break

        expected = x * raw_depths[:, None]
        integer = np.floor(expected).astype(np.int32)
        fractions = expected - integer
        rng = np.random.default_rng(seed)

        # Residual multinomial emission
        for i in range(n_cells):
            res = int(raw_depths[i]) - int(integer[i].sum())
            if res > 0:
                frac = fractions[i]
                csum = np.cumsum(frac)
                if csum[-1] > 0:
                    csum *= res / csum[-1]
                    locs = np.searchsorted(csum, np.arange(res) + rng.random(), side="right")
                    locs = np.clip(locs, 0, n_genes - 1)
                    np.add.at(integer[i], locs, 1)

        return integer

    def fit(
        self,
        X: np.ndarray,
        U: np.ndarray,
        **kwargs,
    ) -> InferenceResult:
        """
        Reverse-engineers interaction matrix from perturbation expression data
        using the empirical differential expression response matrix.

        In the system convention T[j, k] (regulator j in row, target k in col),
        perturbing gene j produces response vector X[:, j].
        Thus, A_est[j, i] corresponds to the empirical shift in gene i upon perturbing j.
        """
        start_time = time.time()
        X_arr = np.real(X).astype(float)
        U_arr = np.real(U).astype(float)
        n, m = X_arr.shape

        # If external sources are registered, fuse them with current X
        if self.sources:
            all_X = [s["X"] for s in self.sources] + [X_arr]
            all_W = [s["weight"] for s in self.sources] + [2.0]
            eff = self.fuse_source_effects(all_X, all_W)
        else:
            eff = X_arr.copy()

        # Invert or scale into Jacobian coefficient estimate:
        # In linear GRN A * X = -U => X approx -A^(-1) * U.
        # AtlasShift empirical transfer estimates direct sensitivity S = X * U^(-1) or normalized response
        # To populate A_est: A_est = -U * pinv(X) or thresholded differential expression:
        # Here we construct continuous interaction strength proportional to effect:
        A_est = np.zeros((n, n), dtype=float)
        for j in range(min(n, m)):
            # Regulator j was perturbed: response of all genes i is in column j
            # Target i is affected by regulator j
            u_j = U_arr[j, j] if j < U_arr.shape[0] and j < U_arr.shape[1] and U_arr[j, j] != 0 else 1.0
            A_est[j, :] = eff[:, j] / u_j

        # Do not allow artificial self-loops from direct perturbation
        np.fill_diagonal(A_est, 0.0)

        # Discretize into signed adjacency matrix (-1, 0, +1)
        adjacency_pred = np.zeros((n, n), dtype=int)
        if self.k_top_edges is not None and self.k_top_edges > 0:
            for i in range(n):
                col_vals = np.abs(A_est[:, i])
                top_idx = np.argsort(col_vals)[::-1][:self.k_top_edges]
                for j in top_idx:
                    if col_vals[j] > 1e-5:
                        adjacency_pred[j, i] = int(np.sign(A_est[j, i]))
        else:
            adjacency_pred = np.where(
                A_est > self.de_threshold, 1,
                np.where(A_est < -self.de_threshold, -1, 0)
            ).astype(int)

        np.fill_diagonal(adjacency_pred, 0)
        time_elapsed = time.time() - start_time

        return InferenceResult(
            algorithm_name="atlasshift",
            A_est=A_est,
            adjacency_pred=adjacency_pred,
            time_elapsed=time_elapsed,
            diagnostics={
                "de_threshold": self.de_threshold,
                "sources_fused": len(self.sources) + 1,
                "space": self.space,
                "amplitude": self.amplitude,
            },
        )
