"""
Lightweight cross-context transfer predictors for the Virtual Cell Challenge benchmark.

The task (the 2026 challenge in miniature): predict the response ``X[:, j]`` to perturbing gene ``j`` in a **target
context** from the responses to the same perturbation in several **source contexts** (the atlases), using only the
perturbations that were measured in the target (the *seen* columns) to calibrate. The perturbation design (which gene is
hit, how hard) and the target's unperturbed state are known for every column; the target responses of the held-out
columns are never read. For a seen column the prediction is made leave-one-out, so it never sees its own target column.

All predictors are small, closed-form and dependency-free (NumPy only). Their three hyper-parameters (calibration ridge 1,
gate noise scale 0.5, regression ridge 1) were chosen on tuning networks (seeds 101-112) that the reported study
(scripts/transfer_study.py, seeds 1-24) does not use. They are written from the *published
descriptions* of what worked in the Virtual Cell Challenge, simplified to what a simulator without single cells can
express; they are not ports of any team's code:

* ``atlas_mean``        reference: the plain average of the source responses (zero-shot transfer, no calibration).
* ``atlas_calibrated``  AtlasShift-style: sources weighted by how well each explains the seen target columns, then a
                        per-gene scale fitted on those columns (calibration to the target context), shrunk towards 1.
* ``transpert_style``   TransPert-style (summary statistics + global scaling): the cross-context consensus (median)
                        keeps only the entries all sources agree on, above the consensus noise floor (a differential-
                        expression-like gate), and one global scale is fitted to the seen target columns.
* ``pseudobulk_ridge``  pseudo-bulk regression with engineered features: a pooled ridge regression of what the source
                        average gets wrong, on the average itself, the on-target (self) effect and its design strength,
                        and the average interacted with the gene's basal expression.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

EPS = 1e-12
MAX_SCALE = 3.0  # calibration may stretch a source response at most 3-fold (and never flips its sign)


# ------------------------------------------------------------------------------------------------ the data
@dataclass(frozen=True)
class TransferData:
    """One transfer problem. ``target`` columns outside ``seen`` are never read."""

    sources: Tuple[np.ndarray, ...]   # k source-context response matrices, each (genes, perturbations)
    target: np.ndarray                # target-context responses; only the ``seen`` columns are observed
    design: np.ndarray                # target perturbation design U (genes, perturbations): known for every column
    control: np.ndarray               # target unperturbed expression (genes,)
    seen: Tuple[int, ...]             # perturbations measured in the target context

    def __post_init__(self) -> None:
        n = self.target.shape[0]
        if not self.sources:
            raise ValueError("at least one source context is needed")
        for s in self.sources:
            if s.shape != self.target.shape:
                raise ValueError(f"source shape {s.shape} differs from the target shape {self.target.shape}")
        if self.design.shape != self.target.shape or self.control.shape != (n,):
            raise ValueError("design must match the target and control must have one value per gene")
        if len(set(self.seen)) != len(self.seen) or not all(0 <= j < self.target.shape[1] for j in self.seen):
            raise ValueError("seen must list distinct column indices of the target")


class TransferModel(ABC):
    """A fitted predictor; it can be applied to another context's sources, design and control."""

    @abstractmethod
    def predict(
        self, sources: Sequence[np.ndarray], design: np.ndarray, control: np.ndarray, columns: Sequence[int]
    ) -> np.ndarray:
        """Predicted responses (genes, len(columns))."""


@dataclass(frozen=True)
class TransferPredictor:
    key: str
    label: str
    kind: str          # "reference" or "method"
    description: str
    fit: Callable[[Sequence[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Sequence[int]], TransferModel]
    # fit(sources, target, design, control, train_columns) -> TransferModel; only ``train_columns`` of target are read


# ------------------------------------------------------------------------------------------------ predictors
def _stack(sources: Sequence[np.ndarray], columns: Sequence[int]) -> np.ndarray:
    return np.stack([np.asarray(s, dtype=float)[:, list(columns)] for s in sources])  # (k, genes, columns)


class _AtlasMean(TransferModel):
    def predict(self, sources, design, control, columns):
        return _stack(sources, columns).mean(axis=0)


def _fit_atlas_mean(sources, target, design, control, train) -> TransferModel:
    return _AtlasMean()


class _AtlasCalibrated(TransferModel):
    def __init__(self, weights: np.ndarray, scale: np.ndarray):
        self.weights, self.scale = weights, scale

    def predict(self, sources, design, control, columns):
        blended = np.tensordot(self.weights, _stack(sources, columns), axes=1)
        return self.scale[:, None] * blended


def _fit_atlas_calibrated(sources, target, design, control, train, ridge: float = 1.0) -> TransferModel:
    observed = np.asarray(target, dtype=float)[:, list(train)]
    stacked = _stack(sources, train)
    errors = np.array([float(np.mean((s - observed) ** 2)) for s in stacked])
    weights = 1.0 / (errors + 1e-6 * (float(errors.mean()) + EPS))       # a source that explains the target gets more say
    weights /= weights.sum()
    blended = np.tensordot(weights, stacked, axes=1)
    num = np.sum(blended * observed, axis=1)
    den = np.sum(blended ** 2, axis=1)
    # Ridge towards a scale of 1: a gene whose seen columns say little keeps the plain (weighted) source response.
    shrink = ridge * float(den.mean()) + EPS
    scale = np.clip((num + shrink) / (den + shrink), 0.0, MAX_SCALE)
    return _AtlasCalibrated(weights, scale)


def _consensus_gate(sources: Sequence[np.ndarray], noise_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Median response over the sources and the 0/1 mask of entries every source agrees on and that clear the noise floor."""
    stacked = _stack(sources, range(np.asarray(sources[0]).shape[1]))
    consensus = np.median(stacked, axis=0)
    off_diagonal = ~np.eye(consensus.shape[0], consensus.shape[1], dtype=bool)
    floor = 1.4826 * float(np.median(np.abs(consensus[off_diagonal]))) if off_diagonal.any() else 0.0
    signs = np.sign(stacked)
    agree = np.all(signs == signs[0], axis=0) & (signs[0] != 0)
    return consensus, (agree & (np.abs(consensus) >= noise_scale * floor)).astype(float)


class _TranspertStyle(TransferModel):
    def __init__(self, scale: float, noise_scale: float):
        self.scale, self.noise_scale = scale, noise_scale

    def predict(self, sources, design, control, columns):
        consensus, gate = _consensus_gate(sources, self.noise_scale)
        return self.scale * (gate * consensus)[:, list(columns)]


def _fit_transpert_style(sources, target, design, control, train, noise_scale: float = 0.5) -> TransferModel:
    consensus, gate = _consensus_gate(sources, noise_scale)
    summary = (gate * consensus)[:, list(train)]
    observed = np.asarray(target, dtype=float)[:, list(train)]
    denominator = float(np.sum(summary ** 2))
    scale = float(np.clip(np.sum(summary * observed) / denominator, 0.0, MAX_SCALE)) if denominator > EPS else 0.0
    return _TranspertStyle(scale, noise_scale)


def _pseudobulk_features(
    sources: Sequence[np.ndarray], design: np.ndarray, control: np.ndarray, columns: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The source mean (genes, columns) and the engineered features (genes, columns, 4) that correct it: the off-target source
    mean (a scale correction that the large on-target entries cannot distort), the on-target indicator, the on-target
    indicator times the design strength, and the off-target source mean times the gene's basal expression level.
    """
    cols = list(columns)
    mean = _stack(sources, cols).mean(axis=0)
    genes = mean.shape[0]
    on_target = (np.arange(genes)[:, None] == np.asarray(cols)[None, :]).astype(float)
    off_target_mean = mean * (1.0 - on_target)
    strength = np.diag(np.asarray(design, dtype=float))[cols][None, :]
    log_level = np.log(np.maximum(np.asarray(control, dtype=float), 1e-6))
    log_level = (log_level - log_level.mean())[:, None]
    return mean, np.stack([off_target_mean, on_target, on_target * strength, off_target_mean * log_level], axis=-1)


class _PseudobulkRidge(TransferModel):
    def __init__(self, coefficients: np.ndarray):
        self.coefficients = coefficients

    def predict(self, sources, design, control, columns):
        mean, features = _pseudobulk_features(sources, design, control, columns)
        return mean + features @ self.coefficients   # the source mean plus the learned correction


def _fit_pseudobulk_ridge(sources, target, design, control, train, ridge: float = 1.0) -> TransferModel:
    mean, features = _pseudobulk_features(sources, design, control, train)
    # Regress what the source mean gets wrong; with a strong ridge the model stays at the plain source average.
    residual = np.asarray(target, dtype=float)[:, list(train)] - mean
    flat = features.reshape(-1, features.shape[-1])
    gram = flat.T @ flat
    penalty = ridge * float(np.trace(gram)) / gram.shape[0] + EPS
    coefficients = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), flat.T @ residual.reshape(-1))
    return _PseudobulkRidge(coefficients)


TRANSFER_PREDICTORS: List[TransferPredictor] = [
    TransferPredictor(
        "atlas_mean", "Source-atlas average (zero-shot transfer)", "reference",
        "The mean of the source-context responses; no calibration to the target context.", _fit_atlas_mean),
    TransferPredictor(
        "atlas_calibrated", "AtlasShift, calibrated (weighted atlases + per-gene scale)", "method",
        "Sources weighted by how well they explain the seen target columns, then a per-gene scale fitted on them.",
        _fit_atlas_calibrated),
    TransferPredictor(
        "transpert_style", "TransPert-style (consensus + global scale)", "method",
        "Median response over the sources, kept only where all sources agree above the noise floor, times one global scale.",
        _fit_transpert_style),
    TransferPredictor(
        "pseudobulk_ridge", "Pseudo-bulk regression (ridge)", "method",
        "Pooled ridge regression of the target on the source mean, the on-target effect and expression-level interactions.",
        _fit_pseudobulk_ridge),
]
TRANSFER_BY_KEY: Dict[str, TransferPredictor] = {p.key: p for p in TRANSFER_PREDICTORS}


# ------------------------------------------------------------------------------------------------ evaluation
def predict_matrix(predictor: TransferPredictor, data: TransferData) -> np.ndarray:
    """
    Predicted target responses for every column (genes x perturbations). Held-out columns are predicted from a fit on all
    seen columns; each seen column is predicted from a fit on the other seen columns (leave-one-out), so no column is ever
    predicted from its own target measurements.
    """
    genes, perturbations = data.target.shape
    seen = list(data.seen)
    held = [j for j in range(perturbations) if j not in set(seen)]
    result = np.zeros((genes, perturbations))

    def fitted(train: Sequence[int]) -> TransferModel:
        return predictor.fit(data.sources, data.target, data.design, data.control, train)

    if held:
        result[:, held] = fitted(seen).predict(data.sources, data.design, data.control, held)
    for j in seen:
        train = [c for c in seen if c != j]
        if train:
            result[:, [j]] = fitted(train).predict(data.sources, data.design, data.control, [j])
        else:  # a single seen column: nothing to calibrate on, fall back to the plain average
            result[:, [j]] = _AtlasMean().predict(data.sources, data.design, data.control, [j])
    return result
