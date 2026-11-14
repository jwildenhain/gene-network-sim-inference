"""
Consensus of interventional-aware network scores.

Community prediction beat single methods in DREAM5, but the members there were diverse expression-only methods. A
consensus of perturbation screens should be built from methods that use the design (which gene was perturbed), and
should be judged against its best member: it is worth keeping only if it beats that member.

``rank_normalise`` maps a score matrix to [0, 1] by the rank of |score| among the scorable entries (rows of perturbed
genes, off-diagonal; ties share their mean rank), so methods on different scales can be averaged. ``consensus`` is the
(weighted) mean of the rank-normalised matrices (a Borda count); ``agreement`` counts the members that put an entry in
their own top ``top_fraction``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import rankdata


def _mask(S: np.ndarray, rows: Optional[np.ndarray]) -> np.ndarray:
    m = ~np.eye(S.shape[0], dtype=bool)
    if rows is not None:
        m &= np.asarray(rows, dtype=bool)[:, None]
    return m


def rank_normalise(S: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    S = np.abs(np.real(np.asarray(S, dtype=float)))
    m = _mask(S, rows)
    out = np.zeros_like(S)
    vals = S[m]
    out[m] = (rankdata(vals, method="average") - 1.0) / max(len(vals) - 1, 1)
    return out


def consensus(scores: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None,
              rows: Optional[np.ndarray] = None) -> np.ndarray:
    if not scores:
        raise ValueError("no score matrices")
    total, acc = 0.0, None
    for name, S in scores.items():
        w = 1.0 if weights is None else float(weights.get(name, 1.0))
        R = rank_normalise(S, rows) * w
        acc = R if acc is None else acc + R
        total += w
    assert acc is not None
    return acc / total


def agreement(scores: Dict[str, np.ndarray], top_fraction: float = 0.02, rows: Optional[np.ndarray] = None) -> np.ndarray:
    out: Optional[np.ndarray] = None
    for S in scores.values():
        R = rank_normalise(S, rows)
        m = _mask(R, rows)
        thr = np.quantile(R[m], 1.0 - top_fraction)
        hit = (R >= thr) & m
        out = hit.astype(int) if out is None else out + hit
    if out is None:
        raise ValueError("agreement needs at least one score matrix")
    return out
