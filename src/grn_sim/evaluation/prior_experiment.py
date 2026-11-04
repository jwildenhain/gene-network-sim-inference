"""
How much is a prior network worth? A controlled experiment on simulated screens.

The prior is made from the truth with chosen quality: it keeps a fraction ``sensitivity`` of the true direct edges and
adds false pairs until a fraction ``precision`` of the prior is true. That grid (and a prior with no true edges) shows
where a prior starts to help a perturbation-based method and where it starts to hurt, which published benchmarks
cannot show because their priors (ChIP, motifs, STRING) have unknown quality.

Three numbers per cell of the grid, all AUPR against the direct truth on the perturbed genes:

* ``prior_alone``: the prior itself used as the score (what the prior is worth without any data);
* ``with_prior``: the method run with the prior (lower L1 penalty for inspre on prior pairs, ``boost`` times the
  score for the other methods);
* ``novel``: ``with_prior`` scored only on pairs that are *not* in the prior, i.e. what the prior helped to find
  beyond itself (compare with the same restriction applied to the no-prior run).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from grn_sim.evaluation.metrics import compute_aupr


def make_prior(direct: np.ndarray, rows: np.ndarray, sensitivity: float, precision: float,
               rng: np.random.Generator) -> np.ndarray:
    """Boolean [regulator, target] prior of the given sensitivity and precision (precision 0 -> only false pairs)."""
    direct = np.asarray(direct, dtype=bool)
    n = direct.shape[0]
    allowed = rows[:, None] & ~np.eye(n, dtype=bool)
    true_idx = np.flatnonzero((direct & allowed).ravel())
    n_tp = 0 if precision <= 0 else int(round(sensitivity * len(true_idx)))
    tp = rng.choice(true_idx, size=n_tp, replace=False) if n_tp else np.array([], dtype=int)
    prior = np.zeros(n * n, dtype=bool)
    prior[tp] = True
    if precision <= 0:
        n_fp = int(round(sensitivity * len(true_idx)))     # as large as the prior of that sensitivity, all wrong
    else:
        n_fp = int(round(n_tp * (1.0 - precision) / precision))
    false_idx = np.flatnonzero((~direct & allowed).ravel())
    if n_fp:
        prior[rng.choice(false_idx, size=min(n_fp, len(false_idx)), replace=False)] = True
    return prior.reshape(n, n)


def prior_penalty(prior: np.ndarray, strength: float = 0.8) -> np.ndarray:
    """L1 penalty multipliers for inspre: 1 off the prior, ``1 - strength`` on it."""
    return 1.0 - strength * np.asarray(prior, dtype=float)


def boost_scores(S: np.ndarray, prior: np.ndarray, boost: float = 2.0) -> np.ndarray:
    return np.abs(S) * np.where(prior, boost, 1.0)


def aupr_on(direct: np.ndarray, S: np.ndarray, rows: np.ndarray, exclude: Optional[np.ndarray] = None) -> float:
    n = direct.shape[0]
    m = rows[:, None] & ~np.eye(n, dtype=bool)
    if exclude is not None:
        m &= ~exclude
    return compute_aupr(np.asarray(direct)[m], np.abs(np.asarray(S))[m])


def summarise(direct: np.ndarray, rows: np.ndarray, prior: np.ndarray, S_no: np.ndarray, S_with: np.ndarray) -> Dict[str, float]:
    return {
        "no_prior": aupr_on(direct, S_no, rows),
        "prior_alone": aupr_on(direct, prior.astype(float), rows),
        "with_prior": aupr_on(direct, S_with, rows),
        "novel_no_prior": aupr_on(direct, S_no, rows, exclude=prior),
        "novel_with_prior": aupr_on(direct, S_with, rows, exclude=prior),
        "prior_precision": float((prior & direct).sum() / max(prior.sum(), 1)),
        "prior_sensitivity": float((prior & direct & rows[:, None]).sum() / max((direct & rows[:, None]).sum(), 1)),
    }
