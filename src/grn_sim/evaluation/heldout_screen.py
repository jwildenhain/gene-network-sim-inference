"""
Held-out-perturbation benchmark on a simulated single-cell screen (section 3 of the Perturb-seq proposal).

The network benchmarks (:mod:`grn_sim.evaluation.screen_benchmark`) ask whether a method recovers *edges*. This one asks the
question a virtual cell is built for: given the responses to some perturbations, predict the response to one that was never
seen. The perturbed genes are split into folds; for every held-out gene ``a`` a predictor sees only the training
perturbations' responses (estimated from the simulated cells, i.e. what an analyst has) and returns a response vector over all
genes. It is scored against the simulator's noise-free total effects of ``a``, on the off-target genes only (the perturbed
gene's own knockdown is trivial to predict and would inflate every score).

Predictors, from nothing to something:

``control``          no effect anywhere (the null every predictor must beat on error)
``train_mean``       the average training response (what the gene-level baseline of the proposal calls the training mean)
``random_response``  the response of a random training perturbation (random-structure baseline: right scale, wrong genes)
``coexpression``     the responses of the training perturbations whose genes behave most like the held-out gene across the
                     training perturbations (k nearest neighbours by correlation of response profiles)
``linear``           ridge regression of the held-out gene's response profile on the profiles of the training genes; the
                     fitted weights mix the training responses

Metrics per held-out perturbation (off-target genes): ``pearson_delta`` (correlation of predicted and true response),
``deg_mse_skill`` (1 - MSE / MSE of the control on the genes the perturbation truly moved: positive means better than
predicting no effect), ``direction`` (share of those genes whose sign is right) and ``top_overlap`` (share of the truly
affected genes among the k largest predicted effects, k = their number). Strata split the held-out perturbations by the
number of affected genes: ``silent`` perturbations move no off-target gene (only the correlation is defined for them) and ``active`` ones
move at least one, which is where a model can show anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PREDICTORS = ("control", "train_mean", "random_response", "coexpression", "linear")
METRICS = ("pearson_delta", "deg_mse_skill", "direction", "top_overlap")


# -- predictors ----------------------------------------------------------------------------------------------------------
def _profiles(R: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """``(n, m)``: the response of every gene across the m training perturbations, the perturbed gene's own entry zeroed."""
    P = R.T.copy()
    P[nodes, np.arange(len(nodes))] = 0.0
    return P


def _zscore(P: np.ndarray) -> np.ndarray:
    P = P - P.mean(axis=1, keepdims=True)
    sd = np.linalg.norm(P, axis=1, keepdims=True)
    return P / np.where(sd > 1e-12, sd, 1.0)


def predict(method: str, R_train: np.ndarray, nodes_train: np.ndarray, nodes_test: Any,
            rng: np.random.Generator, k: int = 5, ridge: float = 1.0) -> np.ndarray:
    """Predicted responses ``(len(nodes_test), n_genes)`` of the held-out genes from the training responses ``R_train (m, n)``."""
    nodes_test = np.asarray(nodes_test, dtype=int)
    m, n = R_train.shape
    R_train = R_train.copy()
    R_train[np.arange(m), nodes_train] = 0.0           # a perturbed gene's own knockdown belongs to that gene, not to others
    out = np.zeros((len(nodes_test), n))
    if method == "control":
        return out
    if method == "train_mean":
        return out + R_train.mean(axis=0)
    if method == "random_response":
        return R_train[rng.integers(0, m, size=len(nodes_test))].copy()
    if method not in ("coexpression", "linear"):
        raise ValueError(f"unknown predictor {method!r}; choose from {', '.join(PREDICTORS)}")
    P = _profiles(R_train, nodes_train)                    # (n, m)
    Z = _zscore(P)
    train_rows = Z[nodes_train]                            # profiles of the training genes, (m, m)
    for i, a in enumerate(nodes_test):
        s = train_rows @ Z[a]                              # similarity of gene a to every training gene
        if method == "coexpression":
            top = np.argsort(-s)[: min(k, m)]
            w = np.clip(s[top], 0.0, None)
            out[i] = (w @ R_train[top] / w.sum()) if w.sum() > 0 else R_train.mean(axis=0)
        else:
            G = train_rows @ train_rows.T                  # (m, m)
            c = np.linalg.solve(G + ridge * np.eye(m), s)
            out[i] = c @ R_train
    return out


# -- metrics -------------------------------------------------------------------------------------------------------------
def score_one(pred: np.ndarray, truth: np.ndarray, gene: int, threshold: float) -> Dict[str, float]:
    """Metrics of one held-out perturbation on the off-target genes; NaN where undefined (no truly affected gene)."""
    keep = np.arange(len(truth)) != gene
    p, t = pred[keep], truth[keep]
    out = {m: float("nan") for m in METRICS}
    out["pearson_delta"] = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-12 and t.std() > 1e-12 else 0.0
    hit = np.abs(t) >= threshold
    out["n_affected"] = float(hit.sum())
    if hit.any():
        base = float(np.mean(t[hit] ** 2))
        out["deg_mse_skill"] = float(1.0 - np.mean((p[hit] - t[hit]) ** 2) / base) if base > 0 else float("nan")
        if np.any(p != 0):                      # an all-zero prediction (the control) has no direction or ranking
            out["direction"] = float(np.mean(np.sign(p[hit]) == np.sign(t[hit])))
            top = np.argsort(-np.abs(p))[: int(hit.sum())]
            out["top_overlap"] = float(hit[top].mean())
    return out


# -- the benchmark -------------------------------------------------------------------------------------------------------
def truth_responses(screen: Any, nodes: np.ndarray) -> np.ndarray:
    """``(len(nodes), n)``: the simulator's noise-free response to each single perturbation, in the order of ``nodes``."""
    column = {c[0]: k for k, c in enumerate(screen.conditions) if len(c) == 1}
    return np.stack([np.asarray(screen.total_effects[:, column[int(g)]], dtype=float) for g in nodes])


def evaluate_heldout(screen: Any, ace: Any, predictors: Sequence[str] = PREDICTORS, n_folds: int = 5, seed: int = 0,
                     threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Cross-validated held-out-perturbation prediction. ``ace`` holds the responses estimated from cells (``lfc``, ``nodes``);
    the score is against the noise-free truth of ``screen``. Returns ``{predictor: {stratum: {metric: mean}}}`` plus ``n``.
    """
    thr = screen.effect_threshold if threshold is None else float(threshold)
    nodes = np.asarray(ace.nodes, dtype=int)
    R = np.asarray(ace.lfc, dtype=float)
    T = truth_responses(screen, nodes)
    p = len(nodes)
    if p < 2 * n_folds:
        raise ValueError(f"{p} perturbed genes are too few for {n_folds} folds")
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(p), n_folds)
    rows: Dict[str, List[Dict[str, float]]] = {m: [] for m in predictors}
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(p), test)
        for m in predictors:
            pred = predict(m, R[train], nodes[train], nodes[test], np.random.default_rng(seed * 1000 + f))
            for row, i in zip(pred, test):
                rows[m].append(score_one(row, T[i], int(nodes[i]), thr))
    n_aff = np.array([r["n_affected"] for r in rows[predictors[0]]])
    strata = {"all": np.ones(p, dtype=bool), "silent": n_aff == 0, "active": n_aff > 0}
    out: Dict[str, Any] = {"n_perturbations": p, "n_folds": n_folds, "median_affected": float(np.median(n_aff))}
    for m in predictors:
        out[m] = {name: {**{met: _nanmean([rows[m][i][met] for i in np.flatnonzero(sel)]) for met in METRICS},
                         "n": int(sel.sum())} for name, sel in strata.items()}
    return out


def _nanmean(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")
