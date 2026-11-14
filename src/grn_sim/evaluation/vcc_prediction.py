"""
Predictions of post-perturbation cells for the VCC 2026 scorer, from the studio's own models.

The challenge asks for the single-cell expression after a perturbation the model has not measured in that context; the
scorer (:mod:`grn_sim.evaluation.vcc2026`) compares predicted cells with real ones. This module turns what the studio can
produce - a predicted mean response per perturbation - into cells, and scores a set of such predictions on one panel of
held-out perturbations:

* :func:`virtual_cells` samples reference cells from noise-free condition means (the cross-atlas contexts of the Virtual
  Cell tab, which the ODE / statistical models give as steady states);
* :func:`panel_from_screen` reads the cells of a simulated Perturb-seq screen with the guide calls of the measurement layer;
* :func:`structure_response` is the response predictor used for the screen's network-inference algorithms: the
  algorithm's ``q`` best regulators per target, a linear refit of their weights in log expression on the *seen*
  perturbations, and the response to a held-out knockdown as the fixed point of that model;
* :func:`score_prediction_rows` emits cells for every prediction (control cells rescaled by the predicted fold change,
  rounded to counts) and places them on the competition scale, next to the two anchors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grn_sim.evaluation import vcc2026 as v

DEFAULT_CELLS = 400            # cells per perturbation of a virtual reference (the challenge downsamples to 400)
HOLD_OUT_FRACTION = 0.25
MIN_HELD_OUT = 2
STRUCTURE_Q = 2                # regulators kept per target when a score matrix is turned into a structure
LFC_CLIP = 5.0                 # |log fold change| cap of a predicted response (exp(5) = 148-fold, beyond any CRISPRi effect)


# -- cells -----------------------------------------------------------------------------------------------------------------

def virtual_cells(profiles: Dict[str, np.ndarray], control: np.ndarray, genes: Sequence[str],
                  cells_per_condition: int = DEFAULT_CELLS, seed: int = 0, depth: float = 20000.0,
                  depth_sd: float = 0.35, extrinsic_sd: float = 0.15, control_label: str = v.CONTROL) -> v.Panel:
    """
    Poisson-sampled cells of every condition: a library size per cell (log-normal around ``depth``), a per-cell, per-gene
    extrinsic factor (log-normal, ``extrinsic_sd``) on the condition's mean composition, counts drawn from the result.
    ``profiles`` maps a perturbation label to its (G,) mean expression; ``control`` is the unperturbed mean.
    """
    rng = np.random.default_rng(seed)
    G = len(genes)
    blocks: List[np.ndarray] = []
    labels: List[str] = []
    for lab, mean in [(control_label, control)] + sorted(profiles.items()):
        mean = np.maximum(np.asarray(mean, dtype=float), 0.0)
        if mean.shape != (G,):
            raise ValueError(f"profile of {lab!r} must have {G} genes, got {mean.shape}")
        k = int(cells_per_condition)
        comp = mean[None, :] * np.exp(rng.normal(0.0, extrinsic_sd, size=(k, G)))
        tot = comp.sum(axis=1, keepdims=True)
        tot[tot <= 0] = 1.0
        lib = np.exp(rng.normal(np.log(depth), depth_sd, size=(k, 1)))
        blocks.append(rng.poisson(lib * comp / tot).astype(float))
        labels += [lab] * k
    return v.Panel(np.vstack(blocks), np.array(labels), list(genes), control_label)


def with_bystanders(panel: v.Panel, n_bystanders: int, share: float = 0.9, seed: int = 0, extrinsic_sd: float = 0.15,
                    prefix: str = "BY") -> v.Panel:
    """
    The panel with ``n_bystanders`` unregulated genes added: real screens measure thousands of genes the perturbed
    network does not touch, and they are what makes a knockdown a small share of a cell's library. Without them a
    60-gene panel lets every knockdown shift the counts-per-million of every other gene, and the context mean then
    "predicts" the direction of most responses. The bystanders take ``share`` of every cell's library (Poisson counts
    around log-normal means spread like the panel's own genes, with a per-cell, per-gene extrinsic factor); the panel's
    genes keep their counts. A fold change vector of the original genes is padded with 1 for the bystanders.
    """
    n = int(n_bystanders)
    if n <= 0:
        return panel
    if not 0.0 < share < 1.0:
        raise ValueError("share must be between 0 and 1")
    rng = np.random.default_rng([int(seed), 97])
    log_means = np.log(np.maximum(panel.counts.mean(axis=0), 1e-3))
    mu = np.exp(rng.normal(float(np.mean(log_means)), float(np.std(log_means)) or 1.0, size=n))
    mu = mu / mu.sum()
    lib = panel.counts.sum(axis=1) * share / (1.0 - share)
    comp = mu[None, :] * np.exp(rng.normal(0.0, extrinsic_sd, size=(panel.counts.shape[0], n)))
    comp = comp / comp.sum(axis=1, keepdims=True)
    extra = rng.poisson(lib[:, None] * comp).astype(float)
    genes = list(panel.genes) + [f"{prefix}{i + 1:05d}" for i in range(n)]
    return v.Panel(np.hstack([panel.counts, extra]), panel.labels, genes, panel.control)


def panel_from_screen(adata: Any, gene_names: Sequence[str], control: str = v.CONTROL,
                      use_called_guide: bool = True) -> v.Panel:
    """
    The cells of a simulated screen as a scorer panel. With the measurement layer the label is the guide the pipeline
    *called* (cells with one called guide; non-targeting guides are the control), as an analyst sees the data; cells without
    a single call and double perturbations are left out. Without guide calls the true condition label is used.
    """
    names = [str(g) for g in gene_names]
    index = set(names)
    obs = adata.obs
    if use_called_guide and "assigned_guide" in obs:
        guide = obs["assigned_guide"].astype(str).values
        status = obs["assignment_status"].astype(str).values
        lab = []
        for g, s in zip(guide, status):
            if s != "single" or "+" in g:
                lab.append("")
                continue
            gene = g.rsplit("_g", 1)[0]
            lab.append(gene if gene in index else control)
        labels = np.array(lab, dtype=object)
    else:
        labels = obs["perturbation"].astype(str).values.astype(object)
        labels = np.array([("" if "+" in lab else lab) for lab in labels], dtype=object)
    keep = labels != ""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return v.Panel(np.asarray(X, dtype=np.float64)[keep], labels[keep].astype(str), names, control)


def hold_out(labels: Sequence[str], seed: int, fraction: float = HOLD_OUT_FRACTION, minimum: int = MIN_HELD_OUT) -> List[str]:
    """The perturbations held out of training: ``fraction`` of ``labels`` (at least ``minimum``), drawn with ``seed``."""
    labels = sorted(set(labels))
    n = len(labels)
    if n < minimum + 1:
        raise ValueError(f"at least {minimum + 1} perturbations are needed to hold {minimum} out, got {n}")
    k = min(max(minimum, int(round(fraction * n))), n - 1)
    rng = np.random.default_rng(seed)
    return sorted(str(x) for x in rng.choice(np.array(labels, dtype=object), size=k, replace=False))


# -- the structure -> linear refit -> response predictor --------------------------------------------------------------------

def _top_q(scores: np.ndarray, q: int) -> np.ndarray:
    n = scores.shape[0]
    work = np.array(scores, dtype=float)
    np.fill_diagonal(work, -1.0)
    support = np.zeros((n, n), dtype=bool)
    for k in range(n):
        for reg in np.argsort(-work[:, k], kind="stable")[:q]:
            if work[reg, k] > 0:
                support[reg, k] = True
    return support


def _refit(support: np.ndarray, X: np.ndarray, nodes: Sequence[int], ridge: float = 1e-3) -> np.ndarray:
    """Least-squares weights ``W[l, k]`` (regulator l -> target k) of ``x_k = sum_l W[l, k] x_l`` on the support, fitted
    over the conditions that do not perturb k itself (there the equation has an exogenous term)."""
    n, p = X.shape
    W = np.zeros((n, n))
    nodes_arr = np.asarray(nodes)
    for k in range(n):
        parents = np.flatnonzero(support[:, k])
        cols = np.flatnonzero(nodes_arr != k)
        if parents.size == 0 or cols.size == 0:
            continue
        D = X[parents][:, cols].T
        W[parents, k] = np.linalg.solve(D.T @ D + ridge * np.eye(parents.size), D.T @ X[k, cols])
    return W


def _respond(W: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Fixed point ``x = W^T x + delta`` (``delta``: the exogenous shift per gene), as a least-squares solve."""
    n = W.shape[0]
    M = np.eye(n) - W.T
    try:
        x = np.linalg.solve(M, delta)
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(M) @ delta
    if not np.all(np.isfinite(x)):
        x = np.linalg.pinv(M) @ delta
    return np.clip(np.nan_to_num(x), -LFC_CLIP, LFC_CLIP)


class StructureResponse:
    """A fitted structure model: ``predict(gene, delta)`` is the log-expression response (G,) to an exogenous shift
    ``delta`` of ``gene``."""

    def __init__(self, W: np.ndarray, orientation: str, training_error: float):
        self.W, self.orientation, self.training_error = W, orientation, training_error

    def predict(self, gene: int, delta: float) -> np.ndarray:
        d = np.zeros(self.W.shape[0])
        d[gene] = delta
        return _respond(self.W, d)


def structure_response(scores: np.ndarray, X_seen: np.ndarray, U_seen: np.ndarray, nodes_seen: Sequence[int],
                       q: int = STRUCTURE_Q) -> StructureResponse:
    """
    From a score matrix ``scores[regulator, target]`` (any non-negative evidence): keep the ``q`` best regulators per target,
    refit their weights on the seen experiments (``X_seen[:, c]`` the response to the perturbation of ``nodes_seen[c]``,
    ``U_seen`` with its own shift on the diagonal), trying both orientations of the matrix and keeping the one that
    reproduces the seen responses better. Only seen columns are used.
    """
    S = np.abs(np.nan_to_num(np.asarray(scores, dtype=float)))
    best: Optional[Tuple[float, np.ndarray, str]] = None
    for name, cand in (("as given", S), ("transposed", S.T)):
        W = _refit(_top_q(cand, q), X_seen, nodes_seen)
        pred = np.column_stack([_respond(W, U_seen[:, c]) for c in range(X_seen.shape[1])])
        err = float(np.linalg.norm(X_seen - pred))
        if not np.isfinite(err):
            err = float("inf")
        if best is None or err < best[0]:
            best = (err, W, name)
    assert best is not None
    err, W, name = best
    return StructureResponse(W, name, err)


def fold_from_lfc(lfc: np.ndarray) -> np.ndarray:
    """Multiplicative change of the mean expression from a natural-log shift."""
    return np.exp(np.clip(np.asarray(lfc, dtype=float), -LFC_CLIP, LFC_CLIP))


# -- scoring a set of predictions ---------------------------------------------------------------------------------------------

@dataclass
class PredictionRow:
    """One predictor's folds (label -> (G,) fold change of every gene's mean, or ``None`` when unavailable)."""

    key: str
    label: str
    kind: str                     # algorithm | reference | oracle | control
    method: str = ""              # how the folds were produced (shown next to the label)
    folds: Optional[Dict[str, np.ndarray]] = None
    reason: str = ""


def score_prediction_rows(real: v.Panel, held_out: Sequence[str], rows: Sequence[PredictionRow], seed: int = 0,
                          cells_per_prediction: Optional[int] = None) -> Dict[str, Any]:
    """
    Scores every row on the panel of ``held_out`` perturbations of ``real``: the reference's cells of those perturbations
    plus its controls make the context; each row's folds are emitted as rescaled control cells (as many as the reference
    has, or ``cells_per_prediction``) and scored. The two anchors are appended as rows of their own, so the table shows
    where 0 and 1 sit. Returns ``{"rows", "anchors", "held_out", "seconds"}``.
    """
    t = time.time()
    reference = real.subset(held_out)
    scorer = v.Scorer(reference, seed)
    ctrl_pool = reference.counts[reference.rows(reference.control)]
    perts = scorer.prepared.perturbations
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        base = {"key": row.key, "label": row.label, "kind": row.kind, "method": row.method}
        if row.folds is None:
            out.append({**base, "available": False, "reason": row.reason, "overall": None, "members": {}})
            continue
        missing = [p for p in perts if p not in row.folds]
        if missing:
            out.append({**base, "available": False, "reason": f"no prediction for {missing[0]}", "overall": None, "members": {}})
            continue
        rng = np.random.default_rng([int(seed), i])
        blocks, labels = [], []
        try:
            for p in perts:
                n = int(cells_per_prediction or scorer.prepared.sizes[p])
                fold = np.asarray(row.folds[p], dtype=float)
                if fold.size < real.n_genes:                      # bystander genes: no predicted change
                    fold = np.concatenate([fold, np.ones(real.n_genes - fold.size)])
                blocks.append(v.emit_cells(ctrl_pool, fold, n, rng))
                labels += [p] * n
            res = scorer.score(v.Panel(np.vstack(blocks), np.array(labels), list(real.genes), real.control))
        except ValueError as exc:                      # a prediction the upload contract rejects is reported, not scored
            out.append({**base, "available": False, "reason": str(exc), "overall": None, "members": {}})
            continue
        out.append({**base, "available": True, "reason": "", "overall": res["overall"], "members": res["members"],
                    "n_scaled": res["n_scaled"]})
    anchors = scorer.anchors()
    unresolved = anchors.get("unresolved", {})
    for key, label, values, scaled in (("baseline", "Context mean (baseline, 0 by definition)", anchors["baseline"], 0.0),
                                       ("replicate", "Replicate experiment (anchor, 1 by definition)", anchors["replicate"], 1.0)):
        members = {m: {"raw": values[m], "scaled": None if (values[m] is None or m in unresolved) else scaled,
                       "reason": unresolved.get(m, ""), "n": None} for m in v.METRICS}
        out.append({"key": key, "label": label, "kind": "anchor", "method": "measured on the reference", "available": True,
                    "reason": "", "overall": scaled, "members": members, "n_scaled": len(v.METRICS) - len(unresolved)})
    return {"rows": out, "anchors": anchors, "held_out": list(perts), "seconds": round(time.time() - t, 1),
            "metrics": list(v.METRICS), "labels": dict(v.LABELS)}


# -- the screen benchmark's algorithms as predictors -------------------------------------------------------------------------

def _restrict_ace(ace: Any, cols: np.ndarray) -> Any:
    """The ACE estimate of the seen nodes only (so no estimator sees a held-out perturbation)."""
    return replace(ace, nodes=ace.nodes[cols], lfc=ace.lfc[cols], se=ace.se[cols], z=ace.z[cols], pvalue=ace.pvalue[cols],
                   qvalue=ace.qvalue[cols], n_cells=ace.n_cells[cols])


def screen_predictions(run: Any, ace: Any, algorithms: Sequence[Any], seed: int, cells: Optional[np.ndarray] = None,
                       cell_labels: Optional[np.ndarray] = None) -> Tuple[v.Panel, List[str], List[PredictionRow]]:
    """
    The VCC task on one simulated screen: a quarter of the perturbed genes is held out; every algorithm of the screen
    benchmark is fitted on the seen perturbations' effects (``ace``) and predicts the held-out knockdowns through
    :func:`structure_response`, with the median on-target effect of the seen knockdowns as the assumed dose. Reference
    rows: the true structure through the same refit, the simulator's noise-free means, and a target-only prior.

    Returns the real panel (cells with called guides), the held-out labels and the prediction rows.
    """
    names = list(run.screen.gene_names)
    panel = panel_from_screen(run.adata, names)
    node_names = [names[int(g)] for g in ace.nodes]
    present = [g for g in node_names if g in set(panel.perturbations)]
    held = hold_out(present, seed)
    held_set = set(held)
    seen_cols = np.array([c for c, g in enumerate(node_names) if g not in held_set], dtype=int)
    X, U = ace.to_xu()
    X_seen, U_seen = X[:, seen_cols], U[:, seen_cols]
    nodes_seen = ace.nodes[seen_cols]
    ace_seen = _restrict_ace(ace, seen_cols)
    delta = float(np.median(ace.own[seen_cols]))
    gene_of = {g: i for i, g in enumerate(names)}
    held_genes = {g: gene_of[g] for g in held}
    if cells is not None and cell_labels is not None:
        keep = np.array([lab not in held_set for lab in cell_labels])
        cells_seen: Optional[np.ndarray] = cells[keep]
    else:
        cells_seen = None

    def rows_from_scores(S: np.ndarray) -> Dict[str, np.ndarray]:
        model = structure_response(S, X_seen, U_seen, nodes_seen)
        return {g: fold_from_lfc(model.predict(j, delta)) for g, j in held_genes.items()}

    rows: List[PredictionRow] = []
    for alg in algorithms:
        rng = np.random.default_rng(seed * 7919 + 13)
        try:
            if alg.uses_cells:
                if cells_seen is None:
                    raise ValueError("this algorithm needs the cells of the seen conditions")
                S = alg.score(ace_seen, X_seen, U_seen, rng, cells_seen)
            else:
                S = alg.score(ace_seen, X_seen, U_seen, rng)
            folds = rows_from_scores(S)
            rows.append(PredictionRow(alg.name, alg.label, "control" if alg.family == "control" else "algorithm",
                                      "structure + linear refit", folds))
        except Exception as exc:                        # a failing algorithm is reported, never scored as chance
            rows.append(PredictionRow(alg.name, alg.label, "algorithm", "structure + linear refit", None,
                                      f"{type(exc).__name__}: {exc}"))
    truth = np.abs(np.asarray(run.screen.adjacency, dtype=float))
    rows.append(PredictionRow("oracle_structure", "True network (oracle structure)", "oracle", "structure + linear refit",
                              rows_from_scores(truth)))
    ref = np.maximum(np.asarray(run.screen.reference, dtype=float), 1e-12)
    single = run.screen.single_index
    means = {g: np.maximum(np.asarray(run.screen.means[:, single[j]], dtype=float), 0.0) / ref
             for g, j in held_genes.items() if j in single}
    if len(means) == len(held_genes):
        rows.append(PredictionRow("truth_means", "Simulator truth (noise-free means)", "oracle", "the generator's own condition means", means))
    self_only: Dict[str, np.ndarray] = {}
    for g, j in held_genes.items():
        f = np.ones(len(names))
        f[j] = float(np.exp(np.clip(delta, -LFC_CLIP, LFC_CLIP)))
        self_only[g] = f
    rows.append(PredictionRow("self_prior", "Target-gene prior (own knockdown only)", "reference", "no model", self_only))
    return panel, held, rows
