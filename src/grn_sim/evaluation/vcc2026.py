"""
The Virtual Cell Challenge 2026 scorer on in-memory count matrices.

A port of the ``vcc2026`` metric profile of Arc's ``cell-eval2`` (the scorer behind https://virtualcellchallenge.org, metric
specification ``docs/vcc2026_metrics/vcc2026-metrics.md`` of that repository, 2026-08-19) to the simulator's data, written
against the specification and the scorer's source. ``cell-eval2`` needs Python 3.11, so it cannot be installed next to this
package; the port follows its data preparation step by step:

* **Profiles** (expression metrics): counts are summed over the cells of a group, normalised to ``TS = 5e4`` and
  log-transformed, ``b = log1p(TS * P / sum P)``; the sampling noise of a profile is a delete-one-cell jackknife ``C``.
* **Differential expression**: per cell counts per million, a two-sided Wilcoxon rank-sum test of every gene (normal
  approximation with tie correction, as scanpy's ``rank_genes_groups``), genes tested only when the control's mean CPM is
  above 5, Benjamini-Hochberg within each perturbation, log2 fold change of the arithmetic means with ``epsilon = 1e-9``.
* **Six metrics**, each with the perturbed gene's own row excluded: perturbation discrimination (``pds``), the unbiased
  capped normalised expression error (``mse``), direction fidelity with yield (``fid``), direction reach (``reach``),
  significant-set Jaccard (``jac``) and the normalised MAE of the fold changes (``nmae``).
* **Anchors and scaling**: the baseline is the context's mean response (the mean per-perturbation profile, each gene's own
  perturbation left out, emitted as resampled, rescaled control cells); the replicate is the reference scored against
  itself over five disjoint half-splits; every member is placed on ``s = (u - b) / (r - b)`` (0 = baseline, 1 = replicate),
  ``mse`` clamped to [0, 1], ``nmae`` floored at -6, the overall score is the plain mean of the six.

Everything here works on dense numpy arrays of a few hundred genes, which is what the studio simulates; the competition's
18,533-gene panels are out of scope. Nothing in this module makes up a number: a member that cannot be computed on a panel
(no gene passes the gate, a degenerate anchor) is reported as unavailable with its reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

TS = 5.0e4                     # bulk_target_sum of the expression comparator
DE_TARGET_SUM = 1.0e6          # per-cell normalisation of the differential-expression test (counts per million)
CPM_MIN = 5.0                  # a gene is tested when its mean CPM in the control cells exceeds this
ALPHA = 0.05                   # Benjamini-Hochberg threshold, per perturbation
EPSILON = 1.0e-9               # pseudocount of the log2 fold change
REACH_PURITY = 0.9             # pass mark of the direction-reach prefix
NMAE_MIN_GATE = 10             # smallest reference gate a perturbation needs to be scored on nmae
N_SPLITS = 5                   # half-splits of the replicate anchor
BASE_SEED = 0                  # their seeds come from numpy.random.SeedSequence(BASE_SEED)
MAX_COUNTS_PER_CELL = 1.0e6
CONTROL = "non-targeting"

METRICS: Tuple[str, ...] = ("pds", "mse", "jac", "nmae", "fid", "reach")       # leaderboard order
CANONICAL = {"pds": "pds_cosine", "mse": "expr_mse_unbiased_capped_norm", "jac": "de_wilcoxon_sig_jaccard",
             "nmae": "de_wilcoxon_lfc_nmae", "fid": "de_wilcoxon_direction_fidelity_yield_raw",
             "reach": "de_wilcoxon_direction_reach_raw"}
LABELS = {"pds": "Perturbation discrimination", "mse": "Expression accuracy (MSE)", "jac": "DE significance overlap (Jaccard)",
          "nmae": "DE log-FC accuracy (NMAE)", "fid": "DE direction fidelity", "reach": "DE direction reach"}
LOWER_IS_BETTER = {"mse", "nmae"}
CLAMP = {"mse": (0.0, 1.0), "nmae": (-6.0, None)}           # scaled-score clamps; the other four are unclamped
WORST_RAW = {"pds": 0.0, "mse": None, "jac": 0.0, "nmae": None, "fid": 0.0, "reach": 0.0}   # structural worst raw value


# -- data ------------------------------------------------------------------------------------------------------------------

@dataclass
class Panel:
    """Single-cell counts of one cell context: ``counts[cell, gene]``, one perturbation label per cell (gene symbol, or
    ``control`` for the non-targeting cells) and the gene names. Labels that are not gene symbols are allowed but then
    exclude nothing (the competition resolves ``target_gene`` against the gene axis exactly as this does)."""

    counts: np.ndarray
    labels: np.ndarray
    genes: List[str]
    control: str = CONTROL

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=np.float64)
        self.labels = np.asarray(self.labels).astype(str)
        self.genes = [str(g) for g in self.genes]
        if self.counts.ndim != 2 or self.counts.shape != (self.labels.size, len(self.genes)):
            raise ValueError(f"counts must be (cells, genes) = ({self.labels.size}, {len(self.genes)}), got {self.counts.shape}")

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def perturbations(self) -> List[str]:
        """The non-control labels, sorted."""
        return sorted(set(self.labels.tolist()) - {self.control})

    def rows(self, label: str) -> np.ndarray:
        return np.flatnonzero(self.labels == label)

    def subset(self, labels: Sequence[str], keep_control: bool = True) -> "Panel":
        want = set(labels) | ({self.control} if keep_control else set())
        keep = np.array([lab in want for lab in self.labels])
        return Panel(self.counts[keep], self.labels[keep], list(self.genes), self.control)

    @classmethod
    def from_anndata(cls, adata: Any, pert_col: str = "perturbation", control: str = CONTROL) -> "Panel":
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        return cls(np.asarray(X, dtype=np.float64), adata.obs[pert_col].astype(str).values, list(adata.var_names), control)


def validate_submission(pred: Panel, real: Panel, allow_fractional: bool = False) -> None:
    """The upload contract: finite, non-negative (integral) counts, no cell above 1e6 counts, exactly the reference's
    perturbations. Raises ``ValueError`` with the first failure."""
    if pred.genes != real.genes:
        raise ValueError("the prediction's genes must equal the reference's, in the same order")
    X = pred.counts
    if not np.all(np.isfinite(X)):
        raise ValueError("the prediction contains non-finite values")
    if np.any(X < 0):
        raise ValueError("the prediction contains negative counts")
    if not allow_fractional and np.any(X != np.round(X)):
        raise ValueError("the prediction contains fractional values; raw counts are required")
    if np.any(X.sum(axis=1) > MAX_COUNTS_PER_CELL):
        raise ValueError(f"a predicted cell has more than {MAX_COUNTS_PER_CELL:g} counts")
    want, have = set(real.perturbations), set(pred.perturbations)
    if have != want:
        extra, missing = sorted(have - want), sorted(want - have)
        raise ValueError(f"perturbation sets differ: extra {extra[:5]}, missing {missing[:5]}")


# -- expression profiles and their sampling noise ---------------------------------------------------------------------------

def bulk_profile(counts: np.ndarray) -> np.ndarray:
    """``log1p(TS * P / sum P)`` of the summed counts of a group (zeros for an all-zero group)."""
    P = np.asarray(counts, dtype=np.float64).sum(axis=0)
    S = P.sum()
    return np.log1p(TS * P / S) if S > 0 else np.zeros_like(P)


def jackknife_per_gene(counts: np.ndarray) -> np.ndarray:
    """
    Delete-one-cell jackknife of the profile's sampling noise, per gene: ``q_g = (n-1)/n sum_i (v_ig - mean_i v_ig)^2``
    with ``v_ig = log1p(TS (P_g - y_ig) / (S - l_i))``, so that ``C = sum_g q_g``; zeros for fewer than two cells. Computed
    as the deviation from the full profile (the same two-pass form as cell-eval2), each gene's term clamped at zero.
    """
    Y = np.asarray(counts, dtype=np.float64)
    n, G = Y.shape
    if n < 2:
        return np.zeros(G)
    P = Y.sum(axis=0)
    S = P.sum()
    lib = Y.sum(axis=1)
    r = S - lib
    if np.any(r < 0):
        raise ValueError("negative leave-one-out remainder: the input is not counts")
    safe = np.where(r > 0, r, 1.0)
    V = np.log1p(TS * (P[None, :] - Y) / safe[:, None])
    V[r <= 0] = 0.0
    b = np.log1p(TS * P / (S if S > 0 else 1.0))
    dev = V - b[None, :]
    s1 = dev.sum(axis=0)
    s2 = (dev * dev).sum(axis=0)
    return np.maximum((n - 1) / n * (s2 - s1 * s1 / n), 0.0)


def jackknife_correction(counts: np.ndarray) -> float:
    """``C``, the per-gene jackknife summed over the genes."""
    return float(jackknife_per_gene(counts).sum())


# -- differential expression ------------------------------------------------------------------------------------------------

def cpm(counts: np.ndarray, target_sum: float = DE_TARGET_SUM) -> np.ndarray:
    """Per-cell normalisation to ``target_sum`` (an empty cell stays zero)."""
    X = np.asarray(counts, dtype=np.float64)
    lib = X.sum(axis=1)
    scale = np.where(lib > 0, target_sum / np.where(lib > 0, lib, 1.0), 0.0)
    return X * scale[:, None]


def wilcoxon_pvalues(group: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Two-sided Mann-Whitney / Wilcoxon rank-sum p-value per column (gene): normal approximation of the rank sum with the tie
    correction, no continuity correction (scanpy ``rank_genes_groups(method="wilcoxon", tie_correct=True)``). A gene whose
    values all tie (zero variance) gets p = 1.
    """
    from scipy.stats import norm

    A = np.asarray(group, dtype=np.float64)
    B = np.asarray(reference, dtype=np.float64)
    n, m = A.shape[0], B.shape[0]
    N = n + m
    pooled = np.vstack([A, B])
    G = pooled.shape[1]
    if N < 2:
        return np.ones(G)
    # One argsort per gene gives the sorted values, the runs of ties (their average rank and size) and which sorted
    # positions belong to the group; scipy's rankdata would sort twice more.
    order = np.argsort(pooled, axis=0, kind="stable")
    srt = np.take_along_axis(pooled, order, axis=0)
    starts = np.vstack([np.ones((1, G), dtype=bool), srt[1:] != srt[:-1]])           # first element of every run of ties
    run_id = np.cumsum(starts, axis=0) - 1                                           # per column, 0-based run index
    flat = (run_id + np.arange(G)[None, :] * N).ravel()
    sizes = np.bincount(flat, minlength=G * N).astype(np.float64)                     # (G * N,) cells per run
    pos = np.broadcast_to(np.arange(N, dtype=np.float64)[:, None], (N, G))
    first = np.bincount(flat, weights=(pos * starts).ravel(), minlength=G * N)       # 0-based position of each run's start
    avg_rank = first + (sizes - 1.0) / 2.0 + 1.0                                     # mid-rank, 1-based
    ranks = avg_rank[flat].reshape(N, G)                                             # rank of every sorted position
    in_group = (order < n).astype(np.float64)
    R = (ranks * in_group).sum(axis=0)
    sizes_g = sizes.reshape(G, N)
    T = 1.0 - (sizes_g ** 3 - sizes_g).sum(axis=1) / (N ** 3 - N)
    std = np.sqrt(T * n * m * (N + 1) / 12.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (R - n * (N + 1) / 2.0) / std
    z = np.where(np.isfinite(z), z, 0.0)
    return 2.0 * norm.sf(np.abs(z))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (NaN stays NaN and is left out of the family)."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    q = p[ok]
    m = q.size
    order = np.argsort(q)
    ranked = q[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


@dataclass
class DETable:
    """Per-perturbation differential expression of one side: arrays over the tested genes (``kept``)."""

    kept: np.ndarray                                   # (G,) bool: the genes tested (control mean CPM > CPM_MIN)
    lfc: Dict[str, np.ndarray] = field(default_factory=dict)       # label -> (G,) log2 fold change (NaN where not kept)
    p_value: Dict[str, np.ndarray] = field(default_factory=dict)   # label -> (G,) raw p (NaN where not kept)
    p_adj: Dict[str, np.ndarray] = field(default_factory=dict)     # label -> (G,) BH-adjusted p (NaN where not kept)

    def significant(self, label: str) -> np.ndarray:
        """(G,) bool: tested, defined and significant."""
        q = self.p_adj[label]
        return np.isfinite(q) & (q < ALPHA)


def differential_expression(cells: Panel, control_cells: np.ndarray, labels: Optional[Sequence[str]] = None) -> DETable:
    """
    The DE table of every perturbation of ``cells`` against ``control_cells`` (raw counts of the reference group). The
    gene gate reads the control alone; BH runs within each perturbation over the tested genes; a NaN fold change is masked
    to p_adj = 1 (cell-eval2's ``nan_lfc_policy="mask"``).
    """
    ctrl = cpm(control_cells)
    if ctrl.shape[0] == 0:
        raise ValueError("the control group has no cells")
    ref_mean = ctrl.mean(axis=0)
    kept = ref_mean > CPM_MIN
    table = DETable(kept=kept)
    labels = list(cells.perturbations) if labels is None else list(labels)
    G = cells.n_genes
    for lab in labels:
        rows = cells.rows(lab)
        if rows.size == 0:
            raise ValueError(f"perturbation {lab!r} has no cells")
        grp = cpm(cells.counts[rows])
        lfc = np.full(G, np.nan)
        p = np.full(G, np.nan)
        q = np.full(G, np.nan)
        if kept.any():
            mean = grp.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                lfc[kept] = np.log2((mean[kept] + EPSILON) / (ref_mean[kept] + EPSILON))
            p[kept] = wilcoxon_pvalues(grp[:, kept], ctrl[:, kept])
            q[kept] = benjamini_hochberg(p[kept])
            masked = kept & ~np.isfinite(lfc)
            q[masked] = 1.0
        table.lfc[lab], table.p_value[lab], table.p_adj[lab] = lfc, p, q
    return table


# -- the prepared reference --------------------------------------------------------------------------------------------------

@dataclass
class Prepared:
    """Everything the six metrics read from one side: per-group profiles, jackknife corrections, cell counts and the DE."""

    panel: Panel
    perturbations: List[str]
    profiles: Dict[str, np.ndarray]           # label -> (G,) bulk profile (control included)
    corrections: Dict[str, float]             # label -> jackknife C (control included)
    sizes: Dict[str, int]                     # label -> cells
    de: DETable
    target_index: Dict[str, int]              # label -> gene column of its own target (resolved labels only)
    control_counts: np.ndarray                # the raw counts of the control cells used by this side's DE
    corrections_gene: Dict[str, np.ndarray] = field(default_factory=dict)   # label -> (G,) per-gene jackknife terms


def resolve_targets(labels: Sequence[str], genes: Sequence[str]) -> Dict[str, int]:
    """Label -> column of the gene it names (exact match); labels naming no gene are absent and exclude nothing."""
    index = {g: i for i, g in enumerate(genes)}
    return {lab: index[lab] for lab in labels if lab in index}


def prepare(panel: Panel, control_counts: Optional[np.ndarray] = None, labels: Optional[Sequence[str]] = None) -> Prepared:
    """
    Prepares one side. ``control_counts`` are the raw counts of the control cells the DE and the effects are measured
    against; by default the panel's own control cells (``control_source="pred"``), for a scored submission the reference's
    (``control_source="real"``, the competition rule: the platform concatenates the held-out real controls onto the
    prediction).
    """
    perts = list(panel.perturbations) if labels is None else list(labels)
    ctrl_rows = panel.rows(panel.control)
    if control_counts is None:
        if ctrl_rows.size == 0:
            raise ValueError(f"no {panel.control!r} cells in the panel")
        control_counts = panel.counts[ctrl_rows]
    control_counts = np.asarray(control_counts, dtype=np.float64)
    profiles: Dict[str, np.ndarray] = {}
    corrections: Dict[str, float] = {}
    per_gene: Dict[str, np.ndarray] = {}
    sizes: Dict[str, int] = {}
    for lab in perts + [panel.control]:
        Y = control_counts if lab == panel.control else panel.counts[panel.rows(lab)]
        if Y.shape[0] == 0:
            raise ValueError(f"perturbation {lab!r} has no cells")
        q = jackknife_per_gene(Y)
        profiles[lab], per_gene[lab], corrections[lab], sizes[lab] = bulk_profile(Y), q, float(q.sum()), int(Y.shape[0])
    de = differential_expression(panel, control_counts, perts)
    return Prepared(panel, perts, profiles, corrections, sizes, de, resolve_targets(perts, panel.genes), control_counts,
                    per_gene)


# -- the six metrics ---------------------------------------------------------------------------------------------------------

def _midrank(row: np.ndarray, match: int) -> float:
    d = row[match]
    if np.isnan(d):
        n_less = int(np.count_nonzero(~np.isnan(row)))
        n_equal = int(row.size - n_less)
    else:
        n_less = int(np.count_nonzero(row < d))
        n_equal = int(np.count_nonzero(row == d))
    return n_less + (n_equal - 1) / 2.0


def perturbation_discrimination(pred: Prepared, real: Prepared) -> Dict[str, float]:
    """``pds``: 1 - (mid-rank of the true match among the cosine distances) / (n - 1), on the feature space with every
    panel target gene removed (``exclusion_scope="panel"``)."""
    perts = real.perturbations
    n = len(perts)
    keep = np.ones(real.panel.n_genes, dtype=bool)
    for col in real.target_index.values():
        keep[col] = False
    if not keep.any():
        raise ValueError("every measured gene is the target of a panel perturbation: the ranked feature space is empty")
    ctrl_r = real.profiles[real.panel.control]
    ctrl_p = pred.profiles[pred.panel.control]
    real_eff = np.stack([real.profiles[p] - ctrl_r for p in perts])[:, keep]
    pred_eff = np.stack([pred.profiles[p] - ctrl_p for p in perts])[:, keep]
    sim = pred_eff @ real_eff.T
    pn = np.sqrt(np.einsum("ig,ig->i", pred_eff, pred_eff))
    rn = np.sqrt(np.einsum("jg,jg->j", real_eff, real_eff))
    denom = pn[:, None] * rn[None, :]
    cosine = np.zeros_like(sim)
    nz = denom != 0
    cosine[nz] = sim[nz] / denom[nz]
    dist = np.clip(1.0 - cosine, 0.0, 2.0)
    D = n - 1
    out: Dict[str, float] = {}
    for i, p in enumerate(perts):
        rank = _midrank(dist[i], i)
        out[p] = 1.0 if D <= 0 else 1.0 - rank / D
    return out


def expression_error(pred: Prepared, real: Prepared) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, float]]:
    """
    ``mse``: per perturbation the corrected numerator ``N_p`` and denominator ``D_p`` (both over every gene but the
    perturbation's own, as ``(summed distance - corrections) / G_p``), the scored ratio of sums ``sum N_p / sum D_p``
    and a diagnostic of the reference: ``sum_den`` (the panel's corrected off-target effect) against ``sum_noise`` (the
    sampling corrections it subtracts, ``sum_p (C_p + C_ctrl) / G_p``). The prediction's correction is capped per row by
    the reference's and in total by the submission's own across-perturbation spread (cell-eval2 issues #247 and #348).

    One deliberate difference from the competition scorer: the perturbation's own gene is excluded from the three sampling
    corrections as well as from the distances. cell-eval2 leaves the corrections whole because its cached moments hold
    ``C`` as one scalar, an approximation worth 0.01 % of the denominator on an 18,533-gene panel but, on a panel of a few
    dozen genes where the knocked-down gene carries much of its own group's sampling noise, enough to make a replicate's
    error negative. The specification names the gene-excluded form as the exact one.
    """
    perts = real.perturbations
    G = real.panel.n_genes
    ctrl = real.panel.control
    b_ctrl = real.profiles[ctrl]
    cols = {k: real.target_index[p] for k, p in enumerate(perts) if p in real.target_index}
    n_row = np.array([G - 1 if k in cols else G for k in range(len(perts))], dtype=float)
    w = np.divide(1.0, n_row, out=np.zeros_like(n_row), where=n_row > 0)
    P = np.stack([pred.profiles[p] for p in perts])
    R = np.stack([real.profiles[p] for p in perts])
    C_pred = np.array([pred.corrections[p] for p in perts])
    C_real = np.array([real.corrections[p] for p in perts])
    C_ctrl = np.full(len(perts), real.corrections[ctrl])
    for k, col in cols.items():                                  # the own gene's share leaves every correction
        C_pred[k] -= pred.corrections_gene[perts[k]][col]
        C_real[k] -= real.corrections_gene[perts[k]][col]
        C_ctrl[k] -= real.corrections_gene[ctrl][col]
    term = np.minimum(C_pred, C_real)
    claim = float((term * w).sum())
    ratio = 1.0
    if claim > 0.0:
        budget = _across_perturbation_budget(P, cols, w)
        ratio = min(1.0, budget / claim)
    term = term * ratio
    d = P - R
    num = np.einsum("ij,ij->i", d, d) - term - C_real
    e = R - b_ctrl[None, :]
    den = np.einsum("ij,ij->i", e, e) - C_real - C_ctrl
    for k, col in cols.items():
        num[k] -= (P[k, col] - R[k, col]) ** 2
        den[k] -= (R[k, col] - b_ctrl[col]) ** 2
    num = np.divide(num, n_row, out=np.full(num.shape, np.nan), where=n_row > 0)
    den = np.divide(den, n_row, out=np.full(den.shape, np.nan), where=n_row > 0)
    noise = np.divide(C_real + C_ctrl, n_row, out=np.full(den.shape, np.nan), where=n_row > 0)
    total_den = float(np.nansum(den))
    ratio_of_sums = float(np.nansum(num) / total_den) if total_den != 0 and np.isfinite(total_den) else float("nan")
    diagnostic = {"sum_den": total_den, "sum_noise": float(np.nansum(noise))}
    return dict(zip(perts, num.tolist())), dict(zip(perts, den.tolist())), ratio_of_sums, diagnostic


def _across_perturbation_budget(P: np.ndarray, cols: Dict[int, int], w: np.ndarray) -> float:
    """``sum_g sum_p w_p (P[p, g] - weighted mean)^2`` over the rows that score gene g (each row's own target left out)."""
    n_rows, n_genes = P.shape
    dev = P - P.mean(axis=0)
    s1 = np.einsum("ij,i->j", dev, w)
    s2 = np.einsum("ij,ij,i->j", dev, dev, w)
    wsum = np.full(n_genes, float(w.sum()))
    cnt = np.full(n_genes, float(n_rows))
    for k, col in cols.items():
        v = dev[k, col]
        s1[col] -= w[k] * v
        s2[col] -= w[k] * v * v
        wsum[col] -= w[k]
        cnt[col] -= 1.0
    ok = (cnt >= 2.0) & (wsum > 0.0)
    if not ok.any():
        return 0.0
    ss = s2[ok] - s1[ok] ** 2 / wsum[ok]
    return float(np.maximum(ss, 0.0).sum())


def _defined(lfc: np.ndarray) -> np.ndarray:
    return np.isfinite(lfc) & (lfc != 0.0)


def _joined(pred: Prepared, real: Prepared, label: str) -> np.ndarray:
    """(G,) bool: genes carried by both DE tables for ``label``, the perturbation's own gene left out."""
    both = pred.de.kept & real.de.kept
    col = real.target_index.get(label)
    if col is not None:
        both = both.copy()
        both[col] = False
    return both


def _real_sig(real: Prepared, label: str) -> np.ndarray:
    sig = real.de.significant(label).copy()
    col = real.target_index.get(label)
    if col is not None:
        sig[col] = False
    return sig


def direction_fidelity(pred: Prepared, real: Prepared) -> Dict[str, float]:
    """``fid``: ``k / max(n_pred, n_real)`` with ``k`` the predicted significant, adjudicable genes whose sign agrees with
    the reference; 0/0 (nothing called on either side) is NaN and drops out of the mean."""
    out: Dict[str, float] = {}
    for p in real.perturbations:
        n_conf = int(_real_sig(real, p).sum())
        both = _joined(pred, real, p)
        in_denom = both & _defined(real.de.lfc[p])
        scored = in_denom & pred.de.significant(p)
        lp, lr = pred.de.lfc[p], real.de.lfc[p]
        match = scored & _defined(lp) & (np.sign(np.nan_to_num(lp)) == np.sign(np.nan_to_num(lr)))
        n_pred, k = int(scored.sum()), int(match.sum())
        denom = max(n_pred, n_conf)
        out[p] = k / denom if denom > 0 else float("nan")
    return out


def direction_reach(pred: Prepared, real: Prepared) -> Dict[str, float]:
    """``reach``: the deepest prefix of the reference-significant genes, ordered by the prediction's own confidence, whose
    directional purity stays at or above 0.9, as a fraction of the reference's budget; NaN when the budget is empty."""
    out: Dict[str, float] = {}
    for p in real.perturbations:
        real_sig = _real_sig(real, p)
        n_conf = int(real_sig.sum())
        if n_conf == 0:
            out[p] = float("nan")
            continue
        pool = np.flatnonzero(real_sig & _joined(pred, real, p))
        if pool.size == 0:
            out[p] = 0.0
            continue
        lp, lr = pred.de.lfc[p][pool], real.de.lfc[p][pool]
        q, pv = pred.de.p_adj[p][pool], pred.de.p_value[p][pool]
        sig_pred = np.isfinite(q) & (q < ALPHA)
        rank_q = np.where(np.isfinite(q), q, np.inf)
        rank_p = np.where(np.isfinite(pv), pv, np.inf)
        abs_l = np.where(np.isfinite(lp), np.abs(lp), -np.inf)
        names = np.array([real.panel.genes[g] for g in pool])
        order = np.lexsort((names, -abs_l, rank_p, rank_q, ~sig_pred))
        in_denom = _defined(lr)[order]
        match = (in_denom & _defined(lp)[order] & (np.sign(np.nan_to_num(lp))[order] == np.sign(np.nan_to_num(lr))[order]))
        depth = np.cumsum(in_denom)
        hits = np.cumsum(match)
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = np.where(depth > 0, hits / np.where(depth > 0, depth, 1), np.nan)
        ok = np.isfinite(purity) & (purity >= REACH_PURITY)
        k_star = int(depth[ok].max()) if ok.any() else 0
        out[p] = k_star / n_conf
    return out


def significance_jaccard(pred: Prepared, real: Prepared) -> Dict[str, float]:
    """``jac``: |R ∩ P| / |R ∪ P| of the significant sets (own gene excluded); an empty union is 1."""
    out: Dict[str, float] = {}
    for p in real.perturbations:
        r = _real_sig(real, p)
        s = pred.de.significant(p).copy()
        col = real.target_index.get(p)
        if col is not None:
            s[col] = False
        union = int((r | s).sum())
        out[p] = 1.0 if union == 0 else float((r & s).sum()) / union
    return out


def fold_change_nmae(pred: Prepared, real: Prepared) -> Dict[str, float]:
    """``nmae``: mean |lfc_pred - lfc_real| / mean |lfc_real| over the reference's significant, finite-fold-change genes
    (own gene excluded); perturbations with a gate below ``NMAE_MIN_GATE`` genes or a zero denominator are omitted, and a
    missing or non-finite predicted fold change reads as 0."""
    out: Dict[str, float] = {}
    for p in real.perturbations:
        gate = _real_sig(real, p) & np.isfinite(real.de.lfc[p])
        if int(gate.sum()) < NMAE_MIN_GATE:
            continue
        lr = real.de.lfc[p][gate]
        lp = pred.de.lfc[p][gate]
        lp = np.where(np.isfinite(lp), lp, 0.0)
        den = float(np.mean(np.abs(lr)))
        if not np.isfinite(den) or den == 0.0:
            continue
        out[p] = float(np.mean(np.abs(lp - lr))) / den
    return out


def nmae_replicate(full: Prepared, half_a: Prepared, half_b: Prepared) -> Dict[str, float]:
    """The replicate estimator of ``nmae``: the gate and the denominator from the full reference, the two fold-change
    vectors from the halves (each against its own control), so the cohort is the member's own."""
    out: Dict[str, float] = {}
    for p in full.perturbations:
        gate = _real_sig(full, p) & np.isfinite(full.de.lfc[p])
        if int(gate.sum()) < NMAE_MIN_GATE:
            continue
        lr = full.de.lfc[p][gate]
        den = float(np.mean(np.abs(lr)))
        if not np.isfinite(den) or den == 0.0:
            continue
        la = half_a.de.lfc[p][gate] if p in half_a.de.lfc else np.zeros(lr.size)
        lb = half_b.de.lfc[p][gate] if p in half_b.de.lfc else np.zeros(lr.size)
        la = np.where(np.isfinite(la), la, 0.0)
        lb = np.where(np.isfinite(lb), lb, 0.0)
        out[p] = float(np.mean(np.abs(la - lb))) / den
    return out


# -- a scored submission ------------------------------------------------------------------------------------------------------

def _mean(values: Dict[str, float]) -> float:
    v = np.array([x for x in values.values() if x is not None and np.isfinite(x)], dtype=float)
    return float(v.mean()) if v.size else float("nan")


def score_submission(real: Prepared, pred_panel: Panel, control_source: str = "real",
                     pred_prepared: Optional[Prepared] = None) -> Dict[str, Any]:
    """
    The six raw metrics of one submission against a prepared reference. ``control_source="real"`` (the competition rule)
    measures the predicted effects and the prediction's DE against the reference's control cells; ``"pred"`` (the
    replicate anchor) against the prediction's own (``pred_prepared`` passes that preparation in when the caller has it).

    Returns ``{"aggregate": {metric: value}, "per_perturbation": {metric: {label: value}}, "cohort": {metric: n},
    "components": {...}}``.
    """
    if control_source not in ("real", "pred"):
        raise ValueError("control_source must be 'real' or 'pred'")
    if pred_prepared is not None and control_source == "pred":
        pred = pred_prepared
    else:
        ctrl = real.control_counts if control_source == "real" else None
        pred = prepare(pred_panel, control_counts=ctrl, labels=real.perturbations)
    pds = perturbation_discrimination(pred, real)
    num, den, mse, mse_diag = expression_error(pred, real)
    fid = direction_fidelity(pred, real)
    reach = direction_reach(pred, real)
    jac = significance_jaccard(pred, real)
    nmae = fold_change_nmae(pred, real)
    per = {"pds": pds, "jac": jac, "nmae": nmae, "fid": fid, "reach": reach}
    aggregate = {k: _mean(v) for k, v in per.items()}
    aggregate["mse"] = mse
    cohort = {k: int(sum(1 for x in v.values() if np.isfinite(x))) for k, v in per.items()}
    cohort["mse"] = len(real.perturbations)
    return {"aggregate": {m: aggregate[m] for m in METRICS}, "per_perturbation": per, "cohort": cohort,
            "components": {"mse_numerator": num, "mse_denominator": den, **mse_diag}, "control_source": control_source}


# -- the context-mean baseline ---------------------------------------------------------------------------------------------

def context_mean_profile(real: Prepared) -> np.ndarray:
    """The mean per-perturbation count profile (per-cell means), each gene's own perturbation left out of its mean."""
    perts = real.perturbations
    panel = real.panel
    bulk = np.stack([panel.counts[panel.rows(p)].mean(axis=0) for p in perts])
    total = bulk.sum(axis=0)
    count = np.full(panel.n_genes, float(len(perts)))
    for i, p in enumerate(perts):
        col = real.target_index.get(p)
        if col is not None:
            total[col] -= bulk[i, col]
            count[col] -= 1.0
    if np.any(count <= 0):
        raise ValueError("every perturbation targets some gene, so its context mean is undefined; the panel needs more "
                         "perturbations than genes they target")
    return total / count


def emit_cells(control_counts: np.ndarray, fold: np.ndarray, n_cells: int, rng: np.random.Generator,
               integer: bool = True, rescale: bool = True) -> np.ndarray:
    """
    Cells of a predicted mean profile: control cells drawn with replacement, every gene multiplied by ``fold`` (the
    predicted mean over the control mean). With ``rescale`` each cell is then brought back to the library size of the
    control cell it came from (every metric reads per-cell or per-group normalised values, so the composition is what
    counts, and the rescaling keeps a prediction under the per-cell count cap whatever the folds). With ``integer`` the
    values are rounded stochastically (floor plus a Bernoulli draw of the fraction), which keeps the expected count and
    makes a valid submission; without it they stay fractional, as cell-eval2's own baseline arm emits them.
    """
    ctrl = np.asarray(control_counts, dtype=np.float64)
    if ctrl.shape[0] == 0:
        raise ValueError("cannot emit cells from an empty control pool")
    fold = np.asarray(fold, dtype=np.float64)
    if fold.shape != (ctrl.shape[1],):
        raise ValueError(f"fold must have one entry per gene ({ctrl.shape[1]}), got {fold.shape}")
    if not np.all(np.isfinite(fold)) or np.any(fold < 0):
        raise ValueError("fold changes must be finite and non-negative")
    draws = rng.integers(0, ctrl.shape[0], size=int(n_cells))
    source = ctrl[draws]
    X = source * fold[None, :]
    if rescale:
        lib_src, lib_new = source.sum(axis=1), X.sum(axis=1)
        scale = np.where(lib_new > 0, lib_src / np.where(lib_new > 0, lib_new, 1.0), 1.0)
        X = X * scale[:, None]
    if integer:
        base = np.floor(X)
        X = base + (rng.uniform(size=X.shape) < (X - base))
    return X


def baseline_prediction(real: Prepared, seed: int = BASE_SEED) -> Panel:
    """
    The context-mean arm, as cell-eval2's supported ``emit="dispersed"`` baseline builds it: the profile of
    :func:`context_mean_profile` for every perturbation, emitted as control cells drawn with replacement (the reference's
    group sizes) and rescaled gene by gene to the profile (fractional values, no library rescaling).
    """
    panel = real.panel
    ctrl_pool = panel.counts[panel.rows(panel.control)]
    ctrl_mean = ctrl_pool.mean(axis=0)
    profile = context_mean_profile(real)
    fold = np.where(ctrl_mean > 0, profile / np.where(ctrl_mean > 0, ctrl_mean, 1.0), 0.0)
    rng = np.random.default_rng(seed)
    blocks, labels = [], []
    for p in real.perturbations:
        n = real.sizes[p]
        blocks.append(emit_cells(ctrl_pool, fold, n, rng, integer=False, rescale=False))
        labels += [p] * n
    return Panel(np.vstack(blocks), np.array(labels), list(panel.genes), panel.control)


# -- the replicate anchor ----------------------------------------------------------------------------------------------------

def split_seeds(base_seed: int = BASE_SEED, n_splits: int = N_SPLITS) -> List[int]:
    return [int(s) for s in np.random.SeedSequence(int(base_seed)).generate_state(int(n_splits))]


def disjoint_halves(panel: Panel, seed: int) -> Tuple[Panel, Panel]:
    """Two disjoint halves of ``floor(n/2)`` cells per group (control included); groups with one cell are dropped."""
    rng = np.random.default_rng(seed)
    a_idx, b_idx = [], []
    for lab in sorted(set(panel.labels.tolist())):
        perm = rng.permutation(panel.rows(lab))
        h = perm.size // 2
        if h < 1:
            continue
        a_idx.append(perm[:h])
        b_idx.append(perm[h:2 * h])
    if not a_idx:
        raise ValueError("no group has two cells to split")
    a, b = np.concatenate(a_idx), np.concatenate(b_idx)
    return (Panel(panel.counts[a], panel.labels[a], list(panel.genes), panel.control),
            Panel(panel.counts[b], panel.labels[b], list(panel.genes), panel.control))


def replicate_anchor(real: Prepared, base_seed: int = BASE_SEED, n_splits: int = N_SPLITS) -> Dict[str, Any]:
    """
    The reference scored against itself: per split half B (as the prediction, with its own control) against half A,
    averaged over the splits. ``nmae`` uses its own estimator (:func:`nmae_replicate`). Returns
    ``{"replicate": {metric: mean}, "splits": [{metric: value}, ...], "seeds": [...]}``.
    """
    seeds = split_seeds(base_seed, n_splits)
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        half_a, half_b = disjoint_halves(real.panel, seed)
        perts = [p for p in real.perturbations if p in half_a.perturbations and p in half_b.perturbations]
        prep_a = prepare(half_a, labels=perts)
        prep_b = prepare(half_b, labels=perts)
        scored = score_submission(prep_a, half_b, control_source="pred", pred_prepared=prep_b)
        vals = dict(scored["aggregate"])
        vals["nmae"] = _mean(nmae_replicate(real, prep_a, prep_b))
        rows.append(vals)
    rep = {m: _mean({i: r[m] for i, r in enumerate(rows)}) for m in METRICS}
    return {"replicate": rep, "splits": rows, "seeds": seeds}


# -- scaling and the competition score -----------------------------------------------------------------------------------

def scale_member(metric: str, u: float, b: float, r: float) -> Tuple[Optional[float], str]:
    """``(u - b) / (r - b)`` with the member's clamps; a missing ``u`` takes the score of the member's structural worst
    value (its floor). ``(None, reason)`` when the anchors leave no scale."""
    if b is None or r is None or not (np.isfinite(b) and np.isfinite(r)):
        return None, "the baseline or the replicate anchor is undefined on this panel"
    span = r - b
    if span == 0 or (metric in LOWER_IS_BETTER and span > 0) or (metric not in LOWER_IS_BETTER and span < 0):
        return None, f"degenerate scale: baseline {b:.4g} and replicate {r:.4g}"
    low, high = CLAMP.get(metric, (None, None))
    if u is None or not np.isfinite(u):
        worst = WORST_RAW[metric]
        if worst is None:                                   # unbounded error metric: the floor itself
            return low, "no perturbation could be scored; the floor is reported"
        s = (worst - b) / span
    else:
        s = (u - b) / span
    if low is not None:
        s = max(s, low)
    if high is not None:
        s = min(s, high)
    return float(s), ""


def resolved_members(baseline: Dict[str, float], replicate: Dict[str, float], replicate_sd: Dict[str, float],
                     mse_effect: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """
    Which members the panel can score, with the reason for each that it cannot. The competition's panels (300
    perturbations, 18,533 genes) always resolve every scale; a simulated panel of a few perturbations may not:

    * a member whose replicate is not separated from the baseline by more than twice the replicate's own spread over the
      five splits has no scale (the two anchors are within noise of each other);
    * the expression error needs the reference's corrected off-target effect to exceed the sampling corrections it
      subtracts (``sum_den > sum_noise``): below that the ratio of sums is a ratio of two noise terms.
    """
    out: Dict[str, str] = {}
    for m in METRICS:
        b, r, sd = baseline.get(m), replicate.get(m), replicate_sd.get(m)
        if b is None or r is None or not (np.isfinite(b) and np.isfinite(r)):
            out[m] = "the baseline or the replicate anchor is undefined on this panel"
            continue
        if sd is not None and np.isfinite(sd) and abs(r - b) <= 2.0 * sd:
            out[m] = (f"scale not resolved: the replicate ({r:.3g}, spread {sd:.2g} over {N_SPLITS} splits) is within "
                      f"noise of the baseline ({b:.3g})")
            continue
        if m == "mse" and mse_effect is not None:
            eff, noise = mse_effect.get("sum_den"), mse_effect.get("sum_noise")
            if eff is None or noise is None or not np.isfinite(eff) or not eff > noise:
                out[m] = (f"not resolved: the reference's off-target effect ({eff:.3g}) does not exceed its sampling "
                          f"noise ({noise:.3g}) on this panel")
                continue
        out[m] = ""
    return out


def competition_score(scored: Dict[str, Any], baseline: Dict[str, float], replicate: Dict[str, float],
                      unresolved: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Scaled members and the overall score of one scored submission against the two anchors. ``unresolved`` (from
    :func:`resolved_members`) names members the panel cannot scale; they keep their raw value, get no scaled value, and
    the overall score is the mean over the members that are scaled (``n_scaled`` says how many of the six).
    """
    members: Dict[str, Dict[str, Any]] = {}
    scaled: List[float] = []
    for m in METRICS:
        u = scored["aggregate"].get(m)
        blocked = (unresolved or {}).get(m, "")
        if blocked:
            s, reason = None, blocked
        else:
            s, reason = scale_member(m, u, baseline.get(m), replicate.get(m))
        members[m] = {"raw": None if u is None or not np.isfinite(u) else float(u), "scaled": s, "reason": reason,
                      "n": scored["cohort"].get(m)}
        if s is not None:
            scaled.append(s)
    overall = float(np.mean(scaled)) if scaled else None
    return {"overall": overall, "members": members, "n_scaled": len(scaled)}


class Scorer:
    """A reference panel with its baseline and replicate anchors computed once; ``score`` places any submission on the
    competition scale."""

    def __init__(self, real: Panel, seed: int = BASE_SEED):
        if not real.perturbations:
            raise ValueError("the reference has no perturbation")
        if real.rows(real.control).size < 2:
            raise ValueError("the reference needs at least two control cells")
        self.real = real
        self.prepared = prepare(real)
        self.baseline_arm = score_submission(self.prepared, baseline_prediction(self.prepared, seed), control_source="real")
        self.anchor = replicate_anchor(self.prepared, base_seed=seed)
        self.baseline = self.baseline_arm["aggregate"]
        self.replicate = self.anchor["replicate"]
        self.replicate_sd = {m: float(np.std([r[m] for r in self.anchor["splits"] if np.isfinite(r[m])]))
                             if any(np.isfinite(r[m]) for r in self.anchor["splits"]) else float("nan") for m in METRICS}
        self.mse_effect = {k: self.baseline_arm["components"][k] for k in ("sum_den", "sum_noise")}
        self.unresolved = resolved_members(self.baseline, self.replicate, self.replicate_sd, self.mse_effect)

    def score(self, pred: Panel, validate: bool = True, allow_fractional: bool = False) -> Dict[str, Any]:
        if validate:
            validate_submission(pred, self.real, allow_fractional=allow_fractional)
        scored = score_submission(self.prepared, pred, control_source="real")
        out = competition_score(scored, self.baseline, self.replicate, self.unresolved)
        out["cohort"] = scored["cohort"]
        out["per_perturbation"] = {m: {k: (None if not np.isfinite(v) else float(v)) for k, v in d.items()}
                                   for m, d in scored["per_perturbation"].items()}
        return out

    def anchors(self) -> Dict[str, Any]:
        """The two reference points per member, as the leaderboard documents them."""
        return {"baseline": {m: (None if not np.isfinite(self.baseline[m]) else float(self.baseline[m])) for m in METRICS},
                "replicate": {m: (None if not np.isfinite(self.replicate[m]) else float(self.replicate[m])) for m in METRICS},
                "replicate_sd": {m: (None if not np.isfinite(self.replicate_sd[m]) else float(self.replicate_sd[m])) for m in METRICS},
                "replicate_splits": [{m: (None if not np.isfinite(r[m]) else float(r[m])) for m in METRICS} for r in self.anchor["splits"]],
                "unresolved": {m: r for m, r in self.unresolved.items() if r},
                "mse_effect": {k: (None if not np.isfinite(x) else float(x)) for k, x in self.mse_effect.items()},
                "n_perturbations": len(self.prepared.perturbations), "n_genes": self.real.n_genes,
                "cells_per_perturbation": {p: self.prepared.sizes[p] for p in self.prepared.perturbations},
                "n_control_cells": int(self.real.rows(self.real.control).size),
                "genes_tested": int(self.prepared.de.kept.sum()),
                "cohort_baseline": self.baseline_arm["cohort"]}
