"""
Stage 4 of the Perturb-seq generator: realism statistics and targets from genome-scale Perturb-seq.

Two families of statistics are computed and compared with published values:

- **ground truth** (:func:`truth_statistics`, from a noise-free :class:`ConditionScreen`): how many perturbations
  move other genes at all, how dense the effect matrix is, how much of it is mediated rather than direct, how
  effects are distributed over perturbations;
- **observed** (:func:`observed_statistics`, from a measured AnnData as an analyst would see it): on-target knockdown,
  perturbations with a detectable transcriptional phenotype on genes other than the target (test against
  non-targeting cells, Benjamini-Hochberg), differentially expressed genes per perturbation (Welch t-test on log-normalised
  pseudobulk-free single-cell values, BH), the false-positive rate of non-targeting guide groups tested the same way,
  and count-level statistics (UMIs and detected genes per cell).

:func:`realism_report` lines the simulated values up against :data:`TARGETS`. The targets are the values the GRN-Sim
research report collected (Replogle et al. 2022 Cell; Aguirre et al. 2025 PLoS Comput Biol; Nadig et al. 2025 Nat
Genet; Brown et al. 2023/2025 inspre); they are ranges to approach, not a proven sufficient set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grn_sim.perturbseq.conditions import ConditionScreen


@dataclass(frozen=True)
class Target:
    low: float
    high: float
    source: str
    note: str = ""


# Ranges to match (fractions unless noted). Widths reflect the spread between studies / tests.
TARGETS: Dict[str, Target] = {
    "on_target_knockdown": Target(0.80, 0.95, "Replogle 2022 (median 85.5% K562, 91.6% RPE1)",
                                  "median fraction of the target's expression removed (CRISPRi)"),
    "fraction_with_phenotype": Target(0.25, 0.45, "Replogle 2022 (31.1%); Aguirre 2025 (41%, energy test)",
                                      "perturbations with a detectable transcriptional phenotype"),
    "affected_pair_density": Target(0.015, 0.05, "Aguirre 2025 (3.1% of ordered gene pairs)",
                                    "ordered (perturbation, gene) pairs with an effect"),
    "bidirectional_share": Target(0.01, 0.06, "Aguirre 2025 (2.4% of affected pairs)",
                                  "affected pairs whose reverse pair is affected too"),
    "fraction_effects_indirect": Target(0.90, 0.995, "Aguirre 2025 (~98.5% mediated); inspre (1:13 direct:total)",
                                        "affected pairs that are not direct regulatory edges"),
    "top5pct_perturbation_effect_share": Target(0.25, 0.70, "Nadig 2025 (heavy-tailed effects: ~45 typical, >500 for "
                                                "essential genes)", "share of all effects caused by the top 5% "
                                                "perturbations"),
    "control_false_positive_rate": Target(0.0, 0.05, "Replogle 2022 (~2% for non-targeting guides)",
                                          "non-targeting guide groups called as having a phenotype"),
}


# Statistics whose genome-scale targets assume that the perturbed genes are *all* genes (or a random sample): under a partial
# screen with effective-gene selection they are shifted by the selection itself and are reported but not scored.
NOT_COMPARABLE_UNDER_PARTIAL = frozenset({"fraction_with_phenotype", "affected_pair_density", "bidirectional_share",
                                          "top5pct_perturbation_effect_share"})


# --- ground-truth statistics ------------------------------------------------------------------------

def truth_statistics(screen: ConditionScreen, effect_threshold: Optional[float] = None) -> Dict[str, float]:
    """Perturbation-level statistics of the noise-free screen (single perturbations only)."""
    thr = screen.effect_threshold if effect_threshold is None else float(effect_threshold)
    singles = [k for k, c in enumerate(screen.conditions) if len(c) == 1]
    genes = [screen.conditions[k][0] for k in singles]
    lfc = screen.total_effects[:, singles]
    n = screen.n_genes
    affected = np.abs(lfc) >= thr
    for col, g in enumerate(genes):
        affected[g, col] = False                     # the perturbed gene itself is not an effect
    direct = np.zeros_like(affected)
    for col, g in enumerate(genes):
        direct[:, col] = screen.adjacency[g, :] != 0
    n_aff = int(affected.sum())
    per_pert = affected.sum(axis=0)
    # pair (perturbed g -> gene i); bidirectional if i is also perturbed and moves g
    pos = {int(g): col for col, g in enumerate(genes)}
    bidir = 0
    for col, g in enumerate(genes):
        for i in np.flatnonzero(affected[:, col]).tolist():
            if i in pos and bool(affected[int(g), pos[i]]):
                bidir += 1
    top = max(1, int(np.ceil(0.05 * len(genes))))
    own = np.array([screen.total_effects[g, k] for g, k in zip(genes, singles)])
    knockdown = 1.0 - np.power(2.0, own) if screen.mode != "input" else np.full(len(genes), np.nan)
    return {
        "n_perturbations": float(len(genes)),
        "fraction_with_effect_truth": float(np.mean(per_pert > 0)) if genes else 0.0,
        "affected_pair_density": float(n_aff / max(len(genes) * (n - 1), 1)),
        "bidirectional_share": float(bidir / n_aff) if n_aff else 0.0,
        "fraction_effects_indirect": float((affected & ~direct).sum() / n_aff) if n_aff else 0.0,
        "direct_to_total_ratio": float((affected & direct).sum() / n_aff) if n_aff else 0.0,
        "top5pct_perturbation_effect_share": float(np.sort(per_pert)[::-1][:top].sum() / n_aff) if n_aff else 0.0,
        "median_genes_affected_per_perturbation": float(np.median(per_pert)) if genes else 0.0,
        "truth_on_target_knockdown": float(np.nanmedian(knockdown)) if genes else float("nan"),
        "silent_edge_fraction": float(np.mean(np.abs(screen.elasticity[screen.adjacency != 0]) < 0.05))
        if np.any(screen.adjacency != 0) else 0.0,
    }


# --- observed statistics ----------------------------------------------------------------------------

def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def _energy_pvalue(D: np.ndarray, n_a: int, rng: np.random.Generator, n_perm: int) -> float:
    """Permutation p-value of the energy distance between the first n_a points and the rest (distance matrix D)."""
    n = D.shape[0]

    def stat(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        return 2.0 * D[np.ix_(idx_a, idx_b)].mean() - D[np.ix_(idx_a, idx_a)].mean() - D[np.ix_(idx_b, idx_b)].mean()

    allidx = np.arange(n)
    obs = stat(allidx[:n_a], allidx[n_a:])
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if stat(perm[:n_a], perm[n_a:]) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def _hotelling_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Hotelling T^2 test (pooled covariance) of mean equality; F-distribution p-value."""
    from scipy import stats as sps

    n1, n2, p = a.shape[0], b.shape[0], a.shape[1]
    if n1 + n2 - p - 1 <= 0:
        return 1.0
    diff = a.mean(axis=0) - b.mean(axis=0)
    S = ((n1 - 1) * np.cov(a, rowvar=False) + (n2 - 1) * np.cov(b, rowvar=False)) / (n1 + n2 - 2)
    S = np.atleast_2d(S) + 1e-9 * np.eye(p)
    t2 = float(n1 * n2 / (n1 + n2) * diff @ np.linalg.solve(S, diff))
    f = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    return float(sps.f.sf(f, p, n1 + n2 - p - 1))


def _log_normalise(X: Any) -> np.ndarray:
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    dense = dense.astype(float)
    totals = dense.sum(axis=1, keepdims=True)
    return np.log1p(np.divide(dense, totals, out=np.zeros_like(dense), where=totals > 0) * 1e4)


def observed_statistics(
    adata: Any,
    n_pcs: int = 20,
    n_perm: int = 200,
    max_control_cells: int = 400,
    fdr: float = 0.05,
    use_assigned_guides: bool = True,
    phenotype_test: str = "hotelling",
    seed: int = 0,
) -> Dict[str, float]:
    """
    Statistics an analyst would compute from the measured screen (an AnnData from ``population_to_anndata``).

    Cells are grouped by their *assigned* guide when guide calls exist (unassigned / multiple / doublet-flagged
    cells dropped, as a pipeline would), else by the true condition. Only single perturbations are tested.

    ``phenotype_test``: ``"hotelling"`` (default; two-sample Hotelling T^2 on the top ``n_pcs`` principal components,
    exact F p-values) or ``"energy"`` (energy-distance permutation test with ``n_perm`` permutations; its smallest
    p-value is 1/(n_perm+1), which Benjamini-Hochberg over hundreds of perturbations cannot call unless many
    perturbations reach it - so few strong phenotypes are missed unless ``n_perm`` is large).
    """
    if phenotype_test not in ("hotelling", "energy"):
        raise ValueError(f"phenotype_test must be 'hotelling' or 'energy', got {phenotype_test!r}")
    from scipy import stats as sps

    rng = np.random.default_rng(seed)
    obs = adata.obs
    genes = list(adata.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    L = _log_normalise(adata.X)

    if use_assigned_guides and "assignment_status" in obs.columns:
        keep = np.array([str(v) == "single" for v in obs["assignment_status"].tolist()], dtype=bool)
        # plain Python strings: pandas/numpy 2.x may hand back object or StringDtype arrays that np.char rejects
        guide = np.array([str(g) for g in obs["assigned_guide"].tolist()], dtype=object)
        target = np.array([g.rsplit("_g", 1)[0] if not g.startswith("non-targeting") else "control" for g in guide],
                          dtype=object)
        nt_guide = np.array([g if g.startswith("non-targeting") else "" for g in guide], dtype=object)
    else:
        keep = np.ones(adata.n_obs, dtype=bool)
        target = np.array(["control" if str(c) == "control" else str(c) for c in obs["condition"].tolist()], dtype=object)
        nt_guide = (np.array([str(g) for g in obs["guide"].tolist()], dtype=object) if "guide" in obs.columns
                    else np.full(adata.n_obs, "", dtype=object))
        nt_guide = np.where(target == "control", nt_guide, "")
    singles = sorted({t for t in target[keep] if t != "control" and "+" not in t})
    ctrl = np.flatnonzero(keep & (target == "control"))
    if ctrl.size < 10 or not singles:
        raise ValueError("need >= 10 control cells and at least one single perturbation")
    ctrl_sub = rng.choice(ctrl, size=min(max_control_cells, ctrl.size), replace=False)

    # PCA on control-centred data for the energy test.
    centre = L[ctrl].mean(axis=0)
    sd = L[ctrl].std(axis=0) + 1e-6
    Z = (L - centre) / sd
    sample = rng.choice(np.flatnonzero(keep), size=min(5000, int(keep.sum())), replace=False)
    _, _, vt = np.linalg.svd(Z[sample] - Z[sample].mean(axis=0), full_matrices=False)
    P = Z @ vt[:n_pcs].T

    V = vt[:n_pcs]

    def test_group(cells: np.ndarray, background: np.ndarray, exclude: Optional[int] = None) -> Tuple[float, int]:
        """Energy-test p-value and DEG count; ``exclude`` (the perturbed gene) is left out of both, so only effects
        on *other* genes count as a transcriptional phenotype (the definition behind the 31-41% targets)."""
        idx = np.concatenate([cells, background])
        pts = P[idx]
        if exclude is not None:
            pts = pts - np.outer(Z[idx, exclude], V[:, exclude])
        if phenotype_test == "hotelling":
            p_energy = _hotelling_pvalue(pts[:cells.size], pts[cells.size:])
        else:
            D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            p_energy = _energy_pvalue(D, cells.size, rng, n_perm)
        _, p = sps.ttest_ind(L[cells], L[background], axis=0, equal_var=False)
        p = np.nan_to_num(p, nan=1.0)
        if exclude is not None:
            p = np.delete(p, exclude)
        return p_energy, int((_bh(p) < fdr).sum())

    p_energy, n_deg, knockdown = [], [], []
    for t in singles:
        cells = np.flatnonzero(keep & (target == t))
        if cells.size < 5:
            continue
        pe, nd = test_group(cells, ctrl_sub, exclude=gidx.get(t))
        p_energy.append(pe)
        n_deg.append(nd)
        if t in gidx:
            g = gidx[t]
            base = np.expm1(L[ctrl, g]).mean()
            if base > 0:
                knockdown.append(1.0 - np.expm1(L[cells, g]).mean() / base)
    called = _bh(np.asarray(p_energy)) < fdr if p_energy else np.zeros(0, dtype=bool)

    # False positives: each non-targeting guide's cells against the other control cells.
    fp = []
    for g in sorted({x for x in nt_guide[ctrl] if x}):
        mine = ctrl[nt_guide[ctrl] == g]
        rest = np.setdiff1d(ctrl_sub, mine)
        if mine.size >= 5 and rest.size >= 10:
            fp.append(test_group(mine, rest)[0])
    fp_called = _bh(np.asarray(fp)) < fdr if fp else np.zeros(0, dtype=bool)

    umis = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    n_deg_arr = np.asarray(n_deg)
    top = max(1, int(np.ceil(0.05 * n_deg_arr.size))) if n_deg_arr.size else 1
    return {
        "n_tested_perturbations": float(len(p_energy)),
        "fraction_with_phenotype": float(called.mean()) if called.size else 0.0,
        "median_degs_per_perturbation": float(np.median(n_deg_arr)) if n_deg_arr.size else 0.0,
        "observed_deg_pair_density": float(int(n_deg_arr.sum()) / max(n_deg_arr.size * (len(genes) - 1), 1)),
        "observed_top5pct_deg_share": float(int(np.sort(n_deg_arr)[::-1][:top].sum()) / max(int(n_deg_arr.sum()), 1)),
        "on_target_knockdown": float(np.median(knockdown)) if knockdown else float("nan"),
        "control_false_positive_rate": float(fp_called.mean()) if fp_called.size else float("nan"),
        "n_control_guides_tested": float(len(fp)),
        "median_umis_per_cell": float(np.median(umis)),
        "median_genes_detected_per_cell": float(np.median(detected)),
        "cells_kept_fraction": float(keep.mean()),
    }


# --- report -----------------------------------------------------------------------------------------

def realism_report(statistics: Dict[str, float], targets: Optional[Dict[str, Target]] = None,
                   partial_selection: bool = False) -> List[Dict[str, Any]]:
    """
    One row per target statistic present in ``statistics``: value, target range, whether it is inside. With
    ``partial_selection`` (a partial screen whose perturbed genes were chosen by effect) the statistics in
    :data:`NOT_COMPARABLE_UNDER_PARTIAL` get ``comparable = False`` and do not count towards the score.
    """
    targets = TARGETS if targets is None else targets
    rows = []
    for name, t in targets.items():
        if name not in statistics:
            continue
        v = float(statistics[name])
        inside = bool(np.isfinite(v) and t.low <= v <= t.high)
        distance = 0.0 if inside else (float("inf") if not np.isfinite(v)
                                        else min(abs(v - t.low), abs(v - t.high)) / max(t.high - t.low, 1e-9))
        comparable = not (partial_selection and name in NOT_COMPARABLE_UNDER_PARTIAL)
        rows.append({"statistic": name, "value": v, "target_low": t.low, "target_high": t.high, "within": inside,
                     "distance": distance, "source": t.source, "note": t.note, "comparable": comparable})
    return rows


def realism_score(rows: Sequence[Dict[str, Any]]) -> float:
    """Sum of range-normalised distances outside the targets (0 = every statistic inside its range)."""
    return float(sum(min(r["distance"], 10.0) for r in rows if r.get("comparable", True)))


def format_report(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [f"{'statistic':38s} {'simulated':>10s}  {'target':>15s}  ok"]
    for r in rows:
        flag = "n/a (selected genes)" if not r.get("comparable", True) else ("yes" if r["within"] else "NO")
        lines.append(f"{r['statistic']:38s} {r['value']:10.3f}  {r['target_low']:6.3f}-{r['target_high']:<7.3f}  {flag}")
    return "\n".join(lines)
