"""
Average causal effects (ACE) of single perturbations with standard errors, calibrated p-values and edge FDR.

This is the step in front of every perturbation-based network method (inspre, mean difference, DOTEARS, ...): it turns
the cells of a screen into a node-by-gene matrix of effects ``lfc[a, j]`` (the shift of gene j when node a is perturbed),
how certain each entry is, and the normalised matrix ``R[a, j] = lfc[a, j] / lfc[a, a]`` that inspre inverts.

Calibration follows the idea of SCEPTRE (Barry et al., Genome Biol. 2021) in a cheaper form: the null distribution of the
test statistic is measured instead of assumed. Groups of non-targeting cells of the size of the real perturbations are
drawn as pseudo-perturbations and compared with the remaining controls; the spread of their Welch z-scores, gene by gene,
sets the scale against which real z-scores are judged. It is *not* SCEPTRE (no negative-binomial regression, no
conditional resampling); it corrects the main failure, a miscalibrated test on skewed single-cell counts, and says so in
its name: ``method="welch_nt_calibrated"``.

Input: an AnnData as produced by :func:`grn_sim.perturbseq.export.population_to_anndata` (``obs['perturbation']`` or
the called guide, control cells labelled ``non-targeting``) or any AnnData with those columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.special import ndtr


@dataclass
class ACEEstimate:
    gene_names: List[str]
    nodes: np.ndarray            # (p,) gene index of every perturbed node
    lfc: np.ndarray              # (p, n) shift in log-normalised expression versus non-targeting cells
    se: np.ndarray               # (p, n) standard error of lfc
    z: np.ndarray                # (p, n) calibrated z-score (lfc / (se * null scale of the gene))
    pvalue: np.ndarray           # (p, n) two-sided p-value from the calibrated z
    qvalue: np.ndarray           # (p, n) Benjamini-Hochberg over all off-target pairs
    n_cells: np.ndarray          # (p,) cells per node
    n_control: int
    null_scale: np.ndarray       # (n,) spread of the pseudo-perturbation z-scores per gene (1 = well calibrated)
    method: str = "welch_nt_calibrated"
    cov: Optional[np.ndarray] = None   # (n, n) pooled within-condition covariance of the log-normalised expression
    n_cov: int = 0                     # cells behind ``cov``

    @property
    def own(self) -> np.ndarray:
        """(p,) effect of each node on its own gene (negative for knockdown)."""
        return self.lfc[np.arange(len(self.nodes)), self.nodes]

    @property
    def own_z(self) -> np.ndarray:
        return self.z[np.arange(len(self.nodes)), self.nodes]

    def normalised(self, min_own_z: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        ``(R, weights)``: lfc divided by each node's own effect, and weights in [0, 1] that fall to 0 for nodes without
        a detectable on-target effect (``|own z| < min_own_z``) and grow with the strength of the knockdown.
        """
        own = self.own
        ok = np.abs(self.own_z) >= min_own_z
        R = np.zeros_like(self.lfc)
        R[ok] = self.lfc[ok] / own[ok, None]
        w = np.zeros(len(self.nodes))
        zc = np.abs(self.own_z[ok])
        w[ok] = np.minimum(zc / max(float(np.median(zc)) if zc.size else 1.0, 1e-9), 1.0)
        return R, w

    def to_xu(self) -> Tuple[np.ndarray, np.ndarray]:
        """The repository's (X, U) pair: X[:, c] the response to node c, U with the node's own response on the diagonal."""
        n = self.lfc.shape[1]
        p = len(self.nodes)
        X = self.lfc.T.copy()                                   # (n, p)
        U = np.zeros((n, p))
        U[self.nodes, np.arange(p)] = self.own
        return X, U

    def denoised(self, alpha: float = 0.05) -> "ACEEstimate":
        """Copy with every off-target effect that is not significant at BH level ``alpha`` set to 0 (own effect kept)."""
        keep = self.qvalue <= alpha
        keep[np.arange(len(self.nodes)), self.nodes] = True
        return ACEEstimate(self.gene_names, self.nodes, np.where(keep, self.lfc, 0.0), self.se, self.z, self.pvalue,
                           self.qvalue, self.n_cells, self.n_control, self.null_scale, self.method, self.cov, self.n_cov)

    def edge_scores(self) -> np.ndarray:
        """(n, n) |lfc| indexed [regulator gene, target gene]; rows of genes without a perturbation stay 0."""
        n = self.lfc.shape[1]
        S = np.zeros((n, n))
        S[self.nodes] = np.abs(self.lfc)
        np.fill_diagonal(S, 0.0)
        return S

    def edge_mask(self, alpha: float = 0.05) -> np.ndarray:
        """(n, n) boolean [regulator, target]: effect significant at BH level ``alpha`` (diagonal excluded)."""
        n = self.lfc.shape[1]
        M = np.zeros((n, n), dtype=bool)
        M[self.nodes] = self.qvalue <= alpha
        np.fill_diagonal(M, False)
        return M


def log_normalise(X: Any, target_sum: float = 1e4) -> np.ndarray:
    """log1p of counts per ``target_sum`` (dense float32)."""
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
        tot = np.asarray(X.sum(axis=1)).ravel()
        scale = target_sum / np.maximum(tot, 1.0)
        X = sparse.diags(scale.astype(np.float32)) @ X
        return np.log1p(X.toarray())
    X = np.asarray(X, dtype=np.float32)
    tot = X.sum(axis=1, keepdims=True)
    return np.log1p(X * (target_sum / np.maximum(tot, 1.0)))


def _group_mean_var(Y: np.ndarray, labels: np.ndarray, n_groups: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onehot = sparse.csr_matrix((np.ones(len(labels), dtype=np.float32), (labels, np.arange(len(labels)))),
                               shape=(n_groups, len(labels)))
    cnt = np.asarray(onehot.sum(axis=1)).ravel()
    s1 = onehot @ Y
    s2 = onehot @ (Y.astype(np.float32) ** 2)
    c = np.maximum(cnt, 1)[:, None]
    mean = s1 / c
    var = np.maximum(s2 / c - mean ** 2, 0.0) * (c / np.maximum(c - 1, 1))
    return mean.astype(np.float64), var.astype(np.float64), cnt


def _bh(p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    q = np.ones_like(p)
    vals = p[mask]
    order = np.argsort(vals)
    ranked = vals[order] * len(vals) / (np.arange(len(vals)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(vals)
    out[order] = np.minimum(ranked, 1.0)
    q[mask] = out
    return q


def estimate_ace(
    adata: Any,
    perturbation_key: str = "perturbation",
    control_label: str = "non-targeting",
    gene_names: Optional[List[str]] = None,
    min_cells: int = 20,
    n_null_groups: int = 100,
    use_called_guide: bool = False,
    seed: int = 0,
) -> ACEEstimate:
    """
    Effects of every single-gene perturbation in ``adata`` against its non-targeting cells.

    ``use_called_guide`` takes the perturbation from the guide the pipeline *called* (``assigned_guide``, only cells with
    ``assignment_status == 'single'``; guide ``GENE_g1`` -> gene ``GENE``) instead of the true label, which is how a real
    analysis sees the data. Double perturbations (names containing ``+``) are skipped.
    """
    rng = np.random.default_rng(seed)
    names = list(gene_names) if gene_names is not None else [str(g) for g in adata.var_names]
    index = {g: i for i, g in enumerate(names)}
    obs = adata.obs
    if use_called_guide and "assigned_guide" in obs:
        guide = obs["assigned_guide"].astype(str).values
        status = obs["assignment_status"].astype(str).values
        lab = np.array([g.rsplit("_g", 1)[0] if (s == "single" and "+" not in g) else "" for g, s in zip(guide, status)],
                       dtype=object)
        ctrl = np.array([(s == "single" and g.rsplit("_g", 1)[0] not in index) for g, s in zip(guide, status)])
        lab[ctrl] = control_label
    else:
        lab = obs[perturbation_key].astype(str).values.astype(object)
    is_ctrl = lab == control_label
    Y = log_normalise(adata.X)
    n = Y.shape[1]
    labels_present, inv = np.unique(lab[~is_ctrl], return_inverse=True)
    keep = [i for i, g in enumerate(labels_present) if g in index and "+" not in g]
    counts = np.bincount(inv, minlength=len(labels_present))
    keep = [i for i in keep if counts[i] >= min_cells]
    if not keep:
        raise ValueError("no single-gene perturbation with enough cells")
    nodes = np.array([index[labels_present[i]] for i in keep], dtype=int)
    order = np.argsort(nodes)
    keep = [keep[i] for i in order]
    nodes = nodes[order]
    pert_idx = np.where(~is_ctrl)[0]
    code = {labels_present[i]: a for a, i in enumerate(keep)}
    sel = np.array([lab[c] in code for c in pert_idx])
    cells_p = pert_idx[sel]
    grp = np.array([code[lab[c]] for c in cells_p])
    p = len(keep)

    ctrl_idx = np.where(is_ctrl)[0]
    Yc = Y[ctrl_idx]
    m0 = Yc.mean(axis=0).astype(np.float64)
    v0 = Yc.var(axis=0, ddof=1).astype(np.float64)
    n0 = len(ctrl_idx)
    mean, var, cnt = _group_mean_var(Y[cells_p], grp, p)
    lfc = mean - m0
    se = np.sqrt(var / np.maximum(cnt, 1)[:, None] + v0 / n0)
    se = np.maximum(se, 1e-6)

    # null: pseudo-perturbations drawn from the controls, against the remaining controls
    typical = int(np.clip(np.median(cnt), 5, max(5, n0 // 3)))
    zs = []
    for _ in range(n_null_groups):
        pick = rng.choice(n0, size=typical, replace=False)
        rest = np.ones(n0, dtype=bool)
        rest[pick] = False
        a, b = Yc[pick].astype(np.float64), Yc[rest].astype(np.float64)
        zs.append((a.mean(0) - b.mean(0)) / np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b) + 1e-12))
    zs = np.asarray(zs)
    null_scale = np.maximum(1.4826 * np.median(np.abs(zs - np.median(zs, axis=0)), axis=0), 0.5)
    null_scale = np.minimum(np.maximum(null_scale, 0.5), 3.0)
    z = (lfc / se) / null_scale[None, :]
    pval = 2.0 * (1.0 - ndtr(np.abs(z)))
    mask = np.ones_like(pval, dtype=bool)
    mask[np.arange(p), nodes] = False
    qval = _bh(pval, mask)
    qval[np.arange(p), nodes] = pval[np.arange(p), nodes]
    # pooled within-condition covariance: every cell centred on the mean of its own condition (controls included)
    Yp = Y[cells_p].astype(np.float64) - mean[grp]
    Yc0 = Yc.astype(np.float64) - m0
    cov = (Yp.T @ Yp + Yc0.T @ Yc0) / max(len(cells_p) + n0 - p - 1, 1)
    return ACEEstimate(names, nodes, lfc, se, z, pval, qval, cnt.astype(int), int(n0), null_scale,
                       cov=cov, n_cov=int(len(cells_p) + n0))


def truth_ace(screen: Any) -> ACEEstimate:
    """Noise-free ACE from a :class:`ConditionScreen` (log2 fold changes of the single perturbations), for upper bounds."""
    singles = [k for k, c in enumerate(screen.conditions) if len(c) == 1]
    nodes = np.array([screen.conditions[k][0] for k in singles], dtype=int)
    order = np.argsort(nodes)
    singles = [singles[i] for i in order]
    nodes = nodes[order]
    lfc = screen.total_effects[:, singles].T.copy()
    p, n = lfc.shape
    z = np.where(np.abs(lfc) > 0, 10.0 * np.sign(lfc), 0.0)
    return ACEEstimate(list(screen.gene_names), nodes, lfc, np.full_like(lfc, 1e-6), z, np.zeros_like(lfc),
                       np.zeros_like(lfc), np.full(p, 1_000_000), 1_000_000, np.ones(n), method="noise_free")
