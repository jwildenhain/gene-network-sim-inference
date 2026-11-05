"""
inspre: sparse inversion of the knockdown-response matrix (Brown, Morris, Lappalainen & Knowles; Nat. Commun. 2024).

A screen that perturbs every gene measures the *total* effect of each gene on every other gene. In a linear cyclic
model with stable feedback (spectral radius of the direct-effect matrix G below 1) the matrix of total effects,
scaled so each gene's effect on itself is 1, is  R = D (I - G)^-1  for a diagonal D, hence the direct effects are

    G = I - V diag(1 / diag V),      V ~ R^-1 sparse.

``inspre_solve`` finds V by ADMM:  min_U 0.5 * sum_i w_i^2 ||(R U - I)_i||^2 + sum_jk lam_jk |V_jk| (off-diagonal),  U = V.
Per-gene weights ``w`` (a perturbation with a weak or noisy knockdown gets less weight, 0 drops it) and a per-edge
penalty matrix (``edge_prior``: lower penalty on prior-supported edges) are supported; both keep the U-update a single
Cholesky factorisation, so 500 genes take seconds.

Differences to the original package (documented, not hidden): weights are per perturbed gene (the package weights every
entry of R by its own standard error, which costs one solve per column); the regularisation level is chosen to match a
target number of edges (``k_interactions`` per gene) when none is given, instead of by cross-validation.

Input here is the repository's (X, U) pair. Column c holds the response X[:, c] to one perturbation, the perturbed gene
being the row where |U[:, c]| is largest. Only perturbed genes can be nodes of the estimated graph (the method needs the
response of each node to its own perturbation); genes without a perturbation column get zero rows and columns.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


def ace_from_xu(X: np.ndarray, U: np.ndarray, min_own_effect: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average-causal-effect matrix from conditions.

    Returns ``(genes, R, own)``: ``genes`` the perturbed gene index per node (one node per gene, first column wins),
    ``R`` the (p, n) matrix with R[a, j] = response of gene j to perturbing node a divided by the response of the
    perturbed gene itself (so R[a, genes[a]] = 1), and ``own`` the (p,) unnormalised own-gene responses.
    Nodes whose own response is below ``min_own_effect`` in magnitude (no knockdown) get an all-zero row of R and 0 in ``own``.
    """
    X = np.real(np.asarray(X, dtype=float))
    U = np.real(np.asarray(U, dtype=float))
    n, m = X.shape
    seen: Dict[int, int] = {}
    for c in range(m):
        col = np.abs(U[:, c])
        if col.max() <= 1e-15:
            continue
        g = int(np.argmax(col))
        seen.setdefault(g, c)
    genes = np.array(sorted(seen), dtype=int)
    R = np.zeros((len(genes), n))
    own = np.zeros(len(genes))
    for a, g in enumerate(genes):
        x = X[:, seen[g]]
        if abs(x[g]) >= min_own_effect:
            R[a] = x / x[g]
            own[a] = x[g]
    return genes, R, own


def _entry_factors(R: np.ndarray, entry_weights: np.ndarray) -> tuple:
    """(rho, per-column Cholesky factors, diag of the squared weights) for entry-weighted ADMM; reusable across lambdas."""
    p = R.shape[0]
    ew2 = np.asarray(entry_weights, dtype=float) ** 2
    rho = float(max(np.mean([np.sum((R * R) * ew2[:, [j]]) for j in range(min(p, 20))]) / max(p, 1), 1e-6))
    cfs = [cho_factor((R.T * ew2[:, j]) @ R + rho * np.eye(p), lower=True, check_finite=False) for j in range(p)]
    return rho, cfs, np.diag(ew2).copy()


def inspre_solve(
    R: np.ndarray,
    lam: float,
    weights: Optional[np.ndarray] = None,
    edge_penalty: Optional[np.ndarray] = None,
    rho: Optional[float] = None,
    max_iter: int = 300,
    tol: float = 1e-4,
    warm: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    entry_weights: Optional[np.ndarray] = None,
    entry_factors: Optional[tuple] = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], int]:
    """
    ADMM for a sparse V with R V ~ I (R square). ``weights`` (p,) are per-row weights (default 1), ``edge_penalty``
    an optional (p, p) matrix scaling the L1 penalty of each off-diagonal entry, ``entry_weights`` an optional (p, p)
    matrix of weights for every entry of R (the original package's inverse-variance weighting). Entry weights need one
    factorisation per column (cost ~p^4 per solve, p^3 memory): use for a few hundred nodes at most; ``weights`` is the
    cheap alternative. Returns ``(V, (V, Z), iterations)``;
    pass the second item as ``warm`` to continue from it (used along a lambda path).
    """
    R = np.asarray(R, dtype=float)
    p = R.shape[0]
    w2 = np.ones(p) if weights is None else np.asarray(weights, dtype=float) ** 2
    RtW = R.T * w2                                   # R^T W^2
    RtR = RtW @ R
    cfs = None
    if entry_weights is not None or entry_factors is not None:
        rho, cfs, diag_w2 = entry_factors if entry_factors is not None else _entry_factors(R, entry_weights)
    else:
        if rho is None:
            rho = float(max(np.trace(RtR) / max(p, 1), 1e-6))
        cf = cho_factor(RtR + rho * np.eye(p), lower=True, check_finite=False)
    pen = np.full((p, p), float(lam)) if edge_penalty is None else float(lam) * np.asarray(edge_penalty, dtype=float)
    thr = pen / rho
    np.fill_diagonal(thr, 0.0)
    V, Z = (np.eye(p), np.zeros((p, p))) if warm is None else (warm[0].copy(), warm[1].copy())
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007  (`it` is returned after the loop)
        if cfs is not None:
            Uk = np.empty((p, p))
            for j in range(p):
                Uk[:, j] = cho_solve(cfs[j], diag_w2[j] * R[j, :] + rho * (V[:, j] - Z[:, j]), check_finite=False)
        else:
            Uk = cho_solve(cf, RtW + rho * (V - Z), check_finite=False)
        T = Uk + Z
        V_new = np.sign(T) * np.maximum(np.abs(T) - thr, 0.0)     # diagonal: threshold 0 -> unpenalised
        Z = Z + Uk - V_new
        r_pri = np.linalg.norm(Uk - V_new)
        r_dual = rho * np.linalg.norm(V_new - V)
        V = V_new
        scale = max(np.linalg.norm(V), 1e-12)
        if r_pri <= tol * scale and r_dual <= tol * scale:
            break
    return V, (V, Z), it


def direct_effects(V: np.ndarray) -> np.ndarray:
    """G = I - V diag(1/diag V): G[a, b] is the direct effect of node a on node b (zero diagonal)."""
    d = np.diag(V).copy()
    d[np.abs(d) < 1e-12] = np.sign(d[np.abs(d) < 1e-12] + 1e-30) * 1e-12
    G = np.eye(V.shape[0]) - V / d[None, :]
    np.fill_diagonal(G, 0.0)
    return G


def spectral_radius(G: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(G)))) if G.size else 0.0


def fit_inspre(
    R: np.ndarray,
    target_edges: Optional[int] = None,
    lam: Optional[float] = None,
    weights: Optional[np.ndarray] = None,
    edge_penalty: Optional[np.ndarray] = None,
    max_iter: int = 300,
    path_steps: int = 14,
    entry_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Direct-effect matrix G from the node-by-node response matrix R. With ``lam`` fixed, one solve; otherwise lambda
    is searched (bisection in log space with warm starts) until the support of G has about ``target_edges`` entries.
    """
    p = R.shape[0]
    diag: Dict[str, Any] = {}
    ef = _entry_factors(R, entry_weights) if entry_weights is not None else None
    diag["entry_weights"] = ef is not None
    if lam is not None:
        V, _, it = inspre_solve(R, lam, weights, edge_penalty, max_iter=max_iter, entry_factors=ef)
        diag.update(lam=float(lam), iterations=it)
    else:
        if target_edges is None:
            target_edges = 3 * p
        target_edges = int(max(1, min(target_edges, p * (p - 1))))
        # lam_hi: no edges survive; lam_lo: dense
        scale = float(max(np.abs(R.T @ R).max(), 1e-9))
        lo, hi = np.log(scale * 1e-6), np.log(scale * 10.0)
        warm = None
        V = np.eye(p)
        it = 0
        for _ in range(path_steps):
            mid = 0.5 * (lo + hi)
            Vm, warm, it = inspre_solve(R, float(np.exp(mid)), weights, edge_penalty, max_iter=min(max_iter, 60), warm=warm,
                                      entry_factors=ef)
            nnz = int(np.count_nonzero(Vm - np.diag(np.diag(Vm))))
            V = Vm
            if nnz > target_edges:
                lo = mid
            else:
                hi = mid
            if abs(nnz - target_edges) <= max(1, int(0.05 * target_edges)):
                break
        V, _, it = inspre_solve(R, float(np.exp(mid)), weights, edge_penalty, max_iter=max_iter, warm=warm,
                                entry_factors=ef)   # polish
        diag.update(lam=float(np.exp(mid)), iterations=it)
    G = direct_effects(V)
    diag["V"] = V
    diag["spectral_radius"] = spectral_radius(G)
    diag["unstable"] = bool(diag["spectral_radius"] >= 1.0)    # the method assumes stable feedback (radius < 1)
    diag["edges"] = int(np.count_nonzero(G))
    return G, diag


class InspreAlgorithm(BaseInferenceAlgorithm):
    """
    inspre on the repository's (X, U) conditions. ``A_est[regulator, target]`` holds direct effects; genes that were
    not perturbed have zero rows and columns. ``density_factor * k_interactions`` edges per node are kept in the
    continuous score (extra depth for the precision-recall curve), the top ``k_interactions`` per node in the
    discretised network.
    """

    def __init__(
        self,
        k_interactions: int = 3,
        density_factor: float = 2.0,
        lam: Optional[float] = None,
        min_own_effect: float = 1e-8,
        weight_by_own_effect: bool = True,
    ):
        super().__init__(name="inspre")
        self.k_interactions = k_interactions
        self.density_factor = density_factor
        self.lam = lam
        self.min_own_effect = min_own_effect
        self.weight_by_own_effect = weight_by_own_effect

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        genes, R, own = ace_from_xu(X, U, self.min_own_effect)
        w = None
        if self.weight_by_own_effect and len(genes):
            a = np.abs(own)
            w = np.minimum(a / max(float(np.median(a[a > 0])) if np.any(a > 0) else 1.0, 1e-12), 1.0)
        return self._solve(np.asarray(X).shape[0], genes, R, w, **kwargs)

    def fit_ace(self, ace: Any, alpha: Optional[float] = 0.05, min_own_z: float = 2.0, entry_weights: bool = False,
                **kwargs) -> InferenceResult:
        """
        inspre on a calibrated ACE estimate (:class:`grn_sim.perturbseq.ace.ACEEstimate`): effects that are not
        significant at BH level ``alpha`` are set to 0 (``None`` keeps all), each node is weighted by the strength of
        its own knockdown, and nodes without a detectable on-target effect are dropped from the fit.
        """
        est = ace.denoised(alpha) if alpha is not None else ace
        R, w = est.normalised(min_own_z)
        if entry_weights:
            kwargs["entry_weights"] = est.entry_weights(min_own_z)[:, est.nodes]
        return self._solve(est.lfc.shape[1], est.nodes, R, w, **kwargs)

    def _solve(self, n: int, genes: np.ndarray, R: np.ndarray, w: Optional[np.ndarray], **kwargs) -> InferenceResult:
        t0 = time.time()
        k = int(kwargs.get("k", self.k_interactions))
        p = len(genes)
        R_nodes = R[:, genes] if p else R
        edge_penalty = kwargs.get("edge_penalty")
        entry_w = kwargs.get("entry_weights")
        if edge_penalty is not None:
            edge_penalty = np.asarray(edge_penalty, dtype=float)[np.ix_(genes, genes)]
        A = np.zeros((n, n))
        diagnostics: Dict[str, Any] = {"nodes": int(p), "perturbed_fraction": float(p / max(n, 1))}
        if p >= 3:
            G, diag = fit_inspre(R_nodes, target_edges=int(self.density_factor * k * p), lam=self.lam, weights=w,
                                 edge_penalty=edge_penalty, entry_weights=entry_w)
            A[np.ix_(genes, genes)] = G
            V = diag.pop("V")
            diagnostics.update(diag)
            rest = np.setdiff1d(np.arange(n), genes)
            if len(rest):
                # unperturbed genes as leaves: the direct effect of node b on gene j is (V R_unperturbed)[b, j],
                # since R_nodes = Dg (I - G)^-1 gives (I - G) = V Dg and the total effects on j are Dg T D_j
                A[np.ix_(genes, rest)] = V @ R[:, rest]
                diagnostics["unperturbed_genes_scored"] = int(len(rest))
        adj = np.zeros((n, n), dtype=int)
        for r in range(n):                                   # the k largest targets of every regulator
            mags = np.abs(A[r])
            top = np.argsort(-mags)[:k]
            for c in top:
                if mags[c] > 0:
                    adj[r, c] = int(np.sign(A[r, c]))
        return InferenceResult(self.name, A, adj, time.time() - t0, diagnostics)
