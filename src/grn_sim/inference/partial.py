"""
Algorithms on partial screens: m perturbed genes out of n measured (m < n), the rest only observed.

The Gardner-family algorithms (S2G, G2S, NIR, RidgeCV, FAST, ...) solve ``A X = -U`` for a square problem with one
perturbation per gene. With m < n columns most of them fail on shape (S2G, G2S, Vote, NetDeconv) or quietly return rows
for the perturbed genes only (RidgeCV, FAST, DMD-GRN). ``PartialAdapter`` makes any algorithm that estimates a steady-state
Jacobian usable:

1. **Perturbed block.** The algorithm is fitted on the square sub-problem of the perturbed genes (rows and columns of X
   and U restricted to them). This gives the edges among perturbed genes, selected by the algorithm itself.
2. **Unperturbed genes.** For a gene j that was never perturbed, ``x_j = sum_b w_jb x_b`` over the perturbed genes b
   (the steady-state response of j is carried by the responses of its regulators). With the algorithm's own estimate
   ``A X = -U`` on the perturbed block, ``X_PP^-1 = -U_PP^-1 A_PP``, so  w_j = -x_j diag(1/u) A_PP : the sparse inverse the
   algorithm found is applied to the observed total effects on j. The weight w_jb is the score of the edge b -> j.

For the per-row algorithms (S2G, G2S, RidgeCV) step 2 uses the algorithm's own row solver instead: the unperturbed gene's
response ``x_j`` (over the m conditions) is the target and the perturbed genes' responses are the candidate regulators,
so each algorithm selects the regulators of every unperturbed gene exactly as it would for a perturbed one. This is the
Gardner (2003) convention for genes without a perturbation, restricted to the perturbed genes as candidates.

Assumption (stated, not tested): unperturbed genes do not regulate perturbed ones (they are leaves for the perturbed
block). Edges from unperturbed genes are not scored at all: a regulator that is never perturbed cannot be identified
from interventions.

Output is ``A_est[regulator, target]`` over all n genes (rows of unperturbed genes are zero). inspre does the same
extension internally (``D = V R_unperturbed``), and its ``fit`` handles partial screens by itself.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Tuple

import numpy as np

from grn_sim.inference import orientation
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.inspre import ace_from_xu


def square_subproblem(X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(P, cols, X_PP, U_PP)``: the perturbed genes, the column of each, and the square problem they define."""
    X = np.real(np.asarray(X, dtype=float))
    U = np.real(np.asarray(U, dtype=float))
    seen = {}
    for c in range(U.shape[1]):
        col = np.abs(U[:, c])
        if col.max() > 1e-15:
            seen.setdefault(int(np.argmax(col)), c)
    P = np.array(sorted(seen), dtype=int)
    cols = np.array([seen[g] for g in P], dtype=int)
    return P, cols, X[np.ix_(P, cols)], U[np.ix_(P, cols)]


def extend_to_unperturbed(A_tr: np.ndarray, X: np.ndarray, U: np.ndarray, P: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """
    ``S[regulator, target]`` over all genes from ``A_tr`` = the algorithm's A[target, regulator] on the perturbed block.
    Perturbed block: ``|A_tr|`` transposed; unperturbed targets: ``w_j = -(x_j / u) A_tr`` (see module docstring).
    """
    n = X.shape[0]
    u = U[P, cols]
    u = np.where(np.abs(u) < 1e-12, 1e-12, u)
    X_all = np.real(np.asarray(X, dtype=float))[:, cols]                 # (n, p): responses to every perturbation
    W = -(X_all / u[None, :]) @ A_tr                                     # (n, p): [gene j, regulator b]
    S = np.zeros((n, n))
    S[P, :] = W.T
    S[np.ix_(P, P)] = A_tr.T
    np.fill_diagonal(S, 0.0)
    return S


def _row_solver(key: str, algo: Any):
    """``f(y, B) -> (indices, coefficients)`` for algorithms that solve one target at a time, or None."""
    if key == "s2g":
        return lambda y, B: algo._getvar(y, B)
    if key == "g2s":
        return lambda y, B: algo._gen2spec_reference(y, B)
    if key == "ridge_cv":
        from sklearn.linear_model import RidgeCV

        def solve(y: np.ndarray, B: np.ndarray):
            m = RidgeCV(alphas=algo.alphas, cv=min(algo.cv, max(2, B.shape[0] // 2))).fit(B, y)
            return np.arange(B.shape[1]), np.asarray(m.coef_, dtype=float)
        return solve
    return None


def extend_with_row_solver(solver: Any, X: np.ndarray, P: np.ndarray, cols: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Fills S[P, j] for every unperturbed gene j with the regulators the row solver selects (candidates: perturbed genes)."""
    n = X.shape[0]
    X = np.real(np.asarray(X, dtype=float))
    B = X[np.ix_(P, cols)].T                                   # (conditions, perturbed regulators)
    inP = np.zeros(n, dtype=bool)
    inP[P] = True
    for j in np.flatnonzero(~inP):
        y = X[j, cols].copy()
        if np.all(np.abs(y) < 1e-15):
            continue
        idx, coef = solver(y, B)
        S[P[np.asarray(idx, dtype=int)], j] = np.abs(coef)
    return S


class PartialAdapter(BaseInferenceAlgorithm):
    """Wraps an (X, U) algorithm that estimates a steady-state Jacobian so it can be fitted on a partial screen."""

    def __init__(self, algo: Any, key: str, data_model: str = orientation.ADDITIVE_GARDNER):
        super().__init__(name=f"{key}+partial")
        self.algo = algo
        self.key = key
        self.data_model = data_model

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs: Any) -> InferenceResult:
        t0 = time.time()
        P, cols, X_PP, U_PP = square_subproblem(X, U)
        res = self.algo.fit(X_PP, U_PP, **kwargs)
        A_tr = orientation.orient_prediction(self.key, res.A_est, self.data_model, target_regulator=True)
        S = extend_to_unperturbed(np.real(A_tr), X, U, P, cols)
        solver = _row_solver(self.key, self.algo)
        if solver is not None:
            S[:, :] = np.where(np.isin(np.arange(S.shape[0]), P)[:, None] & np.isin(np.arange(S.shape[1]), P)[None, :], S, 0.0)
            S = extend_with_row_solver(solver, X, P, cols, S)
        adj = np.zeros(S.shape, dtype=int)
        diag = dict(res.diagnostics)
        diag.update(perturbed_genes=int(len(P)), adapted_from=self.key,
                    unperturbed_extension="row solver" if solver is not None else "inverse of the perturbed block")
        return InferenceResult(self.name, S, adj, time.time() - t0, diag)
