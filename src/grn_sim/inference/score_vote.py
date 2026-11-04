"""
Score-based vote of Simple-to-General (S2G), General-to-Specific (G2S) and, optionally, NIR.

``VotingEnsemble`` keeps an edge only when the members agree on its presence and sign, which discards how strong
each call is. Here every candidate edge (regulator j -> target i) in the union of the members' supports gets a signed
evidence score from each member: the t-statistic of its least-squares coefficient when the edge is added to (or kept
in) that member's support, in that member's own regression, mapped to a z-score through the t distribution. The
members' signed z-scores are combined as ``sum(z) / sqrt(m + 2 * sum_{a<b} rho_ab)``, with rho the correlation of
two members' scores over the candidates (they share data, so they are not independent). Opposite signs cancel.
``|z|`` ranks the edges, so AUPR is defined, and the calls are the top-K regulators per target (K = ``k_interactions``)
or, without K, the edges with ``|z| >= z_threshold``.

Regressions per target gene i, with columns j the candidate regulators:
  * S2G and G2S: ``U[i, :]`` on the columns of ``X``;
  * NIR in regulator-input mode (the Hill simulator's model): ``X[i, :]`` on the rows of ``X + U``.
"""

import time
from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from grn_sim.inference._parallel import default_n_jobs, map_indices
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.s2g import SimpleToGeneral

_Z_CAP = 8.0


def _signed_z(y: np.ndarray, X: np.ndarray, support: List[int], j: int) -> float:
    """Signed z of column ``j`` when added to (or kept in) ``support`` in the least-squares fit of ``y`` on X."""
    cols = list(support) if j in support else list(support) + [j]
    Xs = X[:, cols]
    n_obs = Xs.shape[0]
    pinv = np.linalg.pinv(Xs)
    a = pinv @ y
    rss = float(np.sum((Xs @ a - y) ** 2))
    df = n_obs - np.linalg.matrix_rank(Xs)
    if df <= 0:
        return 0.0
    cov_diag = np.einsum("ij,ji->i", pinv, pinv.T)       # diag(pinv pinv^T) = diag((Xs^T Xs)^+)
    pos = cols.index(j)
    se = np.sqrt(max(rss / df, 1e-300) * cov_diag[pos])
    if not np.isfinite(se) or se <= 0.0:
        return 0.0
    t = a[pos] / se
    p = 2.0 * stats.t.sf(abs(t), df)
    z = _Z_CAP if p <= 0.0 else min(_Z_CAP, float(stats.norm.isf(max(p, 1e-300) / 2.0)))
    return float(np.sign(t)) * z


class ScoreVote(BaseInferenceAlgorithm):
    """S2G + G2S (+ NIR) combined through per-edge evidence scores instead of a presence/sign vote."""

    def __init__(
        self,
        s2g: Optional[SimpleToGeneral] = None,
        g2s: Optional[GeneralToSpecific] = None,
        nir: Optional[NIRAlgorithm] = None,
        k_interactions: Optional[int] = None,
        z_threshold: float = 3.0,
        rho: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(name="score_vote")
        self.s2g = s2g or SimpleToGeneral()
        self.g2s = g2s or GeneralToSpecific()
        self.nir = nir
        self.k_interactions = k_interactions
        self.z_threshold = z_threshold
        self.rho = rho
        self.n_jobs = n_jobs
        self.last_scores: Optional[np.ndarray] = None   # combined signed z, [regulator, target]
        self.last_rho: Optional[float] = None

    @staticmethod
    def _target_scores(payload, i: int):
        """Signed z of every candidate regulator of target ``i`` in each member's regression (None: not a candidate)."""
        regs, supports, y_list, des_list = payload
        cand = sorted({int(j) for s in supports for j in np.flatnonzero(s[:, i])})
        out = []
        for m, support in enumerate(supports):
            sm = [int(j) for j in np.flatnonzero(support[:, i])]
            out.append({j: _signed_z(y_list[m](i), des_list[m], sm, j) for j in cand})
        return out

    def score(self, X: np.ndarray, U: np.ndarray, results: List[InferenceResult]) -> Tuple[np.ndarray, float]:
        """
        Combined signed z matrix [regulator, target] (zero off the candidate set) and the mean pairwise correlation
        used to combine. ``results`` are the members' fits in the same [regulator, target] frame, S2G and G2S first,
        then NIR when present.
        """
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)
        n = X_f.shape[0]
        supports = [r.A_est != 0 for r in results]

        def y_su(i):                       # S2G / G2S: U[i, :] on the columns of X (X[i, :] when gene i is unperturbed)
            y = U_f[i, :].copy()
            return X_f[i, :].copy() if np.all(np.abs(y) < 1e-15) else y

        y_list = [y_su, y_su]
        des_list = [X_f, X_f]
        if len(results) == 3:              # NIR, regulator-input mode: X[i, :] on the rows of X + U
            Z = (X_f + U_f).T
            y_list.append(lambda i: X_f[i, :].copy())
            des_list.append(Z)
        payload = (None, supports, y_list, des_list)
        n_jobs = self.n_jobs if self.n_jobs is not None else default_n_jobs(cap=8)
        per_target = map_indices(self._target_scores, payload, n, n_jobs)

        z = np.zeros((len(results), n, n))
        for i, members in enumerate(per_target):
            for m, scores in enumerate(members):
                for j, v in scores.items():
                    z[m, j, i] = v
        cand = np.zeros((n, n), dtype=bool)
        for s in supports:
            cand |= s
        if self.rho is not None:
            rho = float(self.rho)
        else:
            pair = []
            for a, b in combinations(range(len(results)), 2):
                if cand.sum() > 2 and z[a][cand].std() > 0 and z[b][cand].std() > 0:
                    pair.append(float(np.clip(np.corrcoef(z[a][cand], z[b][cand])[0, 1], 0.0, 0.9)))
                else:
                    pair.append(0.0)
            rho = float(np.mean(pair)) if pair else 0.0
        m = len(results)
        denom = np.sqrt(m + m * (m - 1) * rho)          # m + 2 * sum_{a<b} rho, with a common rho
        return z.sum(axis=0) / denom, rho

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start = time.time()
        results = [self.s2g.fit(X, U, **kwargs), self.g2s.fit(X, U, **kwargs)]
        if self.nir is not None:
            results.append(self.nir.fit(X, U, **kwargs))
        z, rho = self.score(X, U, results)
        self.last_scores, self.last_rho = z, rho
        n = z.shape[0]
        K = kwargs.get("k", self.k_interactions)
        adj = np.zeros(z.shape, dtype=int)
        if K is not None:
            K = max(1, min(int(K), n - 1))
            for j in range(n):                          # column j = target gene
                mag = np.abs(z[:, j])
                top = np.argsort(mag)[::-1][:K]
                top = top[mag[top] > 1e-12]
                adj[top, j] = np.sign(z[top, j]).astype(int)
        else:
            adj[z >= self.z_threshold] = 1
            adj[z <= -self.z_threshold] = -1
        A = np.where(adj != 0, np.mean([r.A_est for r in results], axis=0), 0.0)
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A,
            adjacency_pred=adj,
            time_elapsed=time.time() - start,
            diagnostics={"members": [r.algorithm_name for r in results], "rho": rho,
                         "k_interactions": K, "consensus_edges": int(np.sum(adj != 0))},
        )
