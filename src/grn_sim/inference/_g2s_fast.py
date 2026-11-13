"""
Fast, exact-equivalent engine for the General-to-Specific (G2S) backward elimination.

The reference implementation (``GeneralToSpecific._gen2spec_reference``) recomputes
``pinv(V[:, ind])`` after every removed column, i.e. one SVD of an N x k matrix per step and
O(N M^3) work per target gene.  This engine reproduces the same elimination path with
O(k^2) work per step:

* The reference's "fast LOO" drop score for column i is ``(SSE + alpha_i^2 / ||pinv row i||^2) / N``.
  With a full-column-rank basis ``pinv = G^-1 V^T`` (``G = V^T V``), so ``||pinv row i||^2 = G^-1_ii``
  and ``alpha = G^-1 V^T y``.  We keep a square factor ``W`` with ``W W^T = G^-1`` (initially
  ``W = Vt^T diag(1/s)`` from the SVD) and the projected target ``c = W^T V^T y`` (``alpha = W c``).
  Removing column j is ``G'^-1 = W_{-j} (I - u u^T) W_{-j}^T`` with ``u = w_j / |w_j|``; a Householder
  reflection that maps ``w_j`` onto the last coordinate turns this into "delete row j and the last
  column" -- an orthogonal, numerically stable O(k^2) downdate.  ``SSE`` grows by exactly the drop
  score ``alpha_j^2 / G^-1_jj``.
* The initial factorisation of the full basis ``V`` is shared by all target genes (it is the same
  matrix for every gene), so the only per-gene SVD-sized work is gone.
* Every decision taken on updated (rather than freshly recomputed) quantities is *certified*: a
  rounding-error bound is carried for every drop score and for the AIC/MDL penalty.  When the
  winning drop score is not separated from the runner-up by more than the bounds, or a penalty
  comparison is too close to call, or the basis is rank deficient / not of full column rank
  (``N < k``, collinear columns, singular values under numpy's ``pinv`` cutoff), the engine falls back
  to the reference computation for that step (``pinv`` via the same SVD numpy uses).  Fallback
  steps reproduce the reference bit for bit, so near-ties are resolved exactly like the reference.

The full-column-rank downdate runs in a numba-compiled in-place kernel when numba is importable
(``GRN_SIM_G2S_NUMBA=0`` disables it); otherwise in numpy.  Backends: ``numpy`` (serial) and
``processes`` (target genes spread over a fork-based process pool, one BLAS thread per worker).
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_EPS = float(np.finfo(float).eps)
_RCOND = 1e-15          # numpy.linalg.pinv default
_TOL_C = 16.0           # safety factor on the a-priori rounding-error bounds, full-column-rank bases
_TOL_C_DEF = 256.0      # same for column-rank-deficient bases (reference SVDs differ more between steps)
_KAPPA_MAX = 1e13       # do not downdate worse-conditioned bases; use the reference steps


# ----------------------------------------------------------------------------------------------
# numpy reference building blocks
# ----------------------------------------------------------------------------------------------

def pinv_with_svd(Vb: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """``numpy.linalg.pinv(Vb)`` (default rcond) computed exactly as numpy does, plus the SVD pieces.

    Returns ``(pinv, (u, s, vt))``; the SVD tuple is ``None`` for an empty matrix.
    """
    if Vb.size == 0:
        return np.linalg.pinv(Vb), None
    u, s, vt = np.linalg.svd(Vb, full_matrices=False)
    s_keep = s.copy()
    rcond = np.asarray(_RCOND)
    cutoff = rcond[..., np.newaxis] * np.amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0
    res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
    return res, (u, s_keep, vt)


@dataclass
class _Factor:
    """Factor of the pseudo-inverse of an N x k basis of numerical rank r: ``pinv = W u^T``.

    ``W`` is k x r (``W W^T = G^+``), ``u`` the N x r left singular vectors, ``B = u^T V_b`` the r x k
    column coordinates (only kept while the basis is column-rank deficient, ``r < k``).  ``noise`` bounds
    the singular values ``pinv`` truncates (0 when there are none).
    """
    W: np.ndarray
    u: np.ndarray
    B: Optional[np.ndarray]
    kappa: float
    smax: float
    noise: float


def _make_factor(svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], k: int,
                 colnorm_max: float) -> Optional[_Factor]:
    """Factor for downdating, or ``None`` when ``pinv``'s truncation is not clear-cut for this basis."""
    if svd is None or k == 0:
        return None
    u, s, vt = svd
    smax = float(s[0])
    if not (np.isfinite(smax) and smax > 0.0):
        return None
    cutoff = _RCOND * smax
    r = int(np.sum(s > cutoff))
    if r == 0:
        return None
    smin = float(s[r - 1])
    noise = float(s[r]) if r < s.size else 0.0
    kappa = smax / smin
    if smin <= 10.0 * cutoff or kappa > _KAPPA_MAX:
        return None
    if _noise_eff(noise, smax, r < k) >= 0.3 * _RCOND * max(smax, colnorm_max):
        return None
    W = np.ascontiguousarray(vt[:r].T / s[:r])
    B = np.ascontiguousarray(s[:r, None] * vt[:r]) if r < k else None
    return _Factor(W=W, u=np.ascontiguousarray(u[:, :r]), B=B, kappa=kappa, smax=smax, noise=noise)


def _noise_eff(noise: float, smax: float, deficient: bool) -> float:
    """Effective bound on the singular values pinv truncates (incl. the SVD's own rounding floor)."""
    if noise == 0.0 and not deficient:
        return 0.0
    return max(2.0 * noise, 5e-17 * smax)


@dataclass
class SharedBasis:
    """Factorisation of the full basis ``V`` (N x M), shared by every target gene."""
    V: np.ndarray
    pinv: np.ndarray
    d: np.ndarray                 # squared row norms of pinv (reference drop-score denominators)
    factor: Optional[_Factor]
    colnorms: np.ndarray


def make_shared_basis(V: np.ndarray) -> SharedBasis:
    M = V.shape[1]
    Vfull = V[:, list(range(M))]          # same contiguous copy the reference builds
    P, svd = pinv_with_svd(Vfull)
    d = np.sum(P ** 2, axis=1)
    colnorms = np.sqrt(np.einsum("ij,ij->j", Vfull, Vfull))
    f = _make_factor(svd, M, float(colnorms.max()) if M else 0.0)
    return SharedBasis(V=Vfull, pinv=P, d=d, factor=f, colnorms=colnorms)


@dataclass
class G2SParams:
    criterion: str
    min_interactions: int
    max_interactions: Optional[int]
    drop_rule: str = "exact"      # "exact": gen2spec.m dropanother (pinv per candidate); "loo": shortcut
    # Per-gene budget for the exact pinv residual evaluations (drop_rule="exact"), in floating-point work
    # (each evaluation of an N x k basis costs ~N*k^2). Beyond it, steps that would need them use the
    # deterministic tie-break (see _GeneState._tie_break); None: no budget (always MATLAB-exact).
    exact_budget_flops: Optional[float] = 2e9


def _penalty_value(sse: float, k: int, n_len: int, criterion: str) -> float:
    """Identical arithmetic to GeneralToSpecific._penalty (which receives sum(E^2))."""
    if criterion == "mdl":
        mse = max(sse / n_len, 1e-300)
        return float(n_len * np.log(mse) + k * np.log(n_len))
    return sse + float(k)


_G_MAX = 1e-5          # refresh once the relative error bound of the updated quantities exceeds this


def _hh_drop_py(W: np.ndarray, k: int, c: np.ndarray, j: int, a_out: np.ndarray, d_out: np.ndarray) -> None:
    """Full-column-rank downdate, in place: remove basis column j from ``W W^T = G^-1``.

    Householder-reflects row j of ``W`` onto the last coordinate, moves the last row into row j
    ("swap-delete") and drops the last column; ``c`` is reflected too.  On return ``W[:k-1, :k-1]``
    and ``c[:k-1]`` describe the reduced basis and ``a_out``/``d_out`` hold ``alpha = W c`` and the
    squared row norms of the new ``W``.  ``W`` is a C-contiguous buffer whose leading k x k block is
    active.  This loop version is compiled with numba when available.
    """
    v = np.empty(k)
    nw2 = 0.0
    for col in range(k):
        v[col] = W[j, col]
        nw2 += v[col] * v[col]
    nw = math.sqrt(nw2)
    if v[k - 1] >= 0.0:
        v[k - 1] += nw
    else:
        v[k - 1] -= nw
    vv = 0.0
    for col in range(k):
        vv += v[col] * v[col]
    beta = 2.0 / vv if vv > 0.0 else 0.0
    t = np.empty(k)
    for row in range(k):
        acc = 0.0
        for col in range(k):
            acc += W[row, col] * v[col]
        t[row] = acc * beta
    cv = 0.0
    for col in range(k):
        cv += c[col] * v[col]
    cv *= beta
    for col in range(k - 1):
        c[col] -= cv * v[col]
    if j != k - 1:
        for col in range(k):
            W[j, col] = W[k - 1, col]
        t[j] = t[k - 1]
    for row in range(k - 1):
        tr = t[row]
        ai = 0.0
        di = 0.0
        for col in range(k - 1):
            x = W[row, col] - tr * v[col]
            W[row, col] = x
            ai += x * c[col]
            di += x * x
        a_out[row] = ai
        d_out[row] = di


_KERNEL: Dict[str, Any] = {}


def _hh_kernel():
    """numba-compiled ``_hh_drop_py`` or ``None`` (numba missing or GRN_SIM_G2S_NUMBA=0)."""
    if "fn" not in _KERNEL:
        from grn_sim.inference._native import NATIVE_LOCK
        with NATIVE_LOCK:                                   # compiling loads native modules: never beside a library walk
            if "fn" not in _KERNEL:
                fn = None
                if os.environ.get("GRN_SIM_G2S_NUMBA", "1") != "0":
                    try:
                        import numba
                        fn = numba.njit(cache=False, nogil=True, fastmath={"reassoc", "contract"})(_hh_drop_py)
                        fn(np.eye(3), 3, np.ones(3), 1, np.empty(2), np.empty(2))   # compile now (before forking workers)
                    except Exception:  # pragma: no cover - numba missing or broken
                        fn = None
                _KERNEL["fn"] = fn
    return _KERNEL["fn"]


class _GeneState:
    """Per-gene elimination state: exact (reference quantities) or downdated with error bounds."""

    __slots__ = ("V", "y", "N", "ny", "ind", "colnorms", "exact", "alpha", "d", "sse", "sse_err",
                 "updatable", "W", "c", "B", "dref", "kappa", "smax", "noise", "k0", "t", "amp",
                 "checks", "g", "neff", "n_refresh", "n_updates", "why", "ad", "cn_all", "kernel", "Wbuf", "sse_ref", "n_exact_evals", "smax_lb2",
                 "budget", "n_tie_breaks", "last_svd", "exact_work")

    def __init__(self, V: np.ndarray, y: np.ndarray, colnorms: np.ndarray, budget: Optional[float] = None):
        self.budget = budget
        self.exact_work = 0.0
        self.n_tie_breaks = 0
        self.last_svd: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.V = V
        self.y = y
        self.N = V.shape[0]
        self.ny = float(np.linalg.norm(y))
        self.cn_all = colnorms
        self.colnorms = colnorms
        self.ad: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.kernel = _hh_kernel()
        self.n_refresh = 0
        self.why: Dict[str, int] = {}
        self.n_updates = 0
        self.n_exact_evals = 0
        self.sse_ref = True
        self.smax_lb2 = 0.0

    def _why(self, reason: str) -> None:
        self.why[reason] = self.why.get(reason, 0) + 1

    def _adopt(self, f: Optional[_Factor], k: int) -> None:
        self.updatable = f is not None
        self.ad = None
        self.Wbuf: Optional[np.ndarray] = None
        if f is None:
            return
        # the kernel downdates in place: never touch the shared basis' factor
        if f.B is None and self.kernel is not None:
            self.Wbuf = np.ascontiguousarray(f.W).copy()
            self.W = self.Wbuf
        else:
            self.W = f.W
        self.c = f.u.T @ self.y
        self.B = f.B
        self.dref = np.einsum("ij,ij->i", f.W, f.W)
        self.kappa = f.kappa
        self.smax = f.smax
        self.smax_lb2 = f.smax ** 2
        self.noise = f.noise
        self.checks = f.B is not None or f.noise > 0.0
        self.k0 = k
        self.t = 0
        self.amp = 1.0

    def init_full(self, basis: SharedBasis) -> None:
        self.ind = np.arange(basis.V.shape[1])
        P = basis.pinv
        alpha = P @ self.y
        E = (basis.V @ alpha) - self.y
        self.alpha = alpha
        self.d = basis.d
        self.sse = float(np.sum(E ** 2))
        self.sse_err = 0.0
        self.exact = True
        self.sse_ref = True
        self._adopt(basis.factor, int(self.ind.size))

    def refresh(self) -> None:
        """Recompute everything for the current basis exactly as the reference does."""
        # the reference keeps the columns in their original order (downdates may have permuted them)
        self.ind = np.sort(self.ind)
        self.colnorms = self.cn_all[self.ind]
        Vb = self.V[:, self.ind]
        P, svd = pinv_with_svd(Vb)
        self.last_svd = svd
        alpha = P @ self.y
        E = (Vb @ alpha) - self.y
        self.alpha = alpha
        self.d = np.sum(P ** 2, axis=1)
        self.sse = float(np.sum(E ** 2))
        self.sse_err = 0.0
        self.exact = True
        self.sse_ref = True
        k = int(self.ind.size)
        self._adopt(_make_factor(svd, k, float(self.colnorms.max()) if k else 0.0), k)
        self.n_refresh += 1

    # -- error model ---------------------------------------------------------------------------
    def penalty(self, k: int, criterion: str) -> Tuple[float, float]:
        """(penalty, absolute error bound) for the current basis."""
        val = _penalty_value(self.sse, k, self.N, criterion)
        if self.exact or self.sse_ref:
            return val, 0.0
        g = self.g * self.ny
        err_sse = self.sse_err + g * (2.0 * math.sqrt(max(self.sse, 0.0)) + g) + 4.0 * _EPS * self.sse
        if criterion == "mdl":
            if self.sse <= 0.0 or err_sse >= 0.5 * self.sse or self.sse / self.N < 1e-300:
                return val, math.inf
            return val, self.N * 2.0 * err_sse / self.sse + 8.0 * _EPS * abs(val)
        return val, err_sse + 4.0 * _EPS * abs(val)

    def _cut_lo(self) -> float:
        """Lower bound on pinv's cutoff 1e-15*sigma_max(V_b) for the current basis.

        Uses sigma_max(V_b)^2 >= sigma_max(at refresh)^2 - sum of removed squared column norms
        (Weyl) and sigma_max >= the largest column norm.
        """
        cm = float(self.colnorms.max()) if self.colnorms.size else 0.0
        return _RCOND * max(cm, math.sqrt(max(self.smax_lb2, 0.0)))

    def _bounds_ok(self, d: np.ndarray) -> bool:
        """Error growth factor ``g`` for the downdated W; False if a refresh is needed."""
        g = (_TOL_C_DEF if self.checks else _TOL_C) * _EPS * (self.kappa + self.k0 + self.t) * self.amp
        if self.checks:
            neff = _noise_eff(self.noise, self.smax, self.B is not None)
            if neff >= 0.3 * self._cut_lo():
                self._why("noise")
                return False            # truncated directions might no longer be truncated
            sumd = float(np.sum(d))
            if sumd > 0.0 and 1.0 / math.sqrt(sumd) - neff <= 3.0 * _RCOND * self.smax:
                self._why("kept")
                return False            # a kept direction might fall under pinv's cutoff
            g += _TOL_C * neff * math.sqrt(sumd)
            self.neff = neff
        self.g = g
        if g > _G_MAX:
            self._why("bound")
            return False
        return True

    # -- drop selection -----------------------------------------------------------------------
    def choose_loo(self) -> Tuple[int, float, float]:
        """LOO-shortcut rule: (position j to drop, drop score alpha_j^2/d_j, error bound of the score)."""
        k = int(self.ind.size)
        if not self.exact:
            if self.ad is not None:
                a, d = self.ad
            else:
                W = self.W
                a = W @ self.c
                d = np.einsum("ij,ij->i", W, W)
            if self._bounds_ok(d):
                self.dref = dref = np.maximum(self.dref, d)
                g = self.g
                da = g * np.sqrt(dref) * self.ny
                dd = g * np.sqrt(dref * d)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    s = a * a / d
                    aa = np.abs(a)
                    s_hi = (aa + da) ** 2 / (d - dd)
                    s_lo = np.maximum(aa - da, 0.0) ** 2 / (d + dd)
                    tol = np.maximum(s_hi - s, s - s_lo)
                tol = np.where(dd < 0.5 * d, tol, np.inf)
                tol = tol + 4.0 * _EPS * (self.sse + s)
                j = int(np.argmin(s))
                sj = float(s[j])
                tj = float(tol[j])
                certified = math.isfinite(tj) and math.isfinite(sj)
                if certified and k > 1:
                    lo = s - tol
                    lo[j] = np.inf
                    certified = bool(sj + tj < float(np.min(lo)))
                if certified:
                    return j, sj, tj
                self._why("tie")
            self.refresh()
        # exact reference arithmetic
        dmax = np.maximum(self.d, 1e-30)
        e0 = (self.sse + (self.alpha ** 2) / dmax) / self.N
        j = int(np.argmin(e0))
        if self.updatable:
            self.g = (_TOL_C_DEF if self.checks else _TOL_C) * _EPS * (self.kappa + self.k0) * self.amp
            if self.checks:
                self.neff = _noise_eff(self.noise, self.smax, self.B is not None)
                self.g += _TOL_C * self.neff * math.sqrt(float(np.sum(self.dref)))
        return j, float(self.alpha[j] ** 2 / dmax[j]), 0.0

    # -- exact (gen2spec.m dropanother) rule ----------------------------------------------------
    def _exact_eval(self, cands: Sequence[int]) -> Tuple[int, float, float]:
        """Reference dropanother arithmetic for the candidate positions ``cands`` only.

        Evaluates ``mean((V_b pinv(V_b) y - y)^2)`` with column i removed, exactly as the reference loop
        (columns in original order), and returns (state position of the argmin, its error increase,
        ``sum(E1^2)`` of the reduced basis as the reference computes it).  Ties go to the smallest
        original position, like ``np.argmin`` over the full candidate list.
        """
        k = int(self.ind.size)
        order = np.argsort(self.ind, kind="stable")
        sorted_list = self.ind[order].tolist()
        inv = np.empty(k, dtype=np.intp)
        inv[order] = np.arange(k)
        best_e0 = math.inf
        best_p = -1
        best_e: Optional[np.ndarray] = None
        for p in sorted(int(inv[c]) for c in cands):
            ind0 = sorted_list[:p] + sorted_list[p + 1:]
            Vb = self.V[:, ind0]
            a = np.linalg.pinv(Vb) @ self.y
            yh = Vb @ a
            e = yh - self.y
            e0 = float(np.mean(e ** 2))
            if best_p < 0 or e0 < best_e0:
                best_e0, best_p, best_e = e0, p, e
            self.n_exact_evals += 1
            self.exact_work += float(self.N) * (k - 1) ** 2
        assert best_e is not None
        sse_new = float(np.sum(best_e ** 2))
        return int(order[best_p]), max(sse_new - self.sse, 0.0), sse_new

    def choose_exact(self) -> Tuple[int, float, float, Optional[float]]:
        """Exact rule: (position j, error increase, its error bound, reference sum(E1^2) or None).

        Candidates whose error increase is certified by the downdated quantities are compared with
        intervals; the reference pinv residual is computed only for the candidates that could be the
        minimum (ties, columns inside a linear dependency, unclear rank changes).
        """
        k = int(self.ind.size)
        if not self.exact:
            if self.ad is not None:
                a, d = self.ad
            else:
                a = self.W @ self.c
                d = np.einsum("ij,ij->i", self.W, self.W)
            if not self._bounds_ok(d):
                self.refresh()
        if self.exact:
            if not self.updatable:
                return self._full_exact(k)
            a = self.W @ self.c
            d = np.einsum("ij,ij->i", self.W, self.W)
            self.g = (_TOL_C_DEF if self.checks else _TOL_C) * _EPS * (self.kappa + self.k0) * self.amp
            if self.checks:
                self.neff = _noise_eff(self.noise, self.smax, self.B is not None)
                self.g += _TOL_C * self.neff * math.sqrt(float(np.sum(d)))
        self.dref = dref = np.maximum(self.dref, d)
        g = self.g
        gy = g * self.ny
        da = gy * np.sqrt(dref)
        dd = g * np.sqrt(dref * d)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = a * a / d
            aa = np.abs(a)
            s_hi = (aa + da) ** 2 / (d - dd)
            s_lo = np.maximum(aa - da, 0.0) ** 2 / (d + dd)
        ok = (dd < 0.5 * d) & np.isfinite(s_hi)
        hi = np.where(ok, s_hi, np.inf)
        lo = np.where(ok, s_lo, 0.0)
        inside = None
        if self.B is not None:
            # column-rank deficient: removing a column inside a dependency leaves the column space
            # (and the residual) unchanged; only "essential" columns change it by alpha_i^2/d_i.
            h = np.einsum("ij,ji->i", self.W, self.B)
            one_m_h = 1.0 - h
            with np.errstate(invalid="ignore", divide="ignore"):
                sz = np.sqrt(np.maximum(h * one_m_h, 0.0) / d)
            inside = (one_m_h >= max(1e-6, 1e3 * g)) & (sz > 10.0 * _RCOND * self.smax + self.neff)
            lo = np.where(inside, 0.0, lo)
            hi = np.where(inside, 0.0, hi)
            s = np.where(inside, 0.0, s)
        sse = max(self.sse, 0.0)
        noise = gy * (2.0 * np.sqrt(sse + hi) + gy) + 4.0 * _EPS * (sse + hi)
        lo = lo - noise
        hi = hi + noise
        top = float(np.min(hi))
        if inside is not None:
            # a candidate is excluded on its alpha_i^2/d_i only if it is certified essential: the other
            # columns' component along the direction it alone spans must be pinv-truncated noise
            check = np.flatnonzero(~inside & (lo > top))
            if check.size:
                Wc = self.W[check]
                zB = (Wc @ self.B) / np.sqrt(d[check])[:, None]
                zB[np.arange(check.size), check] = 0.0
                rho = np.sqrt(np.einsum("ij,ij->i", zB, zB))
                unsure = check[rho + self.neff >= 0.3 * self._cut_lo()]
                lo[unsure] = -noise[unsure]
        cands = np.flatnonzero(lo <= top)
        if cands.size == 1 and math.isfinite(top) and (inside is None or inside[int(cands[0])]):
            j = int(cands[0])
            return j, float(s[j]), float(hi[j] - lo[j]), None
        if self._over_budget(int(cands.size)):
            return self._tie_break()
        self._why("exact_eval")
        j, inc, sse_new = self._exact_eval(cands.tolist())
        return j, inc, 0.0, sse_new

    def _full_exact(self, k: int) -> Tuple[int, float, float, Optional[float]]:
        if self._over_budget(k):
            return self._tie_break()
        self._why("full_exact")
        j, inc, sse_new = self._exact_eval(range(k))
        return j, inc, 0.0, sse_new

    def _over_budget(self, n_evals: int) -> bool:
        if self.budget is None:
            return False
        k = int(self.ind.size)
        return self.exact_work + n_evals * float(self.N) * (k - 1) ** 2 > self.budget

    def _tie_break(self) -> Tuple[int, float, float, Optional[float]]:
        """
        Deterministic substitute for the exact pinv evaluations once the gene's budget is spent.

        With the current basis' SVD: a column inside a linear dependency (its row-space norm
        ``sum(vt[:r, i]^2) < 1``) can be removed without changing the column space, so its error increase is
        exactly 0; any other column increases the error by ``alpha_i^2 / d_i``. The smallest increase wins, ties
        going to the lowest original column index. Any tied column gives the same fit at this step (later steps can
        differ); MATLAB's choice among them is set by pinv rounding, which this does not reproduce (reported as
        ``tie_breaks``).
        """
        self._why("tie_break")
        self.n_tie_breaks += 1
        self.refresh()                     # exact alpha, d, sse for the basis, columns in original order
        k = int(self.ind.size)
        svd = self.last_svd
        dependent: np.ndarray = np.zeros(k, dtype=bool)
        if svd is not None:
            _, sv, vt = svd
            r = int(np.sum(sv > _RCOND * float(sv[0]))) if sv.size else 0
            in_rowspace = np.einsum("ij,ij->j", vt[:r], vt[:r]) if r else np.zeros(k)
            dependent = np.asarray((1.0 - in_rowspace) > 1e-8, dtype=bool)
        inc = np.where(dependent, 0.0, self.alpha ** 2 / np.maximum(self.d, 1e-30))
        j = int(np.argmin(inc))            # first minimum: lowest original column index
        return j, float(inc[j]), 0.0, self.sse + float(inc[j])

    def drop(self, j: int, sj: float, tj: float, sse_new: Optional[float] = None) -> None:
        self._drop(j, sj, tj)
        if sse_new is not None and not self.exact:
            self.sse = sse_new
            self.sse_err = 0.0
            self.sse_ref = True

    def _drop(self, j: int, sj: float, tj: float) -> None:
        self.ad = None
        self.sse_ref = False
        if self.updatable and self.B is None and self.kernel is not None:
            self._drop_kernel(j, sj, tj)
            return
        self.ind = np.delete(self.ind, j)
        self.smax_lb2 -= float(self.colnorms[j]) ** 2
        self.colnorms = np.delete(self.colnorms, j)
        if not self.updatable:
            self._why("rankdef")
            self.refresh()
            return
        W = self.W
        w = W[j]
        B = self.B
        if B is not None:
            bj = B[:, j]
            nw = float(np.linalg.norm(w))
            zB = (w @ B) / nw if nw > 0 else np.zeros(B.shape[1])
            zB[j] = 0.0
            rho = float(np.linalg.norm(zB))
            h = float(w @ bj)
            if rho + self.neff < 0.3 * self._cut_lo():
                self.noise += rho       # rank drops: the remaining columns' component along z is noise
            elif 1.0 - h > 1e-8:
                # rank preserved: the column space (fitted values, SSE) is unchanged
                Wm = np.delete(W, j, axis=0)
                tvec = Wm @ bj
                Wm += np.outer(tvec / (1.0 - h), w)
                self.W = Wm
                self.B = np.delete(B, j, axis=1)
                if Wm.shape[0] == Wm.shape[1]:
                    self.B = None
                self.amp *= 1.0 + 2.0 / (1.0 - h)
                self.dref = np.delete(self.dref, j)
                self.t += 1
                self.sse_err = self.sse_err + tj
                self.exact = False
                self.n_updates += 1
                return
            else:
                self._why("ambiguous")
                self.refresh()          # ambiguous rank change: recompute like the reference
                return
        nw = float(np.linalg.norm(w))
        v = w.copy()
        v[-1] += nw if w[-1] >= 0 else -nw
        vv = float(v @ v)
        beta = 2.0 / vv if vv > 0 else 0.0
        tvec = W @ v
        Wn = np.delete(W[:, :-1], j, axis=0)
        Wn -= np.outer(np.delete(tvec, j) * beta, v[:-1])
        self.W = Wn
        c = self.c
        self.c = (c - (beta * float(v @ c)) * v)[:-1]
        if B is not None:
            HB = B - np.outer(v * beta, v @ B)
            self.B = np.delete(HB[:-1], j, axis=1)
            if self.B.shape[0] == self.B.shape[1]:
                self.B = None
        self.dref = np.delete(self.dref, j)
        self.t += 1
        self.sse = self.sse + sj
        self.sse_err = self.sse_err + tj
        self.exact = False
        self.n_updates += 1


    def _drop_kernel(self, j: int, sj: float, tj: float) -> None:
        k = int(self.ind.size)
        if self.Wbuf is None:           # W came from a rank-deficient phase: give it a private buffer
            self.Wbuf = np.array(self.W, order="C")
            self.c = np.array(self.c)
        a = np.empty(k - 1)
        d = np.empty(k - 1)
        self.kernel(self.Wbuf, k, self.c, j, a, d)
        self.W = self.Wbuf[:k - 1, :k - 1]
        self.c = self.c[:k - 1]
        self.ad = (a, d)

        def swap_delete(x: np.ndarray) -> np.ndarray:
            out = x[:k - 1].copy()
            if j != k - 1:
                out[j] = x[k - 1]
            return out

        self.ind = swap_delete(self.ind)
        self.smax_lb2 -= float(self.colnorms[j]) ** 2
        self.colnorms = swap_delete(self.colnorms)
        self.dref = swap_delete(self.dref)
        self.t += 1
        self.sse = self.sse + sj
        self.sse_err = self.sse_err + tj
        self.exact = False
        self.n_updates += 1


def _exact_sse(V: np.ndarray, ind: np.ndarray, y: np.ndarray) -> float:
    Vb = V[:, ind]
    P, _ = pinv_with_svd(Vb)
    E = (Vb @ (P @ y)) - y
    return float(np.sum(E ** 2))


def solve_gene(
    y: np.ndarray,
    basis: SharedBasis,
    params: G2SParams,
    start_ind: Optional[np.ndarray] = None,
    prev_ind: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """gen2spec for one target on the downdate engine, equivalent to the reference loop for
    ``params.drop_rule`` ("exact": gen2spec.m's dropanother; "loo": the LOO-score shortcut).

    ``start_ind``/``prev_ind`` resume the loop from an intermediate basis (``prev_ind`` is the basis
    one step earlier, ``None`` when starting).  Returns (selected indices, coefficients, stats).
    """
    V = basis.V
    N = V.shape[0]
    crit = params.criterion
    cap = params.max_interactions
    min_k = params.min_interactions

    st = _GeneState(V, y, basis.colnorms, params.exact_budget_flops if params.drop_rule == "exact" else None)
    if start_ind is None:
        st.init_full(basis)
    else:
        st.ind = np.sort(np.asarray(start_ind, dtype=np.intp))
        st.refresh()
    ind = st.ind
    k = int(ind.size)

    s_k, s_k_err = st.penalty(k, crit)
    if prev_ind is None:
        s_old, s_old_err = s_k + 1.0, 0.0
        old_ind = ind
        indopt = ind
    else:
        old_ind = np.sort(np.asarray(prev_ind, dtype=np.intp))
        s_old, s_old_err = _penalty_value(_exact_sse(V, old_ind, y), k + 1, N, crit), 0.0
        indopt = old_ind

    def keep_going() -> bool:
        nonlocal s_k, s_k_err, s_old, s_old_err
        if k <= min_k:
            return False
        if cap is not None and k > cap:
            return True
        diff = s_k - s_old
        if abs(diff) > s_k_err + s_old_err + 4.0 * _EPS * max(abs(s_k), abs(s_old)):
            return diff < 0
        if s_k_err > 0.0:
            st._why("penalty")
            st.refresh()
            s_k, s_k_err = st.penalty(k, crit)
        if s_old_err > 0.0:
            s_old = _penalty_value(_exact_sse(V, np.sort(old_ind), y), k + 1, N, crit)
            s_old_err = 0.0
        return s_k < s_old

    while keep_going():
        indopt = st.ind
        old_ind = st.ind
        s_prev, s_prev_err = s_k, s_k_err
        if params.drop_rule == "loo":
            j, sj, tj = st.choose_loo()
            st.drop(j, sj, tj)
        else:
            j, sj, tj, sse_new = st.choose_exact()
            st.drop(j, sj, tj, sse_new)
        k -= 1
        s_k, s_k_err = st.penalty(k, crit)
        s_old, s_old_err = s_prev, s_prev_err

    if cap is not None and len(indopt) > cap:
        indopt = st.ind
    indopt = np.sort(indopt)          # the reference keeps the original column order
    final_a = np.linalg.pinv(V[:, indopt]) @ y
    return (np.asarray(indopt, dtype=int), final_a,
            {"refresh": st.n_refresh, "updates": st.n_updates, "exact_evals": st.n_exact_evals,
             "tie_breaks": st.n_tie_breaks, **st.why})


# ----------------------------------------------------------------------------------------------
# process-pool backend
# ----------------------------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def _worker_init(basis: SharedBasis, Y: np.ndarray, params: G2SParams, starts: Optional[Dict[int, Any]]) -> None:
    from grn_sim.inference._parallel import prepare_worker
    prepare_worker()
    try:
        from threadpoolctl import threadpool_limits
        _WORKER["limits"] = threadpool_limits(limits=1, user_api="blas")
    except Exception:  # pragma: no cover - threadpoolctl missing
        pass
    _WORKER["basis"] = basis
    _WORKER["Y"] = Y
    _WORKER["params"] = params
    _WORKER["starts"] = starts or {}


def _worker_run(genes: Sequence[int]) -> List[Tuple[int, np.ndarray, np.ndarray, Dict[str, int]]]:
    basis = _WORKER["basis"]
    Y = _WORKER["Y"]
    params = _WORKER["params"]
    starts = _WORKER["starts"]
    out = []
    for i in genes:
        s_ind, p_ind = starts.get(i, (None, None))
        ind, a, stats = solve_gene(Y[i], basis, params, s_ind, p_ind)
        out.append((i, ind, a, stats))
    return out


def default_n_jobs() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    return max(1, min(32, n // 2 if n >= 4 else n))


def solve_genes(
    Y: np.ndarray,
    basis: SharedBasis,
    params: G2SParams,
    genes: Sequence[int],
    n_jobs: int = 1,
    starts: Optional[Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
    """Solves the listed target genes (rows of ``Y``), serially or on a process pool."""
    genes = list(genes)
    starts = starts or {}
    results: Dict[int, Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = {}
    n_jobs = max(1, min(int(n_jobs), len(genes))) if genes else 1
    if n_jobs <= 1 or "fork" not in mp.get_all_start_methods():
        with _one_blas_thread():
            for i in genes:
                s_ind, p_ind = starts.get(i, (None, None))
                ind, a, stats = solve_gene(Y[i], basis, params, s_ind, p_ind)
                results[i] = (ind, a, stats)
        return results
    # Small interleaved chunks balance the load (genes differ in how many fallback steps they need).
    n_chunks = min(len(genes), n_jobs * 8)
    chunks = [genes[c::n_chunks] for c in range(n_chunks)]
    _hh_kernel()                      # compile once in the parent; forked workers inherit it
    from grn_sim.inference._native import NATIVE_LOCK
    from grn_sim.inference._parallel import run_chunks
    with NATIVE_LOCK:                 # no other thread loads native libraries while this one forks (see _native.py)
        done, _stalled = run_chunks(_worker_run, chunks, n_jobs, initializer=_worker_init,
                                    initargs=(basis, Y, params, starts))
    parts = [item for k in sorted(done) for item in done[k]]
    missing = [i for k in range(len(chunks)) if k not in done for i in chunks[k]]
    if missing:                       # a stalled pool was killed: solve those genes here, as the serial backend does
        with _one_blas_thread():
            for i in missing:
                s_ind, p_ind = starts.get(i, (None, None))
                ind, a, stats = solve_gene(Y[i], basis, params, s_ind, p_ind)
                parts.append((i, ind, a, stats))
    # Final least squares in the parent with one BLAS thread (as the serial backend runs), so near-zero
    # coefficients do not depend on the backend - only after the pool is gone (see GeneralToSpecific._solve_all).
    with _one_blas_thread():
        for i, ind, _a, stats in parts:
            results[i] = (ind, np.linalg.pinv(basis.V[:, ind]) @ Y[i], stats)
    return results


@contextmanager
def _one_blas_thread() -> Iterator[None]:
    from grn_sim.inference._native import blas_limit
    with blas_limit(1):
        yield
