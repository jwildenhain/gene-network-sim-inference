"""
General-to-Specific (G2S) reverse engineering algorithm.
Implements the exact MATLAB g2s.m and gen2spec.m backward elimination basis pruning
technique with AIC minimization from thesis Chapter 3.

The elimination itself runs on the fast, exact-equivalent engine in
:mod:`grn_sim.inference._g2s_fast` (O(k^2) inverse-Gram-factor downdates with certified decisions
and a reference fallback); ``backend="reference"`` runs the original per-step ``pinv`` loop.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, List, Tuple
import numpy as np

from grn_sim.inference._cache import result_cache
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference import _g2s_fast as fast

_BACKENDS = ("auto", "numpy", "processes", "reference")


@contextmanager
def _single_blas_thread() -> Iterator[None]:
    """Run the elimination with one BLAS/LAPACK thread.

    The many small SVDs are faster single-threaded, and on rank-deficient X the pinv-based
    tie-breaking (like the MATLAB original) depends on the BLAS thread count; fixing it makes all
    backends (process workers use one thread) and repeated runs agree.
    """
    from grn_sim.inference._native import blas_limit
    with blas_limit(1):
        yield


class GeneralToSpecific(BaseInferenceAlgorithm):
    """
    General-to-Specific (G2S) algorithm matching g2s.m and gen2spec.m (Crampin et al.).
    Starts with the full set of basis functions (columns of X) and systematically removes
    the interaction causing the least increase in mean squared error via dropanother()
    until the AIC cost function Caic = sum(E^2) + k stops decreasing.

    Args:
        drop_rule: ``"exact"`` (default) removes the column whose removal gives the smallest pinv
            least-squares residual, evaluated like gen2spec.m's ``dropanother`` (MATLAB-identical, also
            for rank-deficient X, where several candidates tie and pinv rounding picks one).
            ``"loo"`` is the earlier shortcut used for bases wider than ``fast_loo_threshold`` columns:
            score ``SSE + alpha_i^2 / ||pinv row i||^2``, which equals the exact residual only for
            columns outside any linear dependency (identical results on full-column-rank X).
        fast_loo_threshold: with ``drop_rule="loo"``, bases with at most this many columns still use
            the exact rule (the historical behaviour).
        backend: ``"auto"`` (default: process pool for larger networks, serial numpy otherwise),
            ``"numpy"`` (serial), ``"processes"`` (target genes on a fork-based process pool, one BLAS
            thread per worker) or ``"reference"`` (the original per-step ``pinv`` implementation).
            All backends select the same interactions and coefficients.
        n_jobs: worker processes for the process pool (``None``: half the available CPUs, at most 32).
        exact_budget_flops: per-gene budget for the exact pinv residual evaluations of ``drop_rule="exact"``,
            in floating-point work (one evaluation of an N x k basis ~ N*k^2; default 2e9, a few seconds; the
            60-gene MATLAB benchmark needs ~2e8 per gene). Rank-deficient X with many linearly dependent columns makes MATLAB's tie-breaking
            very expensive (hours per gene at 500 genes); past the budget the remaining steps of that gene pick,
            among the columns with the smallest error increase, the lowest original column index. Any tied column
            gives the same fit at that step, but later steps (and so the selected interactions) can differ from
            MATLAB's; the number of
            such steps is reported in ``diagnostics["tie_breaks"]``. ``None`` keeps every step MATLAB-exact.
            Ignored by ``backend="reference"``, which is always exact.
    """

    def __init__(
        self,
        criterion: str = "aic",
        min_interactions: int = 1,
        threshold: float = 0.0,
        matlab_mode: bool = True,
        fast_loo_threshold: int = 25,
        max_interactions: Optional[int] = None,
        drop_rule: str = "exact",
        backend: str = "auto",
        n_jobs: Optional[int] = None,
        exact_budget_flops: Optional[float] = 2e9,
    ):
        super().__init__(name="g2s")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.backend = backend
        self.n_jobs = n_jobs
        if drop_rule not in ("exact", "loo"):
            raise ValueError(f"drop_rule must be 'exact' or 'loo', got {drop_rule!r}")
        self.drop_rule = drop_rule
        if exact_budget_flops is not None and exact_budget_flops < 0:
            raise ValueError(f"exact_budget_flops must be >= 0 or None, got {exact_budget_flops!r}")
        self.exact_budget_flops = exact_budget_flops
        self.criterion = criterion.lower()
        self.min_interactions = min_interactions
        self.threshold = threshold
        self.matlab_mode = matlab_mode
        self.fast_loo_threshold = fast_loo_threshold
        self.max_interactions = max_interactions

    def _penalty(self, E: np.ndarray, k: int) -> float:
        """
        Computes AIC cost function matching gen2spec.m lines 73-75:
            function C = penalty(E, k)
            Caic = sum(E.^2) + k;
            C = Caic;
        """
        rss = float(np.sum(E ** 2))
        if self.criterion == "mdl":
            n_len = len(E)
            mse = max(rss / n_len, 1e-300)
            return float(n_len * np.log(mse) + k * np.log(n_len))
        return rss + float(k)

    def _dropanother_exact(
        self,
        y: np.ndarray,
        V: np.ndarray,
        ind: List[int],
    ) -> Tuple[List[int], np.ndarray]:
        """
        Exact port of dropanother(y, V, ind) in gen2spec.m lines 53-70:
            for i=1:k
               ind0 = ind([1:i-1 i+1:k]);
               Vb = V(:,ind0);
               a = pinv(Vb)*y;
               yh = Vb*a;
               e = yh - y;
               E0(i) = mean(e.^2);
            end
            [E0min, imin] = min(E0);
            ind1 = ind([1:imin-1 imin+1:k]);
        """
        k = len(ind)
        e0 = np.zeros(k, dtype=float)
        for i in range(k):
            ind0 = ind[:i] + ind[i + 1:]
            Vb = V[:, ind0]
            a = np.linalg.pinv(Vb) @ y
            yh = Vb @ a
            e = yh - y
            e0[i] = float(np.mean(e ** 2))

        imin = int(np.argmin(e0))
        ind1 = ind[:imin] + ind[imin + 1:]
        Vb1 = V[:, ind1]
        a1 = np.linalg.pinv(Vb1) @ y
        yh1 = Vb1 @ a1
        E1 = yh1 - y
        return ind1, E1

    def _params(self) -> "fast.G2SParams":
        return fast.G2SParams(criterion=self.criterion, min_interactions=self.min_interactions,
                              max_interactions=self.max_interactions, drop_rule=self.drop_rule,
                              exact_budget_flops=self.exact_budget_flops)

    def _gen2spec(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """[ind, a] = gen2spec(y, X) on the fast engine (same result as ``_gen2spec_reference``)."""
        if self._resolve_backend(1, X.shape[1]) == "reference":
            return self._gen2spec_reference(y, X)
        ind, a, _ = fast.solve_gene(np.asarray(y, dtype=float), fast.make_shared_basis(X), self._params())
        return ind, a

    def _gen2spec_reference(
        self,
        y: np.ndarray,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reference implementation (recomputes pinv after every removed column).
        Exact port of [ind, a] = gen2spec(y, X) in gen2spec.m lines 1-50.
        drop_rule="exact": the dropanother pinv loop for every M (gen2spec.m).  drop_rule="loo": the
        exact loop for M <= fast_loo_threshold and the LOO-score shortcut for wider bases.
        """
        N, M = X.shape
        V = X
        k = M
        ind = list(range(M))
        use_fast = self.drop_rule == "loo" and M > self.fast_loo_threshold

        Vb = V[:, ind]
        Pinv = np.linalg.pinv(Vb)
        alpha = Pinv @ y
        yh = Vb @ alpha
        E = yh - y
        s_k = self._penalty(E, k)
        s_old = s_k + 1.0

        indopt = list(ind)
        cap = self.max_interactions
        while k > self.min_interactions and (s_k < s_old or (cap is not None and k > cap)):
            indopt = list(ind)
            s_prev = s_k
            if not use_fast:
                ind, E = self._dropanother_exact(y, V, ind)
            else:
                row_norms_sq = np.sum(Pinv ** 2, axis=1)
                base_sse = float(np.sum(E ** 2))
                e0 = (base_sse + (alpha ** 2) / np.maximum(row_norms_sq, 1e-30)) / N
                imin = int(np.argmin(e0))
                ind = ind[:imin] + ind[imin + 1:]
                Vb = V[:, ind]
                Pinv = np.linalg.pinv(Vb)
                alpha = Pinv @ y
                E = (Vb @ alpha) - y
            k -= 1
            s_k = self._penalty(E, k)
            s_old = s_prev

        if cap is not None and len(indopt) > cap:
            indopt = list(ind)
        Xb = X[:, indopt]
        final_a = np.linalg.pinv(Xb) @ y
        return np.array(indopt, dtype=int), final_a

    @result_cache
    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        """
        Infers regulatory matrix matching g2s.m / s2g_and_g2s.m:
            Z = X;
            for i=1:N
                [ind1, a1] = gen2spec(U(i,:)', Z);
                A_gs(ind1, i) = a1;
            end
        """
        start_time = time.time()
        X_f = np.real(X).astype(float)
        U_f = np.real(U).astype(float)
        n = X_f.shape[0]
        A_est = np.zeros((n, n), dtype=float)

        # Targets: y_i = U(i,:) (or X(i,:) for an unperturbed row); basis V = X (matlab_mode) or X^T with -y.
        Y = U_f.copy()
        zero_rows = np.all(np.abs(U_f) < 1e-15, axis=1)
        Y[zero_rows] = X_f[zero_rows]
        V = X_f if self.matlab_mode else X_f.T
        if not self.matlab_mode:
            Y = -Y

        diag: Dict[str, Any] = {"criterion": self.criterion, "matlab_mode": self.matlab_mode,
                                "drop_rule": self.drop_rule}
        solved = self._solve_all(Y, V, diag)

        for i in range(n):
            indices, coeffs = solved[i]
            for idx, c in zip(indices, coeffs):
                if abs(c) > self.threshold:
                    if self.matlab_mode:
                        A_est[idx, i] = c
                    else:
                        A_est[i, idx] = c

        adj = np.zeros((n, n), dtype=int)
        adj[A_est > self.threshold] = 1
        adj[A_est < -self.threshold] = -1

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A_est,
            adjacency_pred=adj,
            time_elapsed=elapsed,
            diagnostics=diag,
        )

    def _resolve_backend(self, n_targets: int, M: int) -> str:
        if self.backend == "reference" or (self.drop_rule == "loo" and M <= self.fast_loo_threshold):
            return "reference"
        if self.backend != "auto":
            return self.backend
        # Forking a pool costs ~0.1 s; rank-deficient X needs many exact residual evaluations per gene.
        if n_targets >= 32 and fast.default_n_jobs() > 1:
            return "processes"
        return "numpy"

    def _solve_all(self, Y: np.ndarray, V: np.ndarray, diag: Dict[str, Any]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        n_targets = Y.shape[0]
        M = V.shape[1]
        backend = self._resolve_backend(n_targets, M)
        n_jobs = self.n_jobs if self.n_jobs is not None else fast.default_n_jobs()
        if backend == "reference":
            diag["backend"] = "reference"
            with _single_blas_thread():
                return {i: self._gen2spec_reference(Y[i].copy(), V) for i in range(n_targets)}

        with _single_blas_thread():
            basis = fast.make_shared_basis(V)
        params = self._params()
        genes = list(range(n_targets))
        if backend == "processes":
            # Fork the pool OUTSIDE any BLAS thread limit: forking while the parent's OpenBLAS is limited and
            # restoring the limit afterwards leaves SciPy's bundled OpenBLAS (0.3.28, SciPy 1.15) broken in the
            # parent - the next SciPy call that uses it segfaults. Workers limit themselves; solve_genes runs the
            # parent's final least squares with one thread so all backends give identical coefficients.
            res = fast.solve_genes(Y, basis, params, genes, n_jobs=n_jobs)
        else:
            with _single_blas_thread():
                res = fast.solve_genes(Y, basis, params, genes, n_jobs=1)
        diag["backend"] = backend
        diag["n_jobs"] = n_jobs if backend == "processes" else 1
        diag["refresh_steps"] = int(sum(r[2]["refresh"] for r in res.values()))
        diag["downdate_steps"] = int(sum(r[2]["updates"] for r in res.values()))
        diag["exact_residual_evals"] = int(sum(r[2]["exact_evals"] for r in res.values()))
        diag["tie_breaks"] = int(sum(r[2].get("tie_breaks", 0) for r in res.values()))
        diag["matlab_exact"] = self.drop_rule == "exact" and diag["tie_breaks"] == 0
        return {i: (res[i][0], res[i][1]) for i in genes}
