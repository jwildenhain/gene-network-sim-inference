"""
Predicting the response to perturbations a model has never observed.

Why this module exists
----------------------
The Gardner-style linear model ``A X = -U`` predicts the response to perturbing gene j as the j-th column of
``-A^-1``. The row of ``A`` that belongs to gene j is only identified by the experiment that perturbs gene j, so
when that experiment is held out the row is all zero and the prediction is *exactly* zero. A second problem is
model mismatch: the simulator applies a perturbation to the *effective regulator concentration* ``x + u`` of
nonlinear Hill kinetics, and even a linear regulator-input model ``X = F (X + U)`` with the true structure can
only reach r of about 0.55 in-sample.

Strategy implemented here ("structure -> Hill refit -> steady state"), usable with the output of *any* algorithm:

1. take the algorithm's score matrix, keep the ``q`` best regulators of every target, and choose the edge
   orientation (score matrix or its transpose) that explains the *training* data better;
2. refit a Hill-form model per target with nonlinear least squares on the training experiments only,
   ``x_k = c_k * prod_l h(x_l + u_l; K_l, m_l, sign_l)`` (activator ``1 + e^m/(e^m + K^m)``, inhibitor
   ``K^m/(e^m + K^m)``, the same functional forms as the simulator), with the edge signs taken from a linear refit;
3. predict the response to any perturbation by solving the fixed point ``x = g(x + u)`` (damped iteration).

Held-out columns are never used in any of these steps: the outgoing edges of a held-out gene are learned from how
the gene's activity varies in the *other* experiments.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

K_BOUNDS = (0.05, 20.0)   # Hill half-saturation
M_BOUNDS = (0.5, 8.0)     # Hill coefficient
Y_FLOOR = 1e-6
KNOCKDOWN_FLOOR = 1e-3  # smallest production multiplier (efficiency 0.999)
MODES = ("input", "knockdown")


def design(x: np.ndarray, u: np.ndarray, x_ctrl: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regulator inputs and production-normalised levels of the training experiments plus the control, both (n, m + 1).
    ``mode="input"``: the perturbation is added to the regulator concentration, ``eff = x + u``.
    ``mode="knockdown"``: ``u[j, j] = -efficiency`` scales the production of gene j by ``1 + u``; regulators see the
    plain levels and the targeted gene's level is divided by its scale so that ``x_k / scale_k = c_k * prod h(x_l)``.
    """
    ctrl = np.asarray(x_ctrl, float)[:, None]
    if mode == "knockdown":
        scale = np.maximum(1.0 + u, KNOCKDOWN_FLOOR)
        return np.concatenate([x, ctrl], axis=1), np.concatenate([x / scale, ctrl], axis=1)
    return np.concatenate([x + u, ctrl], axis=1), np.concatenate([x, ctrl], axis=1)


def hill_factor(eff, K, m, sign):
    """Hill regulation factor, broadcast over arrays: activator for ``sign > 0``, inhibitor otherwise."""
    eff = np.maximum(eff, 1e-9)
    em = np.power(eff, m)
    Km = np.power(K, m)
    return np.where(np.asarray(sign) > 0, 1.0 + em / (em + Km), Km / (em + Km))


# ---------------------------------------------------------------------------------------- structure
def top_q_support(scores: np.ndarray, q: int) -> np.ndarray:
    """``scores[l, k]`` = evidence for regulator l -> target k. Keeps the q best regulators of every target (no self edges)."""
    n = scores.shape[0]
    work = np.array(scores, dtype=float)
    np.fill_diagonal(work, -1.0)
    support = np.zeros((n, n), dtype=int)
    for k in range(n):
        for l in np.argsort(-work[:, k])[:q]:
            if work[l, k] > 0:
                support[l, k] = 1
    return support


def linear_refit(support: np.ndarray, X: np.ndarray, U: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """
    Least-squares ``F[k, l]`` (regulator l -> target k) of the regulator-input model ``X[k] = sum_l F[k, l] (X + U)[l]``
    restricted to ``support``. Columns that are entirely zero (masked experiments) are ignored.
    """
    n = X.shape[0]
    Z = X + U
    live = np.any(Z != 0, axis=0)
    F = np.zeros((n, n))
    for k in range(n):
        parents = np.where(support[:, k] != 0)[0]
        if len(parents) == 0 or not live.any():
            continue
        design = Z[parents][:, live].T
        F[k, parents] = np.linalg.solve(design.T @ design + ridge * np.eye(len(parents)), design.T @ X[k, live])
    return F


def infer_signed_structure(A_est, adjacency_pred, X_train, U_train, q: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Signed regulator->target support (``[l, k]`` in {-1, 0, 1}) and the linear refit ``F`` for one algorithm's output.
    Both orientations of the score matrix are tried; the one whose top-q support explains the TRAINING data
    better is kept, because the algorithms do not all report edges in the same orientation.
    """
    scores = np.abs(np.nan_to_num(np.asarray(A_est, float)))
    if adjacency_pred is not None:
        scores = scores + 1e-6 * np.abs(np.asarray(adjacency_pred, float))
    best = None
    for candidate in (scores, scores.T):
        support = top_q_support(candidate, q)
        F = linear_refit(support, X_train, U_train)
        error = float(np.linalg.norm(X_train - F @ (X_train + U_train)))
        if best is None or error < best[0]:
            best = (error, support, F)
    _, support, F = best
    return (np.sign(F.T) * support).astype(int), F


# ---------------------------------------------------------------------------------------- Hill refit
def _fit_target(y: np.ndarray, eff: np.ndarray, signs: np.ndarray):
    """Fit (log c, K, m) of one target; ``y`` (T,), ``eff`` (P, T) regulator effective concentrations."""
    P = len(signs)
    log_y = np.log(np.maximum(y, Y_FLOOR))
    if P == 0:
        return float(np.mean(log_y)), np.zeros(0), np.zeros(0), float(np.sum((log_y - np.mean(log_y)) ** 2))

    def residual(theta):
        K, m = theta[:P], theta[P:]
        log_h = sum(np.log(hill_factor(eff[i], K[i], m[i], signs[i])) for i in range(P))
        return log_y - log_h - np.mean(log_y - log_h)  # log c is solved in closed form

    start = np.r_[np.maximum(np.median(eff, axis=1), 0.2), np.full(P, 2.0)]
    lower = np.r_[np.full(P, K_BOUNDS[0]), np.full(P, M_BOUNDS[0])]
    upper = np.r_[np.full(P, K_BOUNDS[1]), np.full(P, M_BOUNDS[1])]
    try:
        theta = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper), max_nfev=200).x
    except (ValueError, FloatingPointError):
        theta = np.clip(start, lower, upper)
    K, m = theta[:P], theta[P:]
    log_h = sum(np.log(hill_factor(eff[i], K[i], m[i], signs[i])) for i in range(P))
    log_c = float(np.mean(log_y - log_h))
    return log_c, K, m, float(np.sum((log_y - log_h - log_c) ** 2))


class HillResponseModel:
    """Per-target Hill refit on a signed regulator->target structure; predicts steady-state responses to new perturbations."""

    def __init__(self, signed_support: np.ndarray, x_train: np.ndarray, u_train: np.ndarray, x_ctrl: np.ndarray,
                 mode: str = "input"):
        """
        ``signed_support[l, k]``: regulator l -> target k (+1 activator, -1 inhibitor, 0 none).
        ``x_train`` / ``u_train``: absolute expression and perturbation of the TRAINING experiments (columns);
        ``x_ctrl``: unperturbed steady state (also used as one more training point);
        ``mode``: how ``u`` acts, see ``design``.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.n = signed_support.shape[0]
        self.x_ctrl = np.maximum(np.asarray(x_ctrl, float), Y_FLOOR)
        eff, levels = design(x_train, u_train, self.x_ctrl, mode)

        fits = []
        for k in range(self.n):
            parents = np.where(signed_support[:, k] != 0)[0]
            fits.append((parents, _fit_target(levels[k], eff[parents], signed_support[parents, k])[:3], signed_support[parents, k]))

        width = max(1, max(len(p) for p, _, _ in fits))
        self._idx = np.zeros((self.n, width), dtype=int)
        self._valid = np.zeros((self.n, width), dtype=bool)
        self._sign = np.ones((self.n, width))
        self._K = np.ones((self.n, width))
        self._m = np.ones((self.n, width))
        self._log_c = np.zeros(self.n)
        for k, (parents, (log_c, K, m), signs) in enumerate(fits):
            p = len(parents)
            self._idx[k, :p], self._valid[k, :p], self._sign[k, :p] = parents, True, signs
            self._K[k, :p], self._m[k, :p], self._log_c[k] = K, m, log_c

    def production(self, eff: np.ndarray) -> np.ndarray:
        """Steady-state level of every target given the effective regulator concentrations ``eff``."""
        factors = hill_factor(eff[self._idx], self._K, self._m, self._sign)
        return np.exp(self._log_c) * np.where(self._valid, factors, 1.0).prod(axis=1)

    def steady_state(self, u: np.ndarray, iterations: int = 800, damping: float = 0.5, tol: float = 1e-8) -> np.ndarray:
        x = self.x_ctrl.copy()
        scale = np.maximum(1.0 + u, KNOCKDOWN_FLOOR) if self.mode == "knockdown" else 1.0
        for _ in range(iterations):
            target = scale * self.production(x) if self.mode == "knockdown" else self.production(x + u)
            nxt = (1.0 - damping) * x + damping * target
            if np.max(np.abs(nxt - x)) < tol:
                return nxt
            x = nxt
        return x

    def response(self, u: np.ndarray) -> np.ndarray:
        """Relative change ``x / x_ctrl - 1`` after perturbation ``u`` (the repository's response convention)."""
        return self.steady_state(np.asarray(u, float)) / self.x_ctrl - 1.0

    def response_matrix(self, U: np.ndarray) -> np.ndarray:
        """Column j = predicted response to perturbation ``U[:, j]``."""
        return np.column_stack([self.response(U[:, j]) for j in range(U.shape[1])])


def hill_forward_selection(
    x_train: np.ndarray,
    u_train: np.ndarray,
    x_ctrl: np.ndarray,
    max_parents: int = 3,
    screen: int = 6,
    min_gain: float = 2.0,
    mode: str = "input",
) -> np.ndarray:
    """
    Structure search matched to the model class: for every target, regulators (and their activator/inhibitor sign)
    are added greedily while the Hill-form fit improves the BIC by more than ``min_gain``. Candidates are first
    screened by the correlation of their (log) effective concentration with the target's (log) level, so at most
    ``screen`` regulators are fitted per step. Returns ``signed[l, k]``.
    """
    n = x_train.shape[0]
    eff, levels = design(x_train, u_train, x_ctrl, mode)
    T = eff.shape[1]
    log_eff = np.log(np.maximum(eff, 1e-9))
    signed = np.zeros((n, n), dtype=int)
    for k in range(n):
        y = np.maximum(levels[k], Y_FLOOR)
        log_y = np.log(y)
        if np.std(log_y) < 1e-9:
            continue
        corr = [abs(pearson(log_eff[l], log_y)) if l != k else -1.0 for l in range(n)]
        candidates = [int(l) for l in np.argsort(corr)[::-1][:screen] if corr[l] > 0]
        chosen, chosen_signs = [], []
        best_bic = T * np.log(max(np.sum((log_y - log_y.mean()) ** 2) / T, 1e-12))
        while len(chosen) < max_parents:
            step = None
            for l in candidates:
                if l in chosen:
                    continue
                for sign in (1, -1):
                    P = len(chosen) + 1
                    _, _, _, sse = _fit_target(y, eff[chosen + [l]], np.array(chosen_signs + [sign]))
                    bic = T * np.log(max(sse / T, 1e-12)) + (1 + 2 * P) * np.log(T)
                    if step is None or bic < step[0]:
                        step = (bic, l, sign)
            if step is None or step[0] > best_bic - min_gain:
                break
            best_bic = step[0]
            chosen.append(step[1])
            chosen_signs.append(step[2])
        for l, sign in zip(chosen, chosen_signs):
            signed[l, k] = sign
    return signed


def mean_profile_baseline(X: np.ndarray, train_columns) -> np.ndarray:
    """
    The no-structure null: predict every perturbation with the mean response profile of the training columns
    (leave-one-out for the training columns themselves).
    """
    cols = list(train_columns)
    total = X[:, cols].sum(axis=1)
    R = np.tile((total / max(1, len(cols)))[:, None], (1, X.shape[1]))
    if len(cols) > 1:
        for j in cols:
            R[:, j] = (total - X[:, j]) / (len(cols) - 1)
    return R


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
