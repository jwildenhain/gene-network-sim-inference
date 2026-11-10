"""
Statistical generative models of perturbation data.

Panel 1 of the studio produces data mechanistically: Hill kinetics solved to a steady state. The models here produce data the
way causal-discovery and single-cell benchmarks usually do, from a *statistical* model of the graph, without any kinetics:

* :class:`LinearSEM` - linear structural equation model in log expression, ``z = z W + delta + eps`` (Gaussian noise that
  propagates along the edges, like GIES / DCDI / inspre benchmarks);
* :class:`NonlinearSCM` - additive-noise structural causal model, ``z_k = sum_j W_jk g_jk(z_j) + delta_k + eps_k`` with
  saturating, shifted ``tanh`` link functions (the DCDI / CausalBench synthetic style), solved as a fixed point;
* :class:`NBCopula` - log-linear means with propagated perturbation effects and negative-binomial (Gamma-Poisson) cell
  variation whose residual correlation comes from a Gaussian copula of the graph (scDesign3 / SPARSim style count model);
  the noise does *not* travel along edges, only the perturbation does;
* :class:`ProbabilisticBoolean` - every gene is on or off by a threshold rule over its regulators, with random flips
  (probabilistic Boolean network); expression is the time-averaged on-fraction times a per-gene level.

``z`` is the natural-log expression relative to the gene's baseline. A perturbation of gene ``a`` with dose ``d`` (the share of
the gene's production removed by the guide) adds ``delta_a = log(1 - d)`` for CRISPRi knockdowns and knockouts (knockouts
floor at 2% residual production) and ``log(1 + 2 d)`` for activation (CRISPRa). The Boolean model forces the gene off (on)
with probability ``d`` instead.

All models expose the same three calls: ``log_mean(targets, doses)`` (noise-free log expression, cells x genes),
``sample(targets, doses, rng)`` (one cell per dose row) and ``baseline`` (natural-scale expression of an unperturbed cell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Type

import numpy as np

KNOCKOUT_RESIDUAL = 0.02          # production left after a full knockout
ACTIVATION_GAIN = 2.0             # CRISPRa at full dose multiplies production by 1 + gain
MAX_SPECTRAL_RADIUS = 0.8         # |W| is rescaled below this, so every model has a unique, stable solution


@dataclass(frozen=True)
class ModelParams:
    """What every statistical model is tuned by (see ``grn_sim.statgen.calibration`` for the chosen values)."""

    effect_scale: float = 0.6       # typical |edge weight| (log-expression change of the target per unit log change of the regulator)
    noise: float = 0.3              # intrinsic cell-to-cell noise (log-sd; NB dispersion; Boolean flip probability x 10)
    shared_noise: float = 0.1       # log-sd of a factor shared by all genes of a cell (cell size, cycle)
    baseline_sd: float = 1.0        # log-sd of the baseline expression across genes
    weight_spread: float = 1.1      # log-sd of |edge weight| around effect_scale (heavy-tailed effects, a few strong edges)


def shift(mode: str, doses: np.ndarray) -> np.ndarray:
    """Additive log-production shift of a perturbation of the given doses."""
    d = np.clip(np.asarray(doses, dtype=float), 0.0, 1.0)
    if mode == "input":
        return np.log1p(ACTIVATION_GAIN * d)
    floor = KNOCKOUT_RESIDUAL if mode == "knockout" else 0.0
    return np.log(np.maximum(1.0 - d * (1.0 - floor), max(floor, 1e-3)))


def edge_weights(adjacency: np.ndarray, effect_scale: float, rng: np.random.Generator, spread: float = 0.8) -> np.ndarray:
    """Signed weights on the graph's edges: |w| ~ effect_scale x LogNormal(0, spread), damped by sqrt(in-degree), |W| < 0.8."""
    A = np.sign(np.asarray(adjacency, dtype=float))
    np.fill_diagonal(A, 0.0)
    mag = effect_scale * np.exp(rng.normal(0.0, spread, size=A.shape) - spread ** 2 / 2)
    indeg = np.maximum((A != 0).sum(axis=0), 1)
    W = A * mag / np.sqrt(indeg)[None, :]
    rho = float(np.max(np.abs(np.linalg.eigvals(np.abs(W))))) if W.any() else 0.0
    if rho > MAX_SPECTRAL_RADIUS:
        W *= MAX_SPECTRAL_RADIUS / rho
    return W


class StatModel:
    """Base class: the graph, a baseline expression per gene and the perturbation mode."""

    name = "base"
    label = "Statistical model"
    noise_unit = "log-sd"

    def __init__(self, adjacency: np.ndarray, params: ModelParams, mode: str, rng: np.random.Generator):
        self.adjacency = np.asarray(adjacency)
        self.n = self.adjacency.shape[0]
        self.params = params
        self.mode = mode
        self.W = edge_weights(self.adjacency, params.effect_scale, rng, params.weight_spread)
        self.log_base = rng.normal(0.0, params.baseline_sd, size=self.n)
        gene_sd = params.noise * np.exp(rng.normal(0.0, 0.3, size=self.n))      # genes differ in how noisy they are
        self.gene_sd = gene_sd

    @property
    def baseline(self) -> np.ndarray:
        return np.exp(self.log_base + self.log_mean((), np.zeros((1, 0)))[0])

    def _delta(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:
        doses = np.atleast_2d(np.asarray(doses, dtype=float))
        D = np.zeros((doses.shape[0], self.n))
        for t, g in enumerate(targets):
            D[:, g] += shift(self.mode, doses[:, t])
        return D

    def log_mean(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:        # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def refresh(self) -> None:
        """Recomputes the matrices derived from ``W`` (after the weights were changed in place)."""

    def local_effects(self) -> np.ndarray:
        """d z_k / d z_j along each edge at the baseline (the model's edge elasticities)."""
        return self.W.copy()

    def linear_response(self) -> np.ndarray:
        """Total effect of a unit log shift of gene j on gene i at the baseline, [i, j]."""
        return np.linalg.inv(np.eye(self.n) - self.local_effects()).T


class LinearSEM(StatModel):
    name = "linear_sem"
    label = "Linear SEM (Gaussian, propagated noise)"

    def __init__(self, adjacency: np.ndarray, params: ModelParams, mode: str, rng: np.random.Generator):
        super().__init__(adjacency, params, mode, rng)
        self.refresh()

    def refresh(self) -> None:
        self.M = np.linalg.inv(np.eye(self.n) - self.W)          # z = (delta + eps) M

    def log_mean(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:
        return self._delta(targets, doses) @ self.M

    def sample(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        D = self._delta(targets, doses)
        eps = rng.standard_normal(D.shape) * self.gene_sd[None, :]
        z = (D + eps) @ self.M + rng.normal(0.0, self.params.shared_noise, size=(D.shape[0], 1))
        return np.exp(self.log_base + z)


class NonlinearSCM(StatModel):
    name = "nonlinear_scm"
    label = "Nonlinear additive-noise SCM"

    def __init__(self, adjacency: np.ndarray, params: ModelParams, mode: str, rng: np.random.Generator):
        super().__init__(adjacency, params, mode, rng)
        self.beta = rng.uniform(0.5, 3.0, size=self.n)             # steepness of each regulator's link
        self.theta = rng.normal(0.0, 0.5, size=self.n)             # where the link saturates

    def _g(self, z: np.ndarray) -> np.ndarray:
        b, t = self.beta[None, :], self.theta[None, :]
        return (np.tanh(b * (z - t)) - np.tanh(-b * t)) / b        # g(0) = 0, slope <= 1: a contraction with |W| < 1

    def _solve(self, D: np.ndarray, eps: np.ndarray) -> np.ndarray:
        z = D + eps
        for _ in range(200):
            new = self._g(z) @ self.W + D + eps
            if np.max(np.abs(new - z)) < 1e-8:
                return new
            z = 0.5 * z + 0.5 * new
        return z

    def log_mean(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:
        D = self._delta(targets, doses)
        return self._solve(D, np.zeros_like(D))

    def sample(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        D = self._delta(targets, doses)
        z = self._solve(D, rng.standard_normal(D.shape) * self.gene_sd[None, :])
        return np.exp(self.log_base + z + rng.normal(0.0, self.params.shared_noise, size=(D.shape[0], 1)))

    def local_effects(self) -> np.ndarray:
        slope = 1.0 - np.tanh(-self.beta * self.theta) ** 2          # g'(0)
        return self.W * slope[:, None]


class NBCopula(StatModel):
    name = "nb_copula"
    label = "Negative-binomial copula (log-linear means)"
    noise_unit = "NB dispersion"

    def __init__(self, adjacency: np.ndarray, params: ModelParams, mode: str, rng: np.random.Generator):
        super().__init__(adjacency, params, mode, rng)
        self.refresh()
        self.dispersion = np.maximum(params.noise ** 2 * np.exp(rng.normal(0.0, 0.5, size=self.n)), 1e-4)

    def refresh(self) -> None:
        self.M = np.linalg.inv(np.eye(self.n) - self.W)
        cov = self.M.T @ self.M                                      # residual correlation implied by the graph
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
        self.chol = np.linalg.cholesky(corr + 1e-9 * np.eye(self.n))

    def log_mean(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:
        return self._delta(targets, doses) @ self.M

    def sample(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        from scipy.special import ndtr
        from scipy.stats import gamma

        mu = np.exp(self.log_base + self.log_mean(targets, doses))
        u = np.clip(ndtr(rng.standard_normal(mu.shape) @ self.chol.T), 1e-9, 1 - 1e-9)
        shape = 1.0 / self.dispersion[None, :]
        rate = gamma.ppf(u, a=shape, scale=mu / shape)                # Gamma marginal with mean mu, dispersion phi
        return rate * np.exp(rng.normal(0.0, self.params.shared_noise, size=(mu.shape[0], 1)))


class ProbabilisticBoolean(StatModel):
    """
    Stochastic logistic Boolean network: at every sweep gene k is on with probability
    ``sigmoid(b_k + gain * sum_j w_jk s_j)`` (``w`` signed, heavy-tailed weights; ``gain = 4 x effect_scale``; ``b_k`` puts
    the unperturbed on-probability near a gene-specific basal level), then flips with probability ``noise / 10``. Expression is
    the on-fraction over the last sweeps times a per-gene level (an off gene keeps 3%). A guide forces its gene off (on for
    activation) with probability equal to its dose and scales the gene's own transcript as CRISPRi / CRISPRa does.
    """

    name = "boolean"
    label = "Probabilistic Boolean network"
    noise_unit = "flip probability x 10"

    SWEEPS = 40
    AVERAGE = 20
    LOW = 0.03                      # expression of an off gene relative to on

    def __init__(self, adjacency: np.ndarray, params: ModelParams, mode: str, rng: np.random.Generator):
        super().__init__(adjacency, params, mode, rng)
        A = np.sign(np.asarray(adjacency, dtype=float))
        np.fill_diagonal(A, 0.0)
        self.A = A
        indeg = np.maximum((A != 0).sum(axis=0), 1)
        spread = params.weight_spread
        self.Wb = A * np.exp(rng.normal(0.0, spread, size=A.shape) - spread ** 2 / 2) / np.sqrt(indeg)[None, :]
        self.gain = 4.0 * params.effect_scale
        basal = rng.uniform(0.3, 0.8, size=self.n)
        expected_input = 0.5 * self.Wb.sum(axis=0)                     # inputs when every regulator is on half the time
        self.bias = np.log(basal / (1.0 - basal)) - self.gain * expected_input
        self.flip = float(np.clip(params.noise / 10.0, 0.0, 0.45))
        self._seed = int(rng.integers(0, 2 ** 31 - 1))
        self._mean_cache: Dict[tuple, np.ndarray] = {}

    def _run(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        doses = np.atleast_2d(np.asarray(doses, dtype=float))
        k = doses.shape[0]
        s = rng.uniform(size=(k, self.n)) < 0.5
        acc = np.zeros((k, self.n))
        for sweep in range(self.SWEEPS):
            # The same random numbers are drawn whatever is perturbed (common random numbers): a gene that does not depend
            # on the perturbed one then follows exactly the same path as in the control, so it shows no effect at all.
            u_on, u_flip, u_hit = rng.uniform(size=(3, k, self.n))
            p_on = 1.0 / (1.0 + np.exp(-(self.bias[None, :] + self.gain * (s.astype(float) @ self.Wb))))
            new = u_on < p_on
            new ^= u_flip < self.flip
            for t, g in enumerate(targets):
                hit = u_hit[:, g] < doses[:, t]
                new[hit, g] = self.mode == "input"
            s = new
            if sweep >= self.SWEEPS - self.AVERAGE:
                acc += s
        return acc / self.AVERAGE                                     # on-fraction over the last sweeps

    def _expr(self, frac: np.ndarray, targets: Sequence[int] = (), doses: Optional[np.ndarray] = None) -> np.ndarray:
        e = self.LOW + (1.0 - self.LOW) * frac
        if doses is not None:
            for t, g in enumerate(targets):
                e[:, g] *= np.exp(shift(self.mode, doses[:, t]))      # the guide also acts on the gene's own transcript
        return e

    def log_mean(self, targets: Sequence[int], doses: np.ndarray) -> np.ndarray:
        """Expected log expression, by Monte Carlo over 400 noise realisations per distinct dose (deterministic per model)."""
        doses = np.atleast_2d(np.asarray(doses, dtype=float))
        out = np.empty((doses.shape[0], self.n))
        ref = self._mc((), np.zeros(0))
        for i, row in enumerate(doses):
            out[i] = self._mc(tuple(targets), row) - ref
        return out

    def _mc(self, targets: tuple, dose: np.ndarray) -> np.ndarray:
        key = (targets, tuple(np.round(dose, 3)))
        if key not in self._mean_cache:
            rng = np.random.default_rng(self._seed)            # the same stream for every condition (see _run)
            D = np.tile(dose, (400, 1)) if len(targets) else np.zeros((400, 0))
            frac = self._run(targets, D, rng)
            self._mean_cache[key] = np.log(self._expr(frac, targets, D).mean(axis=0))
        return self._mean_cache[key]

    def sample(self, targets: Sequence[int], doses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        doses = np.atleast_2d(np.asarray(doses, dtype=float))
        frac = self._run(targets, doses, rng)
        lognoise = rng.normal(0.0, 0.15, size=frac.shape) + rng.normal(0.0, self.params.shared_noise, size=(frac.shape[0], 1))
        return np.exp(self.log_base) * self._expr(frac, targets, doses) * np.exp(lognoise)

    @property
    def baseline(self) -> np.ndarray:
        return np.exp(self.log_base + self._mc((), np.zeros(0)))

    def local_effects(self) -> np.ndarray:
        return self.gain * 0.25 * self.Wb                              # slope of the logistic at p = 1/2


def jitter_weights(model: StatModel, sd: float, rng: np.random.Generator) -> StatModel:
    """
    Multiplies every edge weight by ``exp(N(0, sd))`` in place (a related cell context: same wiring, different strengths),
    keeps |W| below the stability bound and refreshes the derived matrices. Returns the model.
    """
    factor = np.exp(rng.normal(0.0, sd, size=model.W.shape))
    model.W = model.W * factor
    rho = float(np.max(np.abs(np.linalg.eigvals(np.abs(model.W))))) if model.W.any() else 0.0
    if rho > MAX_SPECTRAL_RADIUS:
        model.W *= MAX_SPECTRAL_RADIUS / rho
    if isinstance(model, ProbabilisticBoolean):
        model.Wb = model.Wb * factor
        model._mean_cache.clear()
    model.refresh()
    return model


MODELS: Dict[str, Type[StatModel]] = {m.name: m for m in (LinearSEM, NonlinearSCM, NBCopula, ProbabilisticBoolean)}


def make_model(name: str, adjacency: np.ndarray, params: ModelParams, mode: str = "knockdown",
               rng: Optional[np.random.Generator] = None) -> StatModel:
    if name not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {name!r}")
    if mode not in ("knockdown", "knockout", "input"):
        raise ValueError("mode must be knockdown, knockout or input")
    return MODELS[name](adjacency, params, mode, rng or np.random.default_rng())
