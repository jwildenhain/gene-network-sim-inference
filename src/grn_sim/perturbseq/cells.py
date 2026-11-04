"""
Stage 2 of the Perturb-seq generator: cell-level biology.

Each simulated cell differs from the population mean of its condition in two ways that real Perturb-seq cells do:

1. **Perturbation strength.** Every gene is targeted by ``guides_per_gene`` guides with their own mean efficiency;
   every cell draws its own efficiency around its guide's mean, and a fraction of cells escape (no effective
   perturbation). Knockouts are biallelic (full loss), monoallelic (half dose) or escaped. The steady state at the
   cell's own dose is interpolated (in log space) from a small grid of solved doses, so per-cell doses cost no extra
   ODE solves.
2. **Network-propagated noise.** Each cell has its own production rates (per-gene lognormal noise plus one shared
   "cell size / state" factor), and optionally intrinsic chemical noise of system size ``omega``. Both propagate
   through the regulatory network, so genes co-vary within a condition along the network's edges - information a
   cell-level model can use and a pseudobulk model cannot.

Two interchangeable noise models implement the same biology:

- ``"lna"`` (default, fast): linear-noise approximation around each condition's steady state. Extrinsic noise
  enters through the linear response ``R = -diag(1/x) J^-1 diag(s)``; intrinsic noise through the Lyapunov equation
  ``J C + C J^T + diag(s + Bx)/omega = 0``.
- ``"sde"`` (exact, slower): chemical-Langevin integration of every cell (Euler-Maruyama) with the cell's own
  production rates and dose, started from its interpolated steady state.

The result is the noise-free-of-measurement expression of every cell (``CellPopulation.expression``); stage 3 turns
it into UMI counts. ``population_to_counts`` provides a simple Poisson/library-size readout until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.perturbseq.conditions import ConditionScreen
from grn_sim.simulator.kinetics import KineticsKernel
from grn_sim.simulator.steady_state import SteadyStateSolver

NOISE_MODELS = ("lna", "sde")
FLOOR = 1e-9
LNA_MAX_LOG_DEV = 6.0   # largest noise-driven log deviation the LNA may produce (~400-fold)


@dataclass
class CellConfig:
    """Parameters of the cell-level biology (defaults: a typical CRISPRi screen)."""

    noise_model: str = "lna"
    guides_per_gene: int = 2
    guide_efficiency_mean: float = 0.8        # mean of the per-guide mean efficiencies (knockdown)
    guide_efficiency_concentration: float = 15.0  # Beta concentration of guide means around that mean
    cell_efficiency_concentration: float = 12.0   # Beta concentration of cell efficiencies around their guide mean
    escape_rate: float = 0.1                  # fraction of perturbed cells without effective perturbation
    knockout_biallelic_rate: float = 0.8      # knockout: share of non-escaped cells with full loss (rest: half dose)
    dose_levels: Tuple[float, ...] = (0.0, 0.5, 0.75, 0.9, 1.0)  # knockouts only ever need (0, 0.5, 1)
    extrinsic_gene_sd: float = 0.15           # log-sd of per-gene production rates between cells
    extrinsic_shared_sd: float = 0.1          # log-sd of a production factor shared by all genes of a cell
    omega: Optional[float] = None             # system size for intrinsic noise (None: no intrinsic noise)
    sde_dt: float = 0.02
    sde_time: float = 15.0
    n_control_guides: int = 4                 # distinct non-targeting guides for the control cells
    blas_threads: Optional[int] = None        # BLAS threads while simulating; None keeps the process limit (1, see grn_sim/__init__)

    def validate(self) -> None:
        if self.noise_model not in NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {NOISE_MODELS}, got {self.noise_model!r}")
        if self.guides_per_gene < 1 or self.n_control_guides < 1:
            raise ValueError("guides_per_gene and n_control_guides must be >= 1")
        if not 0.0 < self.guide_efficiency_mean < 1.0:
            raise ValueError("guide_efficiency_mean must be in (0, 1)")
        for name in ("escape_rate", "knockout_biallelic_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.guide_efficiency_concentration <= 0 or self.cell_efficiency_concentration <= 0:
            raise ValueError("Beta concentrations must be > 0")
        levels = np.asarray(self.dose_levels, dtype=float)
        if levels.size < 2 or levels[0] != 0.0 or levels[-1] != 1.0 or np.any(np.diff(levels) <= 0):
            raise ValueError("dose_levels must increase strictly from 0.0 to 1.0")
        if self.blas_threads is not None and self.blas_threads < 1:
            raise ValueError("blas_threads must be >= 1 (or None for the library default)")
        if self.extrinsic_gene_sd < 0 or self.extrinsic_shared_sd < 0:
            raise ValueError("extrinsic noise levels must be >= 0")
        if self.omega is not None and self.omega <= 0:
            raise ValueError("omega must be > 0 (or None for no intrinsic noise)")


@dataclass
class CellPopulation:
    """Simulated cells: expression (cells x genes, concentration units) plus per-cell metadata."""

    gene_names: List[str]
    expression: np.ndarray
    condition: List[str]            # condition name per cell ("control" for non-targeting cells)
    guide: List[str]                # guide identity per cell
    targets: List[Tuple[int, ...]]  # perturbed gene indices per cell
    dose: np.ndarray                # (cells, 2) realised dose of the first / second target (0 for none)
    escaped: np.ndarray             # (cells,) True if every targeted gene escaped
    noise_model: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.expression.shape[0])

    def obs_records(self) -> List[Dict[str, Any]]:
        return [
            {
                "condition": self.condition[c],
                "guide": self.guide[c],
                "n_perturbed_genes": len(self.targets[c]),
                "dose_1": float(self.dose[c, 0]),
                "dose_2": float(self.dose[c, 1]),
                "escaped": bool(self.escaped[c]),
            }
            for c in range(self.n_cells)
        ]


def _beta(rng: np.random.Generator, mean: np.ndarray, concentration: float, size: Any = None) -> np.ndarray:
    mean = np.clip(mean, 1e-3, 1 - 1e-3)
    return rng.beta(mean * concentration, (1 - mean) * concentration, size=size)


class _Perturber:
    """Builds the production scale / regulator input of a set of targets at given doses."""

    def __init__(self, screen: ConditionScreen):
        self.mode = screen.mode
        self.n = screen.n_genes
        self.factor = np.asarray(screen.efficiency, dtype=float)  # input mode: full regulator input per gene

    def inputs(self, targets: Sequence[int], doses: Sequence[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not targets:
            return None, None
        if self.mode == "input":
            u = np.zeros(self.n)
            for g, d in zip(targets, doses):
                u[g] = d * self.factor[g]
            return u, None
        scale = np.ones(self.n)
        for g, d in zip(targets, doses):
            scale[g] = 1.0 - d
        return None, scale


def _dose_grid(
    network: GeneNetwork, screen: ConditionScreen, levels: np.ndarray, solver: Any, perturber: _Perturber,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For every singly perturbed gene g: the log steady state at every dose level, (levels, n), and the log of g's own
    *unscaled* production over degradation, log(s_g / B_g), (levels,). A knockdown scales g's production by (1 - d),
    so x_g = (1 - d) s_g / B_g: interpolating the smooth log(s_g / B_g) and multiplying by (1 - d) is exact at the
    grid points and accurate in between, where interpolating log x_g directly is not (it diverges towards d = 1).
    """
    log_ref = np.log(np.maximum(screen.reference, FLOOR))
    grid: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for c in screen.conditions:
        if len(c) != 1:
            continue
        g = c[0]
        rows = []
        own = []
        for d in levels:
            if d == 0.0:
                rows.append(log_ref)
                own.append(float(np.log(max(KineticsKernel(network).production(screen.reference)[g], FLOOR)
                                        / network.params.B[g])))
                continue
            u, scale = perturber.inputs([g], [d])
            x, _, _, _ = solver.solve_with_diagnostics(network, x0=screen.reference.copy(), perturbation=u,
                                                       production_scale=scale)
            if screen.mode != "input" and d >= 1.0:
                x = x.copy()
                x[g] = 0.0
            own.append(float(np.log(max(KineticsKernel(network, u).production(x)[g], FLOOR) / network.params.B[g])))
            rows.append(np.log(np.maximum(x, FLOOR)))
        grid[g] = (np.vstack(rows), np.asarray(own))
    return grid


def _single_states(
    entry: Tuple[np.ndarray, np.ndarray], levels: np.ndarray, dose: np.ndarray, gene: int, mode: str,
) -> np.ndarray:
    """log steady states (cells, n) of cells whose ``gene`` is perturbed at per-cell ``dose``."""
    states, own = entry
    out = _interpolate(states, levels, dose)
    if mode != "input":
        own_d = _interpolate(own[:, None], levels, dose)[:, 0]
        remaining = 1.0 - dose
        out[:, gene] = np.where(remaining > 0, np.log(np.maximum(remaining, FLOOR)) + own_d, np.log(FLOOR))
    return out


def _interpolate(grid: np.ndarray, levels: np.ndarray, dose: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation of a (levels, n) log-state grid at per-cell doses -> (cells, n)."""
    idx = np.clip(np.searchsorted(levels, dose, side="right") - 1, 0, len(levels) - 2)
    w = ((dose - levels[idx]) / (levels[idx + 1] - levels[idx]))[:, None]
    return (1.0 - w) * grid[idx] + w * grid[idx + 1]


def _lna_factors(
    kernel: KineticsKernel, x: np.ndarray, omega: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Linear response R (log x per log production) and a square root of the intrinsic log covariance."""
    n = x.size
    xs = np.maximum(x, FLOOR)
    s = kernel.production(xs, scaled=True)
    try:
        # The Jacobian is sparse (edges + diagonal): a sparse LU is ~2x faster than dense LAPACK even single-threaded.
        from scipy.sparse.linalg import splu

        R = -splu(kernel.jacobian(xs, sparse=True)).solve(np.diag(s)) / xs[:, None]
    except RuntimeError:
        R = np.eye(n)
    # Genes held at ~0 (full knockout) carry no noise.
    silent = x <= FLOOR * 10
    R[silent, :] = 0.0
    root = None
    if omega is not None:
        from scipy.linalg import solve_continuous_lyapunov

        D = np.diag((s + kernel.B * xs) / omega)
        try:
            C = solve_continuous_lyapunov(kernel.jacobian(xs), -D)
            C_log = C / np.outer(xs, xs)
            C_log[silent, :] = 0.0
            C_log[:, silent] = 0.0
            vals, vecs = np.linalg.eigh(0.5 * (C_log + C_log.T))
            root = vecs * np.sqrt(np.clip(vals, 0.0, None))
        except (np.linalg.LinAlgError, ValueError):
            root = None
    return R, root


def _extrinsic(rng: np.random.Generator, n_cells: int, n: int, cfg: CellConfig) -> np.ndarray:
    xi = cfg.extrinsic_gene_sd * rng.standard_normal((n_cells, n))
    xi += cfg.extrinsic_shared_sd * rng.standard_normal((n_cells, 1))
    return xi


def _sde_cells(
    kernel: KineticsKernel,
    x0: np.ndarray,
    xi: np.ndarray,
    scales: np.ndarray,
    inputs: np.ndarray,
    cfg: CellConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Euler-Maruyama chemical-Langevin integration of a batch of cells (rows of ``x0``), each with its own production
    multiplier ``exp(xi)``, knockdown scale and regulator input:
    ``dx = (s(x + u) * scale * e^xi - B x) dt + sqrt((s * scale * e^xi + B x) / omega) dW``.
    """
    from scipy.sparse import csr_matrix

    n_edges = kernel.src.size
    n = kernel.n
    incidence = csr_matrix((np.ones(n_edges), (np.arange(n_edges), kernel.dst)), shape=(n_edges, n))
    mult = np.exp(xi) * scales
    x = np.maximum(x0.copy(), 0.0)
    steps = max(1, int(round(cfg.sde_time / cfg.sde_dt)))
    dt = cfg.sde_dt
    for _ in range(steps):
        eff = np.maximum(x + inputs, 1e-12)
        val_m = eff[:, kernel.src] ** kernel.m
        denom = val_m + kernel.k_m + 1e-12
        term = np.where(kernel.act, 1.0 + val_m / denom, kernel.k_m / denom)
        log_prod = np.asarray(incidence.T.dot(np.log(np.maximum(term, 1e-300)).T)).T
        production = np.exp(log_prod) * kernel.v * mult
        decay = kernel.B * x
        x = x + (production - decay) * dt
        if cfg.omega is not None:
            x = x + np.sqrt(np.maximum(production + decay, 0.0) * dt / cfg.omega) * rng.standard_normal(x.shape)
        x = np.maximum(x, 0.0)
    return x


def simulate_cells(
    network: GeneNetwork,
    screen: ConditionScreen,
    cells_per_condition: int = 100,
    n_control_cells: Optional[int] = None,
    config: Optional[CellConfig] = None,
    seed: Optional[int] = None,
    solver: Optional[Any] = None,
) -> CellPopulation:
    """
    Simulates single cells for every condition of ``screen`` (plus non-targeting control cells).

    Args:
        network: the network the screen was simulated on.
        screen: result of :func:`grn_sim.perturbseq.simulate_conditions`.
        cells_per_condition: cells per perturbation condition.
        n_control_cells: non-targeting cells (default: ``cells_per_condition``).
        config: :class:`CellConfig` (default: CRISPRi-like with the LNA noise model).
    """
    cfg = config or CellConfig()
    cfg.validate()
    from grn_sim.inference._native import blas_limit
    # On a many-core machine under load, 64 BLAS threads make each 500x500 dense solve ~30x slower than 1-8.
    with blas_limit(cfg.blas_threads, user_api=None):
        return _simulate_cells(network, screen, cells_per_condition, n_control_cells, cfg, seed, solver)


def _simulate_cells(
    network: GeneNetwork,
    screen: ConditionScreen,
    cells_per_condition: int,
    n_control_cells: Optional[int],
    cfg: CellConfig,
    seed: Optional[int],
    solver: Optional[Any],
) -> CellPopulation:
    if cells_per_condition < 1:
        raise ValueError("cells_per_condition must be >= 1")
    rng = np.random.default_rng(seed)
    n = screen.n_genes
    names = screen.gene_names
    knockout = screen.mode == "knockout"
    # Knockout cells are escaped (0), monoallelic (0.5) or biallelic (1): three solves per gene suffice.
    levels = np.array([0.0, 0.5, 1.0]) if knockout else np.asarray(cfg.dose_levels, dtype=float)
    if screen.mode == "knockdown":
        # Most cells sit near the screen's nominal efficiency: make it a grid point (exact there).
        nominal = np.unique(np.round(np.asarray(screen.efficiency, dtype=float), 6))
        if nominal.size <= 3:
            levels = np.unique(np.concatenate([levels, nominal]))
    solver = solver or SteadyStateSolver()
    perturber = _Perturber(screen)
    grid = _dose_grid(network, screen, levels, solver, perturber)
    log_ref = np.log(np.maximum(screen.reference, FLOOR))

    # Guide-level mean efficiencies (knockdown / input); knockouts use the allele model instead.
    guide_mean = _beta(rng, np.full((n, cfg.guides_per_gene), cfg.guide_efficiency_mean),
                       cfg.guide_efficiency_concentration)

    def draw_doses(gene: int, guide_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = guide_idx.size
        escaped = rng.uniform(size=k) < cfg.escape_rate
        if knockout:
            dose = np.where(rng.uniform(size=k) < cfg.knockout_biallelic_rate, 1.0, 0.5)
        else:
            dose = _beta(rng, guide_mean[gene, guide_idx], cfg.cell_efficiency_concentration)
        dose = np.where(escaped, 0.0, dose)
        return dose, escaped

    blocks: List[np.ndarray] = []
    clipped = [0]
    cond_col: List[str] = []
    guide_col: List[str] = []
    target_col: List[Tuple[int, ...]] = []
    dose_rows: List[np.ndarray] = []
    escaped_rows: List[np.ndarray] = []

    def emit(log_mean: np.ndarray, targets: Tuple[int, ...], doses: np.ndarray, cond_name: str,
             guides: List[str], escaped: np.ndarray, state_for_lna: np.ndarray, lna_doses: Sequence[float]) -> None:
        """Adds a block of cells; the LNA linearises at ``state_for_lna``, the steady state at ``lna_doses``."""
        k = log_mean.shape[0]
        xi = _extrinsic(rng, k, n, cfg)
        if cfg.noise_model == "lna":
            u, scale = perturber.inputs(list(targets), list(lna_doses))
            kernel = KineticsKernel(network, u, scale)
            R, root = _lna_factors(kernel, np.exp(state_for_lna), cfg.omega)
            dev = xi @ np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).T
            if root is not None:
                dev = dev + rng.standard_normal((k, n)) @ np.nan_to_num(root, nan=0.0, posinf=0.0, neginf=0.0).T
            # Near a bifurcation the linear response explodes and the LNA stops being valid; cap the noise-driven
            # deviation at e^+-LNA_MAX_LOG_DEV (and count it) instead of overflowing.
            over = np.abs(dev) > LNA_MAX_LOG_DEV
            if over.any():
                clipped[0] += int(over.sum())
                dev = np.clip(dev, -LNA_MAX_LOG_DEV, LNA_MAX_LOG_DEV)
            log_x = log_mean + dev
            expr = np.exp(log_x)
            expr[log_mean <= np.log(FLOOR) + 1e-9] = 0.0
        else:
            kernel = KineticsKernel(network)
            scales = np.ones((k, n))
            inputs = np.zeros((k, n))
            for t_i, g in enumerate(targets):
                if screen.mode == "input":
                    inputs[:, g] = doses[:, t_i] * perturber.factor[g]
                else:
                    scales[:, g] = 1.0 - doses[:, t_i]
            expr = _sde_cells(kernel, np.exp(log_mean), xi, scales, inputs, cfg, rng)
        blocks.append(expr)
        cond_col.extend([cond_name] * k)
        guide_col.extend(guides)
        target_col.extend([targets] * k)
        padded = np.zeros((k, 2))
        padded[:, :doses.shape[1]] = doses
        dose_rows.append(padded)
        escaped_rows.append(escaped)

    # Non-targeting controls.
    n_ctrl = cells_per_condition if n_control_cells is None else int(n_control_cells)
    if n_ctrl > 0:
        g_idx = rng.integers(0, cfg.n_control_guides, size=n_ctrl)
        emit(np.tile(log_ref, (n_ctrl, 1)), (), np.zeros((n_ctrl, 0)), "control",
             [f"non-targeting_{i + 1}" for i in g_idx], np.zeros(n_ctrl, dtype=bool), log_ref, [])

    for k_cond, targets in enumerate(screen.conditions):
        name = screen.condition_names[k_cond]
        k = cells_per_condition
        nominal_log = np.log(np.maximum(screen.means[:, k_cond], FLOOR))
        if len(targets) == 1:
            g = targets[0]
            g_idx = rng.integers(0, cfg.guides_per_gene, size=k)
            dose, esc = draw_doses(g, g_idx)
            log_mean = _single_states(grid[g], levels, dose, g, screen.mode)
            guides = [f"{names[g]}_g{i + 1}" for i in g_idx]
            typical = float(np.median(dose[~esc])) if np.any(~esc) else 0.0
            state = _single_states(grid[g], levels, np.array([typical]), g, screen.mode)[0]
            emit(log_mean, targets, dose[:, None], name, guides, esc, state, [typical])
        else:
            a, b = targets
            ga = rng.integers(0, cfg.guides_per_gene, size=k)
            gb = rng.integers(0, cfg.guides_per_gene, size=k)
            da, ea = draw_doses(a, ga)
            db, eb = draw_doses(b, gb)
            # Per-cell doses act through each partner's dose grid when the other partner escaped; cells with both
            # partners active take the pair steady state (nominal doses) - a 2-D dose grid would cost n_levels^2
            # solves per pair.
            log_mean = np.tile(nominal_log, (k, 1))
            only_a = eb & ~ea
            only_b = ea & ~eb
            both_escaped = ea & eb
            if only_a.any():
                log_mean[only_a] = _single_states(grid[a], levels, da[only_a], a, screen.mode) if a in grid else log_ref
            if only_b.any():
                log_mean[only_b] = _single_states(grid[b], levels, db[only_b], b, screen.mode) if b in grid else log_ref
            log_mean[both_escaped] = log_ref
            dose = np.column_stack([da, db])
            guides = [f"{names[a]}_g{i + 1}+{names[b]}_g{j + 1}" for i, j in zip(ga, gb)]
            emit(log_mean, targets, dose, name, guides, both_escaped, nominal_log,
                 [float(screen.efficiency[a]) if screen.mode != "knockout" else 1.0,
                  float(screen.efficiency[b]) if screen.mode != "knockout" else 1.0])

    expression = np.vstack(blocks) if blocks else np.zeros((0, n))
    return CellPopulation(
        gene_names=list(names),
        expression=expression,
        condition=cond_col,
        guide=guide_col,
        targets=target_col,
        dose=np.vstack(dose_rows) if dose_rows else np.zeros((0, 2)),
        escaped=np.concatenate(escaped_rows) if escaped_rows else np.zeros(0, dtype=bool),
        noise_model=cfg.noise_model,
        meta={"config": cfg, "seed": seed, "mode": screen.mode, "lna_clipped_values": clipped[0]},
    )


def population_to_counts(
    population: CellPopulation,
    mean_library_depth: int = 20000,
    depth_dispersion: float = 0.35,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal measurement model (stage 3 replaces it): per-cell lognormal library size and Poisson sampling of each
    cell's expression composition. Returns ``(counts, library_size)``.
    """
    rng = np.random.default_rng(seed)
    expr = np.maximum(population.expression, 0.0)
    totals = expr.sum(axis=1, keepdims=True)
    comp = np.divide(expr, totals, out=np.zeros_like(expr), where=totals > 0)
    depth = rng.lognormal(np.log(mean_library_depth) - 0.5 * depth_dispersion ** 2, depth_dispersion,
                          size=(expr.shape[0], 1))
    counts = rng.poisson(comp * depth).astype(np.int32)
    return counts, depth[:, 0]
