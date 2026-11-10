"""
Stage 4 of the Perturb-seq generator: calibrating networks towards genome-scale Perturb-seq.

The thesis parameterisation (Hill coefficients 1-6, half-saturation constants uniform in 1-2.5 regardless of the
regulator's level) leaves many edges on a flat part of their Hill curve: their regulator can be knocked down without
the target noticing ("silent edges"). :func:`calibrate_kinetics` anchors every half-saturation constant to its
regulator's steady-state level (as SERGIO does), optionally with a narrower Hill-coefficient range, so knockdowns
propagate. :data:`PRESETS` collects network regimes; ``"replogle_like"`` follows the regime Aguirre et al. (2025)
found to reproduce genome-scale Perturb-seq (2-4 regulators per gene, 5-10 modules, heavy-tailed out-degree,
~80% activating edges) together with anchored kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType, generate_topology
from grn_sim.simulator.steady_state import SteadyStateSolver


def calibrate_kinetics(
    network: GeneNetwork,
    spread: float = 0.5,
    hill_range: Optional[Tuple[float, float]] = (1.0, 3.0),
    max_rounds: int = 120,
    damping: float = 0.2,
    tol: float = 0.02,
    seed: Optional[int] = None,
    solver: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Re-parameterises ``network`` in place so every edge sits near the responsive middle of its Hill curve:
    ``K[j, k] = x_ss[j] * f[j, k]`` with fixed ``f ~ LogNormal(0, spread)``, for activating and inhibiting edges.
    Hill coefficients are redrawn uniformly from ``hill_range`` (``None`` keeps them).

    The steady state moves when the constants change, so the anchoring is a fixed-point iteration. Undamped it
    diverges and at damping 0.5 activation-rich networks keep flipping between global states; at the default 0.2 it
    usually converges (e.g. 27 rounds for 200 genes with 80% activating edges), and the step is halved (down to
    damping/8) whenever the change stops improving for 8 rounds. It runs damped in log space, ``log K <- (1 - damping) log K + damping log(x f)``, until 95% of genes change by less than ``tol``
    (|d log x|) or ``max_rounds`` is reached. A few genes can keep flipping between two states (bistable pockets
    created by strong anchored inhibition); they are reported, not chased.

    Returns diagnostics: rounds run, the 95th-percentile and maximum final |d log x|, the number of genes still
    moving by more than ``tol``, the achieved spread of ``K / (x f)`` and convergence.
    """
    if spread < 0 or max_rounds < 1 or not 0.0 < damping <= 1.0 or tol <= 0:
        raise ValueError("need spread >= 0, max_rounds >= 1, 0 < damping <= 1 and tol > 0")
    if hill_range is not None and not 0.0 < hill_range[0] <= hill_range[1]:
        raise ValueError("hill_range must be (low, high) with 0 < low <= high")
    rng = np.random.default_rng(seed)
    solver = solver or SteadyStateSolver()
    A = network.topology.adjacency
    edges = A != 0
    p = network.params
    if hill_range is not None:
        draw = rng.uniform(hill_range[0], hill_range[1], size=A.shape)
        p.n_k = np.where(A > 0, draw, 0.0)
        p.n_i = np.where(A < 0, draw, 0.0)
    log_f = rng.normal(0.0, spread, size=A.shape) if spread > 0 else np.zeros(A.shape)
    x, ok, _, _ = solver.solve_with_diagnostics(network)
    log_K = np.log(np.maximum(x, 1e-6))[:, None] + log_f
    change = float("inf")
    dlog = np.zeros(1)
    rounds = 0
    best = float("inf")
    since_best = 0
    step = damping
    for round_no in range(1, max_rounds + 1):
        rounds = round_no
        log_K = (1.0 - step) * log_K + step * (np.log(np.maximum(x, 1e-6))[:, None] + log_f)
        K = np.exp(log_K)
        p.Ka = np.where(A > 0, K, 0.0)
        p.Ki = np.where(A < 0, K, 0.0)
        x_new, ok, _, _ = solver.solve_with_diagnostics(network, x0=x.copy())
        dlog = np.abs(np.log(np.maximum(x_new, 1e-9) / np.maximum(x, 1e-9)))
        change = float(np.quantile(dlog, 0.95))
        x = x_new
        if change < tol:
            break
        # Oscillating between states: halve the step when the change has not improved for 8 rounds.
        if change < 0.9 * best:
            best, since_best = change, 0
        else:
            since_best += 1
            if since_best >= 8 and step > damping / 8:
                step *= 0.5
                since_best = 0
    network.steady_state_unperturbed = x
    ratio = (np.exp(log_K) / np.maximum(x, 1e-6)[:, None] / np.exp(log_f))[edges]
    return {
        "rounds": rounds,
        "final_q95_dlogx": change,
        "final_max_dlogx": float(np.max(dlog)),
        "genes_still_moving": int(np.sum(dlog >= tol)),
        "converged": bool(change < tol),
        "final_damping": step,
        "steady": bool(ok),
        "anchor_ratio_10_90": tuple(float(v) for v in np.quantile(ratio, [0.1, 0.9])) if ratio.size else (1.0, 1.0),
        "n_edges": int(edges.sum()),
    }


@dataclass
class NetworkPreset:
    """A network regime: topology plus kinetics calibration."""

    topology: str = "modular"
    k_av: float = 2.5
    ratio: float = 0.8                    # share of activating edges
    n_modules: Optional[int] = None       # None: ~1 per 50 genes, 5..10
    p_within: float = 0.85
    attachment_offset: float = 0.5
    anchor_kinetics: bool = True
    spread: float = 0.5
    hill_range: Optional[Tuple[float, float]] = (1.0, 3.0)
    guide_efficiency_mean: float = 0.92   # cell model: mean guide knockdown that gives ~81% observed knockdown
    extra: Dict[str, Any] = field(default_factory=dict)

    def cell_config(self, **overrides: Any) -> Any:
        """The :class:`grn_sim.perturbseq.CellConfig` that goes with this preset."""
        from grn_sim.perturbseq.cells import CellConfig

        return CellConfig(**{"guide_efficiency_mean": self.guide_efficiency_mean, **overrides})

    def build(self, n: int, seed: Optional[int] = None) -> GeneNetwork:
        """Generates and (optionally) calibrates a network of ``n`` genes."""
        rng = np.random.default_rng(seed)
        if self.topology == "modular":
            modules = self.n_modules or int(np.clip(n // 50, 5, 10))
        else:
            modules = 2
        topo = generate_topology(n, k_av=self.k_av, net_type=TopologyType(self.topology), ratio=self.ratio,
                                 subdiv=modules, rng=rng, p_within=self.p_within,
                                 attachment_offset=self.attachment_offset)
        net = GeneNetwork(topo, rng=rng)
        if self.anchor_kinetics:
            calibrate_kinetics(net, spread=self.spread, hill_range=self.hill_range,
                               seed=int(rng.integers(0, 2 ** 31 - 1)))
        return net


PRESETS: Dict[str, NetworkPreset] = {
    # The studio's historical defaults: power-law topology, thesis kinetics.
    "thesis": NetworkPreset(topology="power_law", k_av=1.8, ratio=0.5, anchor_kinetics=False,
                            guide_efficiency_mean=0.8),
    # Aguirre et al. 2025 regime + anchored kinetics, tuned with scripts/calibrate_perturbseq.py at 500 genes
    # (knockdown screen, 100 cells per perturbation, default measurement). At that size it meets 6 of the 7 realism
    # targets: knockdown 0.81, phenotype share 0.35, affected-pair density 0.024, bidirectional share 0.05, top-5%
    # effect share 0.29, control false positives 0. It misses the mediated share of effects (0.79 vs >= 0.90): denser
    # networks (k_av 3) reach 0.92 but then overshoot density and feedback. Genome-scale screens combine low density
    # with ~98.5% mediated effects because they span thousands of genes; at 500 genes the two cannot both be met.
    "replogle_like": NetworkPreset(),
    # Pathway architectures (see grn_sim.core.topology): same anchored kinetics as replogle_like, different wiring.
    "cascade": NetworkPreset(topology="cascade", k_av=2.0, ratio=0.7),
    "feedforward": NetworkPreset(topology="feedforward", k_av=2.2, ratio=0.7),
    "lineage": NetworkPreset(topology="lineage", k_av=2.2, ratio=0.7),
    "regulon": NetworkPreset(topology="regulon", k_av=2.0, ratio=0.7),
}
