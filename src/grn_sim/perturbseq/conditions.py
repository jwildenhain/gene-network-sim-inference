"""
Stage 1 of the Perturb-seq generator: condition-level mechanics and ground truth.

For every condition of a screen (single knockdowns/knockouts and optional double knockouts) this solves the Hill
kinetics to steady state and records what a benchmark needs to know about the truth behind the data:

- ``means``: steady-state expression per condition (the noise-free mean a cell population would have);
- ``total_effects``: log2 fold change of every gene in every condition (the *total* causal effect, direct plus
  everything mediated through the network: what a knockout screen measures);
- ``elasticity``: local sensitivity ``d log s_k / d log x_j`` of every edge at the unperturbed state (edges with a
  tiny elasticity are silent: their regulator can move without the target noticing);
- ``linear_response``: ``d log x_i / d log v_j``, the small-perturbation total-effect matrix implied by the Jacobian
  (``-J^-1`` scaled); comparing it to ``total_effects`` shows how non-linear the knockouts are;
- ``labels``: why each single perturbation does or does not move other genes (``no_targets``,
  ``no_downstream_effect``, ``effective``);
- for double perturbations, the deviation from the multiplicative (log-additive) null, i.e. the epistasis.

Modes: ``knockdown`` (CRISPRi: production of the target scaled by ``1 - e``), ``knockout`` (production set to 0) and
``input`` (the thesis convention: ``+factor`` added to the regulator activity; the gene's own transcript is not hit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.kinetics import edge_elasticities, evaluate_jacobian, production_rates
from grn_sim.simulator.steady_state import SteadyStateSolver

PERTURBATION_MODES = ("knockdown", "knockout", "input")
LABELS = ("no_targets", "no_downstream_effect", "effective")

PairSpec = Union[None, int, str, Sequence[Tuple[int, int]]]


@dataclass
class ConditionScreen:
    """Noise-free result of a perturbation screen plus its ground truth (genes are rows, conditions are columns)."""

    gene_names: List[str]
    mode: str
    efficiency: np.ndarray                 # (n,) knockdown efficiency per gene (1 for knockout; factor for input)
    reference: np.ndarray                  # (n,) unperturbed steady state
    conditions: List[Tuple[int, ...]]      # perturbed gene indices per condition (singles first, then pairs)
    means: np.ndarray                      # (n, c) steady state per condition
    steady: np.ndarray                     # (c,) whether each condition reached a steady state
    total_effects: np.ndarray              # (n, c) log2 fold change vs reference
    adjacency: np.ndarray                  # (n, n) signed true network, A[j, k] = effect of j on k
    elasticity: np.ndarray                 # (n, n) d log s_k / d log x_j at the reference state
    linear_response: np.ndarray            # (n, n) d log x_i / d log v_j at the reference state
    labels: List[str]                      # one per single condition
    effect_threshold: float
    epistasis: Optional[np.ndarray] = None  # (n_pairs,) norm of the deviation from the log-additive null
    pair_deviation: Optional[np.ndarray] = None  # (n, n_pairs) deviation vector per pair
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def condition_names(self) -> List[str]:
        return ["+".join(self.gene_names[i] for i in c) for c in self.conditions]

    @property
    def single_index(self) -> Dict[int, int]:
        """Gene index -> column of its single-perturbation condition."""
        return {c[0]: k for k, c in enumerate(self.conditions) if len(c) == 1}

    @property
    def pair_columns(self) -> List[int]:
        return [k for k, c in enumerate(self.conditions) if len(c) == 2]

    def off_target_mask(self) -> np.ndarray:
        """(n, c) True for genes that are not themselves perturbed in that condition."""
        mask = np.ones((self.n_genes, len(self.conditions)), dtype=bool)
        for k, c in enumerate(self.conditions):
            mask[list(c), k] = False
        return mask

    def summary(self) -> Dict[str, Any]:
        """Screen-level statistics comparable to genome-scale Perturb-seq (see the realism targets in the proposal)."""
        singles = [k for k, c in enumerate(self.conditions) if len(c) == 1]
        lfc = self.total_effects[:, singles]
        off = self.off_target_mask()[:, singles]
        affected = (np.abs(lfc) >= self.effect_threshold) & off
        direct = np.zeros_like(affected)
        for col, k in enumerate(singles):
            j = self.conditions[k][0]
            direct[:, col] = self.adjacency[j, :] != 0
        n_aff = int(affected.sum())
        out: Dict[str, Any] = {
            "n_genes": self.n_genes,
            "n_single": len(singles),
            "n_pairs": len(self.pair_columns),
            "mode": self.mode,
            "effect_threshold_log2": self.effect_threshold,
            "fraction_with_effect": float(np.mean([lab == "effective" for lab in self.labels])) if self.labels else 0.0,
            "label_counts": {lab: int(sum(1 for x in self.labels if x == lab)) for lab in LABELS},
            "affected_pair_density": float(affected.sum() / max(off.sum(), 1)),
            "fraction_effects_indirect": float((affected & ~direct).sum() / n_aff) if n_aff else 0.0,
            "fraction_direct_edges_detected": float((affected & direct).sum() / max(int(direct.sum()), 1)),
            "silent_edge_fraction": float(np.mean(np.abs(self.elasticity[self.adjacency != 0]) < 0.05))
            if np.any(self.adjacency != 0) else 0.0,
            "all_steady": bool(np.all(self.steady)),
        }
        if self.epistasis is not None and len(self.epistasis):
            out["median_epistasis"] = float(np.median(self.epistasis))
        return out

    def to_uns(self) -> Dict[str, Any]:
        """Ground truth as plain arrays/strings, ready for ``AnnData.uns`` (h5ad cannot store tuples or None)."""
        uns: Dict[str, Any] = {
            "mode": self.mode,
            "efficiency": np.asarray(self.efficiency, dtype=float),
            "condition_names": np.asarray(self.condition_names, dtype=object),
            "reference_expression": self.reference,
            "condition_means": self.means,
            "total_effects_log2": self.total_effects,
            "steady": self.steady.astype(bool),
            "ground_truth_network": self.adjacency.astype(float),
            "edge_elasticity": self.elasticity,
            "linear_response": self.linear_response,
            "single_labels": np.asarray(self.labels, dtype=object),
            "effect_threshold_log2": float(self.effect_threshold),
        }
        if self.epistasis is not None:
            uns["pair_epistasis"] = self.epistasis
            uns["pair_deviation_log2"] = self.pair_deviation
        return uns


def _select_pairs(pairs: PairSpec, candidates: Sequence[int], rng: np.random.Generator) -> List[Tuple[int, int]]:
    if pairs is None or (isinstance(pairs, int) and pairs <= 0):
        return []
    if isinstance(pairs, str):
        if pairs != "all":
            raise ValueError(f"pairs must be None, an int, 'all' or a list of index pairs, got {pairs!r}")
        return list(combinations(sorted(candidates), 2))
    if isinstance(pairs, int):
        pool = list(combinations(sorted(candidates), 2))
        if pairs >= len(pool):
            return pool
        picked = rng.choice(len(pool), size=pairs, replace=False)
        return [pool[i] for i in sorted(picked)]
    out = []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a == b:
            raise ValueError(f"a double perturbation needs two different genes, got ({a}, {b})")
        out.append((min(a, b), max(a, b)))
    return out


def simulate_conditions(
    network: GeneNetwork,
    mode: str = "knockdown",
    efficiency: Union[float, Sequence[float], np.ndarray] = 0.8,
    genes: Optional[Sequence[int]] = None,
    pairs: PairSpec = None,
    pair_pool: str = "effective",
    effect_threshold: float = 0.1,
    gene_names: Optional[List[str]] = None,
    solver: Optional[Any] = None,
    seed: Optional[int] = None,
    lfc_pseudocount: float = 1e-6,
) -> ConditionScreen:
    """
    Simulates a perturbation screen on ``network`` and returns its noise-free means and ground truth.

    Args:
        mode: ``knockdown``, ``knockout`` or ``input``.
        efficiency: knockdown efficiency in (0, 1) (scalar or per gene); for ``input`` the added regulator activity.
            Ignored for ``knockout``.
        genes: genes to perturb singly (default: all).
        pairs: double perturbations: ``None`` (none), an int (that many random pairs), ``"all"`` or explicit pairs.
        pair_pool: for random/"all" pairs, draw from ``"effective"`` single perturbations (the informative ones,
            default) or from ``"all"`` perturbed genes.
        effect_threshold: |log2 FC| at which an off-target gene counts as affected (used for the labels).
        solver: steady-state solver (default: the precise :class:`SteadyStateSolver`).
        lfc_pseudocount: added to both expression levels before taking log2 ratios (a knocked-out gene goes to 0).
    """
    if mode not in PERTURBATION_MODES:
        raise ValueError(f"mode must be one of {PERTURBATION_MODES}, got {mode!r}")
    if pair_pool not in ("effective", "all"):
        raise ValueError(f"pair_pool must be 'effective' or 'all', got {pair_pool!r}")
    rng = np.random.default_rng(seed)
    n = network.n
    names = list(gene_names) if gene_names is not None else [f"Gene_{i + 1:03d}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"gene_names has {len(names)} entries for {n} genes")
    genes_list = list(range(n)) if genes is None else [int(g) for g in genes]

    eff = np.broadcast_to(np.asarray(efficiency, dtype=float), (n,)).copy()
    if mode == "knockout":
        eff[:] = 1.0
    elif mode == "knockdown" and (np.any(eff <= 0.0) or np.any(eff >= 1.0)):
        raise ValueError("knockdown efficiency must be in (0, 1); use mode='knockout' for a full knockout")

    solver = solver or SteadyStateSolver()
    x_ss, ok, _, _ = solver.solve_with_diagnostics(network)
    if not ok or np.any(x_ss <= 0):
        x_ss, ok, _, _ = solver.solve_with_diagnostics(network, x0=np.ones(n))
    network.steady_state_unperturbed = x_ss

    def solve(targets: Tuple[int, ...]) -> Tuple[np.ndarray, bool]:
        u = None
        scale = None
        if mode == "input":
            u = np.zeros(n)
            u[list(targets)] = eff[list(targets)]
        else:
            scale = np.ones(n)
            scale[list(targets)] = 1.0 - eff[list(targets)]
        x, steady, _, _ = solver.solve_with_diagnostics(network, x0=x_ss.copy(), perturbation=u, production_scale=scale)
        if mode == "knockout":
            x = x.copy()
            x[list(targets)] = 0.0
        return x, bool(steady)

    def lfc(x: np.ndarray) -> np.ndarray:
        return np.log2((x + lfc_pseudocount) / (x_ss + lfc_pseudocount))

    conditions: List[Tuple[int, ...]] = [(g,) for g in genes_list]
    cols: List[np.ndarray] = []
    steady: List[bool] = []
    for c in conditions:
        x, s = solve(c)
        cols.append(x)
        steady.append(s)

    adjacency = np.asarray(network.topology.adjacency).copy()
    single_lfc = np.column_stack([lfc(x) for x in cols]) if cols else np.zeros((n, 0))
    labels: List[str] = []
    for k, g in enumerate(genes_list):
        off = np.delete(np.abs(single_lfc[:, k]), g)
        if not np.any(adjacency[g, :] != 0):
            labels.append("no_targets")
        elif off.size and off.max() >= effect_threshold:
            labels.append("effective")
        else:
            labels.append("no_downstream_effect")

    pool = [g for g, lab in zip(genes_list, labels) if lab == "effective"] if pair_pool == "effective" else genes_list
    pair_list = _select_pairs(pairs, pool, rng)
    single_of = {g: k for k, g in enumerate(genes_list)}
    epistasis = deviation = None
    if pair_list:
        dev_cols = []
        eps = []
        for a, b in pair_list:
            x, s = solve((a, b))
            conditions.append((a, b))
            cols.append(x)
            steady.append(s)
            dual = lfc(x)
            if a in single_of and b in single_of:
                d = dual - (single_lfc[:, single_of[a]] + single_lfc[:, single_of[b]])
            else:
                # A pair partner without a single condition: compute its single effect on the fly.
                la = single_lfc[:, single_of[a]] if a in single_of else lfc(solve((a,))[0])
                lb = single_lfc[:, single_of[b]] if b in single_of else lfc(solve((b,))[0])
                d = dual - (la + lb)
            d[[a, b]] = 0.0  # the perturbed genes themselves are dominated by the perturbation
            dev_cols.append(d)
            eps.append(float(np.linalg.norm(d)))
        deviation = np.column_stack(dev_cols)
        epistasis = np.asarray(eps)

    means = np.column_stack(cols) if cols else np.zeros((n, 0))
    total = np.column_stack([lfc(x) for x in cols]) if cols else np.zeros((n, 0))

    # Local (linearised) quantities at the unperturbed state.
    elasticity = edge_elasticities(x_ss, network)
    J = evaluate_jacobian(x_ss, network)
    s_rates = production_rates(x_ss, network)
    try:
        # dx/dlog v_j = -J^-1[:, j] * s_j ; in log space divide by x_i.
        linear = -np.linalg.solve(J, np.diag(s_rates)) / np.maximum(x_ss, 1e-12)[:, None]
    except np.linalg.LinAlgError:
        linear = np.full((n, n), np.nan)

    return ConditionScreen(
        gene_names=names,
        mode=mode,
        efficiency=eff,
        reference=x_ss,
        conditions=conditions,
        means=means,
        steady=np.asarray(steady, dtype=bool),
        total_effects=total,
        adjacency=adjacency,
        elasticity=elasticity,
        linear_response=linear,
        labels=labels,
        effect_threshold=effect_threshold,
        epistasis=epistasis,
        pair_deviation=deviation,
        meta={"basal_steady": bool(ok), "solver": type(solver).__name__, "seed": seed},
    )
