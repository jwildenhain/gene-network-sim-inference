"""
The algorithm registry: one place that knows every reverse-engineering algorithm.

The web studio, the job queue, the scoring orientation, the parameter sweep and the API all read this registry, so a
new algorithm needs a single ``register`` call (see the end of this module for the built-in ones):

    from grn_sim.inference.registry import AlgorithmSpec, register

    register(AlgorithmSpec(
        key="my_algo", label="My algorithm", long_name="My algorithm (what it does)", paradigm="Regression",
        build=lambda ctx: MyAlgo(k_interactions=ctx.k_est),
        additive_target_regulator=True,           # how it orients its output, see grn_sim.inference.orientation
    ))

Once registered it is accepted by ``/api/infer`` and the job queue, run by the studio and the sweep, listed by
``/api/algorithms`` (which the page uses to add its checkbox and menu entries), and scored in the right orientation.
Stored results need nothing: the archive keys them by the algorithm's key.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grn_sim.inference import orientation
from grn_sim.inference.base import BaseInferenceAlgorithm


@dataclass(frozen=True)
class BuildContext:
    """What a builder may depend on when it creates the algorithm for one network."""
    k_est: int = 3              # the network's mean in-degree, the usual K for top-K methods
    use_reg: bool = True        # regulator-input fits (Hill / Perturb-seq data) rather than the additive model
    is_sos: bool = False        # the E. coli SOS benchmark
    seed: int = 42


@dataclass
class AlgorithmSpec:
    key: str
    label: str                                              # short name for checkboxes and menus
    build: Callable[[BuildContext], BaseInferenceAlgorithm]
    long_name: str = ""                                     # leaderboard / sweep name (defaults to ``label``)
    paradigm: str = "Reverse-engineering algorithm"
    badge: str = ""
    color: Optional[str] = None                             # chart colour; the page picks one when absent
    dash: Optional[Tuple[int, ...]] = None
    core: bool = False                                      # part of the "Thesis Core" selection
    order: int = 1000                                       # listing order (built-ins use 10, 20, ...)
    regulator_input: bool = False                           # its perturbation_mode="regulator" fit returns T[regulator, target]
    uses_regulator_model: Optional[bool] = None             # shown as "regulator_input" in results (default: regulator_input)
    negate_u_in_additive: bool = False                      # legacy: fitted on -U under the additive model
    # output orientation (see grn_sim.inference.orientation.declare_output_orientation)
    target_row: bool = False
    additive_target_regulator: bool = False
    orientation_free: bool = False
    # Virtual Cell Challenge: set ``response_sign`` (+1 or -1) when A_est is a steady-state Jacobian, so the algorithm
    # also takes part in the held-out response comparison (R = sign * pinv(A_est)); left None it only joins the
    # structure comparison. ``response_unsupported`` names the reason A_est cannot serve as a Jacobian.
    response_sign: Optional[float] = None
    response_unsupported: str = ""
    # partial screens (m perturbed genes of n measured): "native" = fits (X, U) with fewer columns than genes as it is,
    # "adapter" = needs grn_sim.inference.partial.PartialAdapter (fit on the perturbed block, extended to the others)
    partial: str = "native"
    extra: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.long_name:
            self.long_name = self.label
        if self.uses_regulator_model is None:
            self.uses_regulator_model = self.regulator_input


_REGISTRY: Dict[str, AlgorithmSpec] = {}


def register(spec: AlgorithmSpec, replace: bool = False) -> AlgorithmSpec:
    """Adds an algorithm. Raises ValueError for a key that is taken (unless ``replace``) or not a usable key."""
    if not spec.key or not spec.key.replace("_", "").isalnum() or spec.key != spec.key.lower():
        raise ValueError(f"algorithm key must be lower-case letters, digits and underscores, got {spec.key!r}")
    if spec.key in _REGISTRY and not replace:
        raise ValueError(f"algorithm {spec.key!r} is already registered")
    _REGISTRY[spec.key] = spec
    orientation.declare_output_orientation(
        spec.key, target_row=spec.target_row, additive_target_regulator=spec.additive_target_regulator,
        regulator_input=spec.regulator_input, orientation_free=spec.orientation_free)
    return spec


def unregister(key: str) -> None:
    """Removes an algorithm from the registry (the orientation declaration stays; it is harmless without the key)."""
    _REGISTRY.pop(key, None)


def get(key: str) -> AlgorithmSpec:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown algorithm {key!r}") from None


def is_registered(key: str) -> bool:
    return key in _REGISTRY


def specs() -> List[AlgorithmSpec]:
    return sorted(_REGISTRY.values(), key=lambda s: (s.order, s.key))


def keys() -> List[str]:
    return [s.key for s in specs()]


def core_keys() -> List[str]:
    return [s.key for s in specs() if s.core]


def build(key: str, ctx: Optional[BuildContext] = None) -> BaseInferenceAlgorithm:
    return get(key).build(ctx or BuildContext())


def build_partial(key: str, ctx: Optional[BuildContext] = None) -> BaseInferenceAlgorithm:
    """The algorithm ready for a partial screen: wrapped in a ``PartialAdapter`` when its spec says so. The adapter's output is
    ``A_est[regulator, target]`` over all genes (already oriented); a native algorithm's output follows its own orientation."""
    algo = build(key, ctx)
    if get(key).partial == "adapter":
        from grn_sim.inference.partial import PartialAdapter
        return PartialAdapter(algo, key)
    return algo


def build_all(ctx: Optional[BuildContext] = None, only: Optional[Sequence[str]] = None) -> Dict[str, BaseInferenceAlgorithm]:
    ctx = ctx or BuildContext()
    return {s.key: s.build(ctx) for s in specs() if only is None or s.key in only}


def describe() -> List[Dict[str, object]]:
    """What the page needs to list the algorithms (served by /api/algorithms)."""
    return [{"key": s.key, "label": s.label, "long_name": s.long_name, "paradigm": s.paradigm, "badge": s.badge,
             "color": s.color, "dash": list(s.dash) if s.dash else None, "core": s.core, "order": s.order,
             "regulator_input": bool(s.uses_regulator_model), "partial": s.partial} for s in specs()]


def sweep_metadata() -> Dict[str, Dict[str, str]]:
    """Name / type / badge per algorithm for the sweep's leaderboard."""
    return {s.key: {"name": s.long_name, "type": s.paradigm, "badge": s.badge} for s in specs()}


# ---------------------------------------------------------------------------------------------------------------------
# Built-in algorithms, configured as in the web studio's runs (K = the network's mean in-degree)
# ---------------------------------------------------------------------------------------------------------------------
def _register_builtin() -> None:
    from grn_sim.inference.arachne import AracneAlgorithm
    from grn_sim.inference.atlasshift import AtlasShift
    from grn_sim.inference.by_chance import ByChance
    from grn_sim.inference.causalgrn import CausalGRNAlgorithm
    from grn_sim.inference.cellbox import CellBoxNeuralODE
    from grn_sim.inference.cpa_lite import CPALite
    from grn_sim.inference.deeprig import DeepRIGAlgorithm
    from grn_sim.inference.dmdgrn import DMDGRNAlgorithm
    from grn_sim.inference.ensemble import VotingEnsemble
    from grn_sim.inference.fast_svd import FastSVD
    from grn_sim.inference.g2s import GeneralToSpecific
    from grn_sim.inference.genie3 import GENIE3Algorithm
    from grn_sim.inference.inspre import InspreAlgorithm
    from grn_sim.inference.interventional_baselines import (GeneFeatureBaseline, GRNBoostLite, MeanDifference,
                                                            PearsonCorrelation)
    from grn_sim.inference.network_deconv import NetworkDeconvolution
    from grn_sim.inference.nir import NIRAlgorithm
    from grn_sim.inference.notears import NOTEARSAlgorithm
    from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
    from grn_sim.inference.s2g import SimpleToGeneral
    from grn_sim.inference.score_vote import ScoreVote
    from grn_sim.inference.sfinn import SFINNAlgorithm

    def mode(c: BuildContext) -> str:
        return "regulator" if c.use_reg else "additive"

    def k_if_reg(c: BuildContext) -> Optional[int]:
        return c.k_est if c.use_reg else None

    def nir_member(c: BuildContext) -> NIRAlgorithm:
        return NIRAlgorithm(k_interactions=c.k_est, perturbation_mode="regulator", adaptive_k=True)

    def g2s_member(c: BuildContext) -> GeneralToSpecific:
        return GeneralToSpecific(drop_rule="loo")   # AIC chooses the size: no edge count from the truth

    builtin = [
        AlgorithmSpec("s2g", "S2G (Forward AIC)", lambda c: SimpleToGeneral(), "Simple-to-General (S2G)",
                      "Forward Stepwise AIC", "Top Single Stepwise", "#0284c7", core=True, additive_target_regulator=True),
        AlgorithmSpec("g2s", "G2S (Backward AIC)", lambda c: g2s_member(c), "General-to-Specific (G2S)",
                      "Backward Stepwise AIC", "Stepwise Backward", "#d97706", core=True, additive_target_regulator=True),
        AlgorithmSpec("vote", "Voting Ensemble", lambda c: VotingEnsemble(
            g2s=GeneralToSpecific(drop_rule="loo"), extra_members=[nir_member(c)] if c.use_reg else None),
                      "Voting Ensemble", "Majority (S2G, G2S, NIR)", "Consensus Winner", "#16a34a", core=True,
                      uses_regulator_model=True, additive_target_regulator=True),
        AlgorithmSpec("score_vote", "Score Vote (evidence-weighted)", lambda c: ScoreVote(
            s2g=SimpleToGeneral(), g2s=g2s_member(c), nir=nir_member(c) if c.use_reg else None, k_interactions=c.k_est),
                      "Score Vote", "Evidence-weighted (S2G, G2S, NIR)", "Evidence Consensus", "#15803d", (6, 2),
                      uses_regulator_model=True, additive_target_regulator=True),
        AlgorithmSpec("nir", "NIR (Multi-Reg)", lambda c: NIRAlgorithm(
            k_interactions=c.k_est, perturbation_mode=mode(c), adaptive_k=c.use_reg), "NIR (Multiple Regression)",
                      "Linear Matrix Inverse", "Matrix Inverse Baseline", "#8b5cf6", core=True, regulator_input=True,
                      negate_u_in_additive=True, additive_target_regulator=True),
        AlgorithmSpec("fast", "FAST (SVD)", lambda c: FastSVD(
            k_interactions=c.k_est, add_diagonal_identity=c.is_sos, perturbation_mode=mode(c), adaptive_k=c.use_reg),
                      "FAST (SVD Inversion)", "SVD Truncation", "SVD Regularized", "#ef4444", (4, 3), core=True,
                      regulator_input=True, negate_u_in_additive=True, additive_target_regulator=True),
        AlgorithmSpec("by_chance", "By-Chance", lambda c: ByChance(k_interactions=c.k_est, seed=c.seed),
                      "By-Chance (Random)", "Null Bernoulli Model", "Null Random Baseline", "#64748b", (3, 3), core=True,
                      orientation_free=True),
        AlgorithmSpec("atlasshift", "AtlasShift", lambda c: AtlasShift(de_threshold=0.15), "AtlasShift (Empirical)",
                      "Cross-Atlas Latent Transfer", "Empirical Transfer", "#22c55e"),
        AlgorithmSpec("cellbox", "CellBox Neural ODE", lambda c: CellBoxNeuralODE(epochs=70, perturbation_mode=mode(c), k_interactions=k_if_reg(c)),
                      "CellBox Neural ODE", "Adjoint Gradient Optimizer", "Deep Neural ODE", "#ec4899",
                      regulator_input=True, additive_target_regulator=True),
        AlgorithmSpec("genie3", "GENIE3 (Forest)", lambda c: GENIE3Algorithm(n_estimators=40, seed=c.seed, k_interactions=k_if_reg(c)), "GENIE3",
                      "Random Forest Tree Importance", "ML Tree Ensemble", "#10b981", target_row=True),
        AlgorithmSpec("netdeconv", "Net Deconvolution", lambda c: NetworkDeconvolution(k_interactions=k_if_reg(c)),
                      "Network Deconvolution", "Direct-Indirect Deconvolution", "Spectral Deconvolution", "#0ea5e9", (2, 2)),
        AlgorithmSpec("ridge_cv", "RidgeCV (L2)", lambda c: KaggleMultiOutputRidge(
            k_interactions=k_if_reg(c), perturbation_mode=mode(c)), "Multi-Output RidgeCV", "L2 Regularized Inverse",
                      "L2 Ridge Baseline", "#06b6d4", regulator_input=True, negate_u_in_additive=True,
                      additive_target_regulator=True),
        AlgorithmSpec("cpa_lite", "CPA-Lite Autoenc", lambda c: CPALite(epochs=40, k_interactions=k_if_reg(c)), "CPA-Lite Autoencoder",
                      "Latent Perturbation AE", "Latent Autoencoder", "#f97316"),
        AlgorithmSpec("arachne", "ARACNE (MI+DPI)", lambda c: AracneAlgorithm(k_interactions=k_if_reg(c)), "ARACNE",
                      "Information Theoretic (MI + DPI)", "Mutual Info & DPI", "#84cc16"),
        AlgorithmSpec("causalgrn", "CausalGRN (Causal)", lambda c: CausalGRNAlgorithm(k_interactions=k_if_reg(c)),
                      "CausalGRN", "Interventional Causal Discovery", "Causal Do-Calculus", "#e11d48", target_row=True),
        AlgorithmSpec("dmdgrn", "DMD-GRN (Koopman)", lambda c: DMDGRNAlgorithm(k_interactions=k_if_reg(c)), "DMD-GRN",
                      "Dynamic Mode Decomposition", "Koopman Dynamics", "#a855f7", (2, 2)),
        AlgorithmSpec("deeprig", "DeepRIG (GAE+GCN)", lambda c: DeepRIGAlgorithm(k_interactions=k_if_reg(c), seed=c.seed),
                      "DeepRIG", "Graph Autoencoder (GCN)", "Deep Graph AE", "#14b8a6", target_row=True),
        AlgorithmSpec("sfinn", "SFINN (GCN+FNN)", lambda c: SFINNAlgorithm(k_interactions=k_if_reg(c)), "SFINN",
                      "Shared Factor & Integrated NN", "Neighborhood GCN+FNN", "#eab308", target_row=True),
        AlgorithmSpec("notears", "NOTEARS (DAG)", lambda c: NOTEARSAlgorithm(k_interactions=k_if_reg(c)), "NOTEARS",
                      "Differentiable Acyclic DAG", "Continuous DAG Optimization", "#f59e0b", (3, 2)),
        AlgorithmSpec("inspre", "inspre (sparse inverse)", lambda c: InspreAlgorithm(k_interactions=c.k_est, perturbation_mode="regulator" if c.use_reg else "additive"),
                      "inspre (Sparse Inversion of the Response Matrix)", "Sparse Inversion of Total Effects",
                      "Cyclic Total-to-Direct", "#0d9488", (5, 2)),
        AlgorithmSpec("mean_difference", "Mean Difference", lambda c: MeanDifference(k_interactions=c.k_est),
                      "Mean Difference (control vs perturbed)", "Interventional Contrast Baseline",
                      "Total-Effect Baseline", "#78716c", (2, 4)),
        AlgorithmSpec("pearson", "Pearson |r|", lambda c: PearsonCorrelation(k_interactions=c.k_est),
                      "Pearson Correlation", "Co-expression Baseline", "Correlation Baseline", "#a8a29e", (2, 4)),
        AlgorithmSpec("grnboost", "GRNBoost-lite", lambda c: GRNBoostLite(k_interactions=c.k_est, seed=c.seed),
                      "GRNBoost (gradient-boosted importances)", "Boosted Tree Importance", "Boosted Trees", "#65a30d",
                      target_row=True),
        AlgorithmSpec("gene_features", "Gene-level features", lambda c: GeneFeatureBaseline(k_interactions=c.k_est),
                      "Gene-level Feature Baseline (hubness only)", "Pair-agnostic Baseline", "Hubness Control",
                      "#d6d3d1", (1, 4)),
    ]
    needs_adapter = {"s2g", "g2s", "vote", "score_vote", "ridge_cv", "fast", "netdeconv"}   # measured: grn_sim/docs/partial_screen_*
    for i, spec in enumerate(builtin):
        spec.partial = "adapter" if spec.key in needs_adapter else "native"
        spec.order = 10 * (i + 1)
        register(spec, replace=True)


_register_builtin()
