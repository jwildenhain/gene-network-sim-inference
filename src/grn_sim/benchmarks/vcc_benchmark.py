"""
Virtual Cell Challenge (VCC 2026) Comparative Benchmark & Simulation Engine Suite.
Compares Mechanistic Dynamical GRN Inference (S2G, G2S, Voting Ensemble, CellBox, etc.)
against Empirical Cross-Atlas Perturbation Transfer (AtlasShift).
Provides:
1. 18-Algorithm Comparative Evaluation on Network Topology Edge Recovery
2. In-Distribution vs Out-of-Distribution (OOD) Perturbation Generalization, with two response predictors per
   algorithm: the linear Jacobian operator and the structure -> Hill refit of ``hill_response.py`` (plus the
   mean-profile null and a true-structure oracle as reference rows)
3. Target-by-Target Dynamic Response Profile Simulator
4. Combinatorial Double-Perturbation & Synthetic Lethality Screen (Epistasis Matrix)
5. Single-Cell Perturb-seq Virtual Sequencer with 2D PCA Embeddings
6. Multi-Context Rewiring Sensitivity Decay Curve
"""

from typing import Dict, Any, List, NamedTuple, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA

from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator, SimulationExperiment

from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.by_chance import ByChance
from grn_sim.inference.atlasshift import AtlasShift
from grn_sim.inference.cellbox import CellBoxNeuralODE
from grn_sim.inference.genie3 import GENIE3Algorithm
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.cpa_lite import CPALite
from grn_sim.inference.network_deconv import NetworkDeconvolution
from grn_sim.inference.arachne import AracneAlgorithm
from grn_sim.inference.causalgrn import CausalGRNAlgorithm
from grn_sim.inference.dmdgrn import DMDGRNAlgorithm
from grn_sim.inference.deeprig import DeepRIGAlgorithm
from grn_sim.inference.sfinn import SFINNAlgorithm
from grn_sim.inference.notears import NOTEARSAlgorithm
from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.benchmarks.hill_response import (
    HillResponseModel,
    hill_forward_selection,
    infer_signed_structure,
    mean_profile_baseline,
)


# Accepted ranges for user-supplied parameters (the benchmark is O(n^3) in the gene count).
MIN_GENES = 4
MAX_GENES = 32
MAX_K_AV = 5.0
PERTURBATION_MODES = ("input", "knockdown")
KNOCKDOWN_EFFICIENCY = 0.8  # CRISPRi: the targeted gene's production is cut by 80% (VCC data: >80% knockdown in 83% of cells)

# Sign of the perturbation matrix each algorithm expects at fit time (web_run_rgn.m convention,
# shared with server.py and parameter_sweep.py).
FIT_INPUT_SIGN: Dict[str, float] = {"nir": -1.0, "fast": -1.0, "ridge_cv": -1.0}



class ResponseModel(NamedTuple):
    """A model whose A_est is turned into a response predictor R = sign * pinv(A_est) (X_pred = R @ u)."""
    key: str
    label: str            # shown in the UI "Compare Model" dropdown
    sign: float           # given the FIT_INPUT_SIGN above; A_est conventions differ per algorithm
    unsupported: str = ""  # reason A_est cannot be used as a steady-state Jacobian


# DMD-GRN solves A X = -U; NIR/Ridge are fit on -U; S2G/G2S/Vote on +U. CellBox is listed with +1 but
# learns near-zero weights at the default training budget, so its sign is not meaningful.
RESPONSE_MODELS: List[ResponseModel] = [
    ResponseModel("vote", "Voting Ensemble (Consensus)", 1.0),
    ResponseModel("cellbox", "CellBox Neural ODE", 1.0),
    ResponseModel("s2g", "S2G (Forward AIC)", 1.0),
    ResponseModel("g2s", "G2S (Backward AIC)", 1.0),
    ResponseModel("ridge_cv", "Kaggle RidgeCV (L2)", 1.0),
    ResponseModel("cpa_lite", "CPA-Lite (Latent)", 1.0,
                  unsupported="A_est is a latent-space projection (Decoder @ Encoder), not a steady-state Jacobian"),
    ResponseModel("nir", "NIR (Gardner SVD)", 1.0),
    ResponseModel("dmdgrn", "DMD-GRN (Koopman)", -1.0),
]
RESPONSE_SIGN: Dict[str, float] = {m.key: m.sign for m in RESPONSE_MODELS}

# Regulators kept per target when an algorithm's score matrix is turned into a structure for the Hill refit.
HILL_STRUCTURE_Q = 2

# Names for the algorithms that are not response models.
EXTRA_LABELS: Dict[str, str] = {
    "genie3": "GENIE3 (Random Forest)",
    "netdeconv": "Network Deconvolution",
    "fast": "FAST (Truncated SVD)",
    "arachne": "ARACNe (MI + DPI)",
    "causalgrn": "CausalGRN (DAG)",
    "sfinn": "SFINN (Symmetric NN)",
    "deeprig": "DeepRIG (Graph Autoencoder)",
    "notears": "NOTEARS (Acyclic DAG)",
    "by_chance": "By Chance (random structure)",
}

# Every algorithm in the summary table, in display order, with its paradigm label.
ALGORITHMS: List[Tuple[str, str]] = [
    ("vote", "Mechanistic Dynamical GRN (Consensus Voting)"),
    ("cellbox", "Mechanistic Dynamical GRN (Neural ODE)"),
    ("s2g", "Mechanistic Dynamical GRN (Forward AIC)"),
    ("g2s", "Mechanistic Dynamical GRN (Backward AIC)"),
    ("dmdgrn", "Mechanistic Dynamical GRN (Koopman Dynamics)"),
    ("ridge_cv", "L2 Tikhonov Regularized Inverse"),
    ("cpa_lite", "Latent Perturbation Space (CPA)"),
    ("genie3", "Tree Ensemble & Random Forest"),
    ("netdeconv", "Network Eigen-Deconvolution"),
    ("nir", "Linear SVD / NIR (Gardner 2003)"),
    ("fast", "FAST Truncated SVD (Thesis)"),
    ("causalgrn", "Causal DAG & Interventional Do-Calculus"),
    ("sfinn", "Symmetric Neighborhood Neural Net"),
    ("deeprig", "Graph Autoencoder (GAE + GCN)"),
    ("notears", "Continuous Constrained Acyclic DAG"),
    ("arachne", "Information-Theoretic & DPI"),
    ("atlasshift", "Empirical Cross-Atlas Transfer (VCC 2026)"),
    ("by_chance", "Null Baseline (Uniform Chance)"),
]


class HillModel(NamedTuple):
    """One entry of the structure -> Hill refit comparison. ``kind``: algorithm | search | reference."""
    key: str
    label: str
    kind: str


def hill_models() -> List[HillModel]:
    """All predictors of the Hill method, in display order: the model-matched search, every algorithm, the references."""
    labels = {m.key: m.label for m in RESPONSE_MODELS}
    labels.update(EXTRA_LABELS)
    models = [HillModel("hill_fs", "Hill forward search (model-matched structure)", "search")]
    models += [HillModel(key, f"{labels.get(key, key.upper())} structure", "algorithm") for key, _ in ALGORITHMS if key != "atlasshift"]
    models += [
        HillModel("null", "Mean-profile null (no structure)", "reference"),
        HillModel("self_prior", "Target-gene prior (own gene only, from the design)", "reference"),
        HillModel("oracle", "Oracle: true structure (upper bound)", "reference"),
    ]
    return models


def validate_vcc_params(num_genes: int, k_av: float, rewire_fraction: float, perturbation_mode: str = "input") -> None:
    """Raises ValueError if the benchmark parameters are outside the supported range."""
    if perturbation_mode not in PERTURBATION_MODES:
        raise ValueError(f"perturbation must be one of {list(PERTURBATION_MODES)}, got {perturbation_mode!r}")
    try:
        n = int(num_genes)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"num_genes must be an integer in [{MIN_GENES}, {MAX_GENES}]") from None
    if n != num_genes or not MIN_GENES <= n <= MAX_GENES:
        raise ValueError(f"num_genes must be an integer in [{MIN_GENES}, {MAX_GENES}], got {num_genes!r}")
    if not np.isfinite(k_av) or not 0.0 < k_av <= min(MAX_K_AV, n - 1):
        raise ValueError(f"k_av must be in (0, {min(MAX_K_AV, n - 1):g}] for {n} genes, got {k_av!r}")
    if not np.isfinite(rewire_fraction) or not 0.0 <= rewire_fraction <= 1.0:
        raise ValueError(f"rewire_fraction must be in [0, 1], got {rewire_fraction!r}")


def compute_profile_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Pearson correlation, Cosine similarity, and RMSE between two response vectors."""
    yt = np.asarray(y_true, dtype=float).flatten()
    yp = np.asarray(y_pred, dtype=float).flatten()

    norm_t = np.linalg.norm(yt)
    norm_p = np.linalg.norm(yp)
    cosine = float(np.dot(yt, yp) / (norm_t * norm_p)) if norm_t > 1e-12 and norm_p > 1e-12 else 0.0

    std_t = np.std(yt)
    std_p = np.std(yp)
    if std_t > 1e-12 and std_p > 1e-12:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = 0.0

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"pearson_r": round(pearson, 4), "cosine_sim": round(cosine, 4), "rmse": round(rmse, 4)}


def profile_scores(y_true: np.ndarray, y_pred: np.ndarray, target: int) -> Dict[str, float]:
    """``compute_profile_similarity`` plus ``offtarget_r``: Pearson r with the perturbed gene itself left out.

    In knockdown data the targeted gene's own drop is most of the profile, so r over all genes mostly measures
    whether a model knows which gene was hit; the off-target r measures the downstream regulatory response."""
    scores = compute_profile_similarity(y_true, y_pred)
    scores["offtarget_r"] = compute_profile_similarity(np.delete(np.asarray(y_true), target), np.delete(np.asarray(y_pred), target))["pearson_r"]
    return scores


def _rewire_edges(network: GeneNetwork, fraction: float, rng: np.random.Generator, shift: int = 1) -> None:
    """
    Moves ``fraction`` of the regulatory edges to a different target gene, in place.

    ``GeneNetwork`` zeroes Ka/Ki/n_k/n_i for gene pairs without an edge, so swapping the adjacency
    alone would create edges whose Hill terms degenerate to constants (x**0 == 1) and regulate
    nothing. A moved edge therefore carries the kinetic parameters of the edge it replaces, and it
    never lands on an existing edge, so the edge count is preserved.
    """
    n = network.n
    adj = network.topology.adjacency.copy()
    rows, cols = np.where(adj != 0)
    if fraction <= 0.0 or len(rows) == 0:
        return
    kinetics = (network.params.Ka, network.params.Ki, network.params.n_k, network.params.n_i)
    for k in rng.choice(len(rows), size=max(1, int(len(rows) * fraction)), replace=False):
        r, c_old = rows[k], cols[k]
        for offset in range(shift, shift + n):
            c_new = (c_old + offset) % n
            if c_new != r and adj[r, c_new] == 0:
                break
        else:
            continue  # regulator already hits every other gene: nowhere to move the edge
        adj[r, c_old] = 0
        adj[r, c_new] = 1 if rng.random() > 0.45 else -1
        for param in kinetics:
            param[r, c_new] = param[r, c_old]
            param[r, c_old] = 0.0
    network.topology.adjacency = adj


def _build_runners(k_est: int) -> Dict[str, Any]:
    """Fresh instances of every reverse-engineering algorithm except AtlasShift."""
    s2g = SimpleToGeneral()
    g2s = GeneralToSpecific()
    return {
        "vote": VotingEnsemble(s2g=s2g, g2s=g2s),
        "s2g": s2g,
        "g2s": g2s,
        "cellbox": CellBoxNeuralODE(epochs=40),
        "genie3": GENIE3Algorithm(n_estimators=30),
        "ridge_cv": KaggleMultiOutputRidge(),
        "cpa_lite": CPALite(epochs=30),
        "netdeconv": NetworkDeconvolution(),
        "nir": NIRAlgorithm(k_interactions=k_est),
        "fast": FastSVD(k_interactions=k_est),
        "arachne": AracneAlgorithm(),
        "causalgrn": CausalGRNAlgorithm(),
        "dmdgrn": DMDGRNAlgorithm(),
        "deeprig": DeepRIGAlgorithm(),
        "sfinn": SFINNAlgorithm(),
        "notears": NOTEARSAlgorithm(),
        "by_chance": ByChance(k_interactions=k_est),
    }


def _response_operator(A_est: np.ndarray, sign: float) -> Tuple[Optional[np.ndarray], str]:
    """
    Linear response operator R with X_pred[:, j] = R @ u_j, or (None, reason) when ``A_est``
    carries no usable Jacobian (a degenerate fit must not be reported as a prediction).
    """
    A = np.asarray(A_est, dtype=float)
    if not np.all(np.isfinite(A)):
        return None, "A_est contains non-finite values"
    if not np.any(A):
        return None, "fit returned an all-zero A_est (no interactions learned)"
    # Self-degradation is not identified when the diagonal is flat; regularise before inverting.
    ridge = 0.05 * np.eye(A.shape[0]) if np.all(np.diag(A) == 0) else 0.0
    return sign * np.linalg.pinv(A + ridge), ""


def _hill_held_out_responses(
    exp: SimulationExperiment,
    seen: List[int],
    held_fits: Dict[str, Any],
    fit_failures: Dict[str, str],
    X_train: np.ndarray,
    U_train: np.ndarray,
    mode: str = "input",
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Structure -> Hill refit -> steady-state response matrix (genes x perturbations) for every entry of ``hill_models()``.
    Only the ``seen`` experiments are used for any fit; the other columns of ``exp`` are only used for scoring.
    Returns (responses, reasons for the entries that could not be produced).
    """
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    x_seen, u_seen, x_ctrl = x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state
    responses: Dict[str, np.ndarray] = {}
    unavailable: Dict[str, str] = {}

    def refit(key: str, signed: np.ndarray) -> None:
        if not np.any(signed):
            unavailable[key] = "no regulators were selected"
            return
        try:
            responses[key] = HillResponseModel(signed, x_seen, u_seen, x_ctrl, mode=mode).response_matrix(exp.U)
        except Exception as exc:  # one failing refit must not take down the whole benchmark
            unavailable[key] = f"Hill refit failed: {exc}"

    for key, _ in ALGORITHMS:
        if key == "atlasshift":
            continue
        if key in fit_failures:
            unavailable[key] = fit_failures[key]
            continue
        A = np.asarray(held_fits[key].A_est, dtype=float)
        if not np.all(np.isfinite(A)) or not np.any(A):
            unavailable[key] = "fit returned no usable interaction scores"
            continue
        signed, _ = infer_signed_structure(A, held_fits[key].adjacency_pred, X_train, U_train, q=HILL_STRUCTURE_Q)
        refit(key, signed)

    refit("hill_fs", hill_forward_selection(x_seen, u_seen, x_ctrl, mode=mode))
    refit("oracle", np.asarray(exp.true_matrix).astype(int))
    responses["null"] = mean_profile_baseline(exp.X, seen)
    if mode == "knockdown":
        responses["self_prior"] = np.diag(np.diag(exp.U))  # -efficiency at the targeted gene, nothing elsewhere
    else:
        unavailable["self_prior"] = "the perturbed gene's own transcript is not set directly in this perturbation mode"
    return responses, unavailable


def _mean_metrics(scores: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ("pearson_r", "cosine_sim", "rmse", "offtarget_r")
    return {k: round(float(np.mean([s[k] for s in scores])) if scores else 0.0, 3) for k in keys}


def _virtual_single_cells(
    exp: SimulationExperiment,
    gene_names: List[str],
    seed: int,
    n_control: int = 20,
    cells_per_condition: int = 15,
    max_conditions: int = 8,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """Poisson-sampled virtual Perturb-seq cells: controls plus the first few perturbation conditions."""
    palette = ["#0284c7", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444", "#14b8a6"]
    rng = np.random.default_rng(seed)
    cells: List[np.ndarray] = []
    meta: List[Dict[str, str]] = []

    ctrl_rate = np.maximum(0.2, exp.reference_steady_state * 60.0)
    for _ in range(n_control):
        cells.append(rng.poisson(ctrl_rate))
        meta.append({"condition": "Control", "target": "None", "color": "#94a3b8"})

    for p_idx in range(min(max_conditions, len(gene_names))):
        # perturbed_states is (genes, perturbations): column p_idx is the state after perturbing gene p_idx.
        rate = np.maximum(0.2, exp.perturbed_states[:, p_idx] * 60.0)
        for _ in range(cells_per_condition):
            cells.append(rng.poisson(rate))
            meta.append({
                "condition": f"Perturb_{gene_names[p_idx]}",
                "target": gene_names[p_idx],
                "color": palette[p_idx % len(palette)],
            })
    return np.asarray(cells, dtype=float), meta


def _dual_perturbation_screen(sim: PerturbationSimulator, exp: SimulationExperiment) -> Tuple[np.ndarray, int]:
    """
    Simulates every Gi + Gj double perturbation on the ground-truth network and returns the
    deviation from the multiplicative null model, eps_ij = ||L_ij - (L_i + L_j)|| with
    L = log2(x_perturbed / x_basal) (plus the number of double perturbations that did not reach a
    steady state). Independent perturbations multiply fold changes, so their log fold changes add.

    Two choices keep eps meaningful:
    - Everything is re-simulated without measurement noise: the noise in ``exp.X`` alone puts a floor
      of ~0.18 under eps (growing with the gene count) that would swamp the classification thresholds.
    - Log ratios are used instead of ``X = x/x_ss - 1``: a lowly expressed gene (x_ss ~ 0.03) has ratio
      responses of +20..+40 and otherwise accounts for >99% of eps^2 on its own.
    """
    n = exp.network.n
    floor = 1e-6
    x_ss = np.maximum(exp.reference_steady_state, floor)

    knockdown = sim.perturbation_mode == "knockdown"

    def log_fold_change(u: np.ndarray) -> Tuple[np.ndarray, bool]:
        x, steady, _, _ = sim.solver.solve_with_diagnostics(
            exp.network,
            x0=exp.reference_steady_state.copy(),
            perturbation=None if knockdown else u,
            production_scale=np.maximum(1.0 + u, 1e-3) if knockdown else None,
        )
        return np.log2(np.maximum(x, floor) / x_ss), bool(steady)

    single = [log_fold_change(exp.U[:, i])[0] for i in range(n)]
    eps = np.zeros((n, n), dtype=float)
    unsteady = 0
    for i in range(n):
        for j in range(i + 1, n):
            dual, steady = log_fold_change(exp.U[:, i] + exp.U[:, j])
            eps[i, j] = eps[j, i] = float(np.linalg.norm(dual - (single[i] + single[j])))
            unsteady += int(not steady)
    return eps, unsteady


def run_vcc_comparative_benchmark(
    num_genes: int = 12,
    k_av: float = 1.8,
    seed: int = 2026,
    rewire_fractions: Optional[List[float]] = None,
    target_rewire_fraction: float = 0.15,
    perturbation_mode: str = "input",
) -> Dict[str, Any]:
    """
    Executes a multi-context comparative study:
    1. Simulates multiple source cell contexts (Context 1 = K562-like, Context 2 = HCT116-like, Context 3 = HEK293T-like).
    2. Simulates an unseen destination context (Context A) with customizable network rewiring.
    3. Evaluates 18 Reverse-Engineering Algorithms vs Empirical AtlasShift on:
       - Causal interaction matrix recovery (Coverage, FDR, Power, Total Errors)
       - Target-by-target transcriptional response prediction (Pearson r, Cosine Sim, RMSE)
       - In-Distribution vs Out-of-Distribution (OOD) perturbation generalization
       - Combinatorial epistasis & synthetic lethality screen (Gi x Gj)
       - Virtual single-cell sequencing with 2D PCA embedding
       - Context rewiring sensitivity curve (rho in [0%, 50%])
    """
    validate_vcc_params(num_genes, k_av, target_rewire_fraction, perturbation_mode)
    if rewire_fractions is None:
        rewire_fractions = [0.0, 0.1, 0.2, 0.35, 0.5]

    gene_names = [f"Gene_{i+1:02d}" for i in range(num_genes)]

    # 1. Base Shared Regulatory Network (Scale-free Barabási-Albert)
    base_net = GeneNetwork.create(
        n=num_genes,
        k_av=k_av,
        net_type=TopologyType.POWER_LAW,
        ratio=0.55,
        seed=seed,
    )
    # "input": +1 added to the regulator concentration of the perturbed gene (thesis convention, an activity boost).
    # "knockdown": CRISPRi-like, the perturbed gene's production is scaled by 1 - KNOCKDOWN_EFFICIENCY.
    sim = PerturbationSimulator(
        perturbation_factor=1.0,
        measurement_noise=0.01,
        perturbation_mode=perturbation_mode,
        knockdown_efficiency=KNOCKDOWN_EFFICIENCY,
    )

    # 2. Simulate 3 Source Atlases (slight basal expression differences simulating cell lines)
    source_experiments = []
    source_names = ["K562_GWPS", "HCT116_Perturb", "HEK293T_Perturb"]
    for idx, s_name in enumerate(source_names):
        s_net = GeneNetwork.create(
            n=num_genes,
            k_av=k_av,
            net_type=TopologyType.POWER_LAW,
            ratio=0.55,
            seed=seed,
        )
        s_net.params.v = np.clip(base_net.params.v * (1.0 + 0.15 * np.sin(idx + 1 + np.arange(num_genes))), 0.2, 4.0)
        exp_s = sim.run(s_net, seed=seed + idx * 10)
        source_experiments.append((s_name, exp_s))

    # 3. Simulate Target Context with Rewiring (Context A)
    target_net = GeneNetwork.create(
        n=num_genes,
        k_av=k_av,
        net_type=TopologyType.POWER_LAW,
        ratio=0.55,
        seed=seed,
    )
    _rewire_edges(target_net, target_rewire_fraction, np.random.default_rng(seed + 999), shift=1)
    target_exp = sim.run(target_net, seed=seed + 999)

    X_tgt = target_exp.X
    U_tgt = target_exp.U
    T_true = target_exp.true_matrix

    # 4. Fit AtlasShift with the 3 source atlases
    atlasshift = AtlasShift(de_threshold=0.15)
    for s_name, exp_s in source_experiments:
        atlasshift.add_source(s_name, exp_s.X, exp_s.U, weight=1.0)
    res_atlas = atlasshift.fit(X_tgt, U_tgt)

    # 5. Fit All Available Algorithms (edge recovery is scored against the true network)
    k_est = max(1, int(np.round(k_av)))
    runners = _build_runners(k_est)
    runners["atlasshift"] = atlasshift

    eval_dict: Dict[str, Any] = {}
    full_fits: Dict[str, Any] = {}
    for name, runner in runners.items():
        if name == "atlasshift":
            res = res_atlas
        else:
            res = runner.fit(X_tgt, FIT_INPUT_SIGN.get(name, 1.0) * U_tgt)
        full_fits[name] = res
        eval_dict[name] = evaluate_inference(T_true, res.adjacency_pred)

    # Summary table formatted for display (rows follow the ALGORITHMS order)
    summary_table = []
    for key, paradigm in ALGORITHMS:
        m = eval_dict[key]
        summary_table.append({
            "algorithm": key.upper(),
            "paradigm": paradigm,
            "coverage_pct": round(m.coverage_pct, 1),
            "false_positives_pct": round(m.false_positives_pct, 1),
            "power_pct": round(m.inferred_connections_pct, 1),
            "fdr_pct": round(m.fp_per_inferred_pct, 1),
            "total_errors": int(m.total_errors),
        })

    # 6. Response Profile Prediction: In-Distribution vs Out-of-Distribution (OOD)
    # Split targets into Seen (75%) vs Unseen (25% OOD). Every model is refit without the unseen
    # perturbation experiments, so OOD scores measure generalisation to perturbations the model never
    # observed. The linear solvers need square (n x n) data, so the held-out experiments are masked
    # (X = U = 0 contributes nothing to least squares) instead of dropped.
    # Two predictors are scored for the algorithms:
    #  - "linear": R = sign * pinv(A_est). A held-out gene's row of A is never identified, so this is exactly 0 OOD.
    #  - "hill":   the algorithm's regulators are kept (top-q per target) and a Hill-form model is refit on the seen
    #              experiments; the response is its steady state (see hill_response.py).
    # Reference rows: the mean response of the seen columns (no structure at all) and the true structure (oracle).
    n_targets = num_genes
    n_unseen = max(2, int(n_targets * 0.25))
    unseen_indices = list(range(n_targets - n_unseen, n_targets))
    seen_indices = [j for j in range(n_targets) if j not in unseen_indices]

    X_train, U_train = X_tgt.copy(), U_tgt.copy()
    X_train[:, unseen_indices] = 0.0
    U_train[:, unseen_indices] = 0.0

    held_fits: Dict[str, Any] = {}
    fit_failures: Dict[str, str] = {}
    for key, runner in _build_runners(k_est).items():
        try:
            held_fits[key] = runner.fit(X_train, FIT_INPUT_SIGN.get(key, 1.0) * U_train)
        except Exception as exc:  # one failing model must not take down the whole benchmark
            fit_failures[key] = f"fit failed on held-out data: {exc}"

    response_ops: Dict[str, np.ndarray] = {}
    unavailable: Dict[str, str] = {}
    for model in RESPONSE_MODELS:
        if model.unsupported:
            unavailable[model.key] = model.unsupported
        elif model.key in fit_failures:
            unavailable[model.key] = fit_failures[model.key]
        else:
            op, reason = _response_operator(held_fits[model.key].A_est, model.sign)
            if op is None:
                unavailable[model.key] = reason
            else:
                response_ops[model.key] = op

    hill_catalog = hill_models()
    hill_responses, hill_unavailable = _hill_held_out_responses(
        target_exp, seen_indices, held_fits, fit_failures, X_train, U_train, mode=perturbation_mode
    )

    target_profiles = []
    scores: Dict[Tuple[str, str], Dict[str, List[Dict[str, float]]]] = {}  # (method, key) -> {"seen": [...], "ood": [...]}

    def record(method: str, key: str, is_ood: bool, score: Dict[str, float]) -> None:
        scores.setdefault((method, key), {"seen": [], "ood": []})["ood" if is_ood else "seen"].append(score)

    for j in range(n_targets):
        y_true_j = X_tgt[:, j]
        is_ood = j in unseen_indices

        # AtlasShift prediction
        if not is_ood:
            y_pred_atlas = np.mean([exp_s.X[:, j] for _, exp_s in source_experiments], axis=0)
        else:
            # Unseen OOD perturbation: AtlasShift has no transfer observations!
            y_pred_atlas = np.zeros(num_genes, dtype=float)

        score_atlas = profile_scores(y_true_j, y_pred_atlas, j)
        record("reference", "atlasshift", is_ood, score_atlas)

        # Linear model predictions: X_pred = R @ U (R carries the per-algorithm sign convention)
        preds_dict: Dict[str, List[float]] = {}
        metrics_dict: Dict[str, Dict[str, float]] = {}
        for mod_name, op in response_ops.items():
            yp_m = op @ U_tgt[:, j]
            preds_dict[mod_name] = [round(float(v), 4) for v in yp_m]
            metrics_dict[mod_name] = profile_scores(y_true_j, yp_m, j)
            record("linear", mod_name, is_ood, metrics_dict[mod_name])

        # Hill refit predictions
        hill_preds: Dict[str, List[float]] = {}
        hill_metrics: Dict[str, Dict[str, float]] = {}
        for key, response in hill_responses.items():
            hill_preds[key] = [round(float(v), 4) for v in response[:, j]]
            hill_metrics[key] = profile_scores(y_true_j, response[:, j], j)
            record("hill", key, is_ood, hill_metrics[key])

        target_profiles.append({
            "target_idx": j,
            "target_gene": gene_names[j],
            "is_ood": is_ood,
            "status": "Out-of-Distribution (OOD Unseen)" if is_ood else "In-Distribution (Seen in Atlases)",
            "y_true": [round(float(v), 4) for v in y_true_j],
            "y_atlas": [round(float(v), 4) for v in y_pred_atlas],
            "predictions": preds_dict,
            "hill_predictions": hill_preds,
            "metrics": {
                "atlasshift": score_atlas,
                **metrics_dict,
            },
            "hill_metrics": hill_metrics,
        })

    def split_means(method: str, key: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        entry = scores.get((method, key), {"seen": [], "ood": []})
        return _mean_metrics(entry["seen"]), _mean_metrics(entry["ood"])

    avg_seen_vote, avg_ood_vote = split_means("linear", "vote")
    avg_seen_atlas, avg_ood_atlas = split_means("reference", "atlasshift")
    null_ood = split_means("hill", "null")[1]

    def table_row(method: str, key: str, label: str, kind: str, available: bool, reason: str) -> Dict[str, Any]:
        seen_m, ood_m = split_means(method, key) if available else ({}, {})
        return {
            "method": method,
            "key": key,
            "label": label,
            "kind": kind,
            "available": available,
            "reason": reason,
            "seen": seen_m,
            "held_out": ood_m,
            "held_out_minus_null": round(ood_m["pearson_r"] - null_ood["pearson_r"], 3) if available else None,
            "held_out_offtarget_minus_null": round(ood_m["offtarget_r"] - null_ood["offtarget_r"], 3) if available else None,
        }

    response_table = [table_row("linear", m.key, m.label, "algorithm", m.key in response_ops, unavailable.get(m.key, ""))
                      for m in RESPONSE_MODELS]
    response_table += [table_row("hill", m.key, m.label, m.kind, m.key in hill_responses, hill_unavailable.get(m.key, ""))
                       for m in hill_catalog]
    response_table.append(table_row("reference", "atlasshift", "AtlasShift (cross-atlas average)", "reference", True, ""))

    # 7. Combinatorial Epistasis & Synthetic Lethality Screen (Gi x Gj)
    # Every pair is simulated on the true network; eps_ij is the deviation from additive single effects.
    epistasis_matrix, unsteady_pairs = _dual_perturbation_screen(sim, target_exp)
    synergy_pairs: List[Dict[str, Any]] = []

    for i in range(num_genes):
        for j in range(i + 1, num_genes):
            score = float(epistasis_matrix[i, j])
            classification = "Additive / Independent"
            if score > 0.45:
                classification = "Synergistic / Synthetic Lethal"
            elif score > 0.25:
                classification = "Buffering / Epistatic Shift"

            if score > 0.25:
                synergy_pairs.append({
                    "gene_a": gene_names[i],
                    "gene_b": gene_names[j],
                    "pair": f"{gene_names[i]} × {gene_names[j]}",
                    "epistasis_score": round(score, 3),
                    "classification": classification,
                })

    synergy_pairs.sort(key=lambda x: x["epistasis_score"], reverse=True)

    # 8. Single-Cell Perturb-seq Virtual Sequencer & 2D PCA Embeddings
    # Generates ~140 virtual cells across Control + perturbations with Poisson count noise
    cells_mat, cell_metadata = _virtual_single_cells(target_exp, gene_names, seed + 777)
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(cells_mat)
    var_explained = [round(float(v * 100), 1) for v in pca.explained_variance_ratio_]

    single_cell_pca = {
        "num_cells": len(cell_metadata),
        "variance_explained": var_explained,
        "cells": [
            {
                "x": round(float(pca_coords[i, 0]), 2),
                "y": round(float(pca_coords[i, 1]), 2),
                "condition": cell_metadata[i]["condition"],
                "target": cell_metadata[i]["target"],
                "color": cell_metadata[i]["color"],
            }
            for i in range(len(cell_metadata))
        ],
    }

    # 9. Context Rewiring Sensitivity Sweep (rho in [0..0.5])
    # Transfer of the model fitted on the target context (all perturbations) to rewired contexts.
    # A model without a usable Jacobian yields an empty series rather than a stand-in for another model.
    rewire_decay: Dict[str, List[float]] = {
        "rewire_rates": [int(r * 100) for r in rewire_fractions],
        "mechanistic_r": [],
        "atlasshift_r": [],
        "cellbox_r": [],
        "ridge_r": [],
        "hill_r": [],
    }
    sweep_ops = {
        key: _response_operator(full_fits[name].A_est, RESPONSE_SIGN[name])[0]
        for key, name in (("mechanistic_r", "vote"), ("cellbox_r", "cellbox"), ("ridge_r", "ridge_cv"))
    }
    atlas_pred = np.mean([exp_s.X for _, exp_s in source_experiments], axis=0)

    # Hill refit on the Voting Ensemble's structure, fitted on every experiment of the target context.
    hill_sweep_model: Optional[HillResponseModel] = None
    vote_fit = full_fits["vote"]
    if np.any(vote_fit.A_est) and np.all(np.isfinite(vote_fit.A_est)):
        signed_vote, _ = infer_signed_structure(vote_fit.A_est, vote_fit.adjacency_pred, X_tgt, U_tgt, q=HILL_STRUCTURE_Q)
        if np.any(signed_vote):
            x_abs_all = (X_tgt + 1.0) * target_exp.reference_steady_state[:, None]
            hill_sweep_model = HillResponseModel(signed_vote, x_abs_all, U_tgt, target_exp.reference_steady_state, mode=perturbation_mode)

    for rf in rewire_fractions:
        net_rf = GeneNetwork.create(n=num_genes, k_av=k_av, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)
        _rewire_edges(net_rf, rf, np.random.default_rng(seed + int(rf * 1000)), shift=2)
        exp_rf = sim.run(net_rf, seed=seed + 100)

        # Evaluate mean Pearson r across targets
        r_atlas = [compute_profile_similarity(exp_rf.X[:, j], atlas_pred[:, j])["pearson_r"] for j in range(num_genes)]
        rewire_decay["atlasshift_r"].append(round(float(np.mean(r_atlas)), 3))
        for key, op in sweep_ops.items():
            if op is None:
                continue
            r_model = [
                compute_profile_similarity(exp_rf.X[:, j], op @ exp_rf.U[:, j])["pearson_r"]
                for j in range(num_genes)
            ]
            rewire_decay[key].append(round(float(np.mean(r_model)), 3))
        if hill_sweep_model is not None:
            r_hill = [
                compute_profile_similarity(exp_rf.X[:, j], hill_sweep_model.response(exp_rf.U[:, j]))["pearson_r"]
                for j in range(num_genes)
            ]
            rewire_decay["hill_r"].append(round(float(np.mean(r_hill)), 3))

    return {
        "status": "success",
        "parameters": {
            "num_genes": num_genes,
            "k_av": k_av,
            "target_rewire_fraction": target_rewire_fraction,
            "gene_names": gene_names,
            "sources": source_names,
            "seen_targets": n_targets - n_unseen,
            "unseen_ood_targets": n_unseen,
            "response_models": [
                {"key": m.key, "label": m.label, "available": m.key in response_ops, "reason": unavailable.get(m.key, "")}
                for m in RESPONSE_MODELS
            ],
            "hill_models": [
                {"key": m.key, "label": m.label, "kind": m.kind, "available": m.key in hill_responses,
                 "reason": hill_unavailable.get(m.key, "")}
                for m in hill_catalog
            ],
            "hill_structure_q": HILL_STRUCTURE_Q,
            "perturbation_mode": perturbation_mode,
            "knockdown_efficiency": KNOCKDOWN_EFFICIENCY if perturbation_mode == "knockdown" else None,
        },
        "summary_table": summary_table,
        "response_table": response_table,
        "in_distribution": {
            "mechanistic": avg_seen_vote,
            "atlasshift": avg_seen_atlas,
            "hill_search": split_means("hill", "hill_fs")[0],
            "null": split_means("hill", "null")[0],
        },
        "out_of_distribution": {
            "mechanistic": avg_ood_vote,
            "atlasshift": avg_ood_atlas,
            "hill_search": split_means("hill", "hill_fs")[1],
            "null": split_means("hill", "null")[1],
            "oracle": split_means("hill", "oracle")[1],
        },
        "rewire_decay": rewire_decay,
        "target_profiles": target_profiles,
        "combinatorial_screen": {
            "gene_names": gene_names,
            "epistasis_matrix": [[round(float(v), 3) for v in row] for row in epistasis_matrix],
            "top_synergy_pairs": synergy_pairs[:10],
            "unsteady_pairs": unsteady_pairs,
        },
        "single_cell_pca": single_cell_pca,
    }
