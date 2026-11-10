"""
Statistical benchmark of network-inference algorithms on simulated Perturb-seq screens ("virtual cell" experiments).

One *setting* fixes how the experiment is simulated (pathway architecture, network size, perturbation type, number of
double perturbations, how the perturbed genes are selected); one *simulation* is a setting plus a seed (a fresh network,
a fresh screen). Every algorithm gets the same cells and is scored against the simulator's direct edges (AUPR over the
rows of perturbed genes, plus precision at k and the AUPR against the total-effect truth). Each simulation also scores a
**random ranking** with the same pipeline, so every algorithm is compared with chance *on the same data*.

``aggregate`` turns the per-simulation rows of any filter into one line per algorithm: mean, spread, lift over the base rate
(the AUPR a random ranking has in expectation), the share of simulations won against the random ranking, a paired one-sided
Wilcoxon test against it (Benjamini-Hochberg corrected across the algorithms of the group), and a verdict. An algorithm is
"better than random" only if its mean lift is at least ``MIN_LIFT`` and the corrected test is significant; otherwise the
verdict says it is not appropriate for this kind of experiment. Beating chance is a low bar here (a gene's hubness alone
carries signal), so a second paired test compares each algorithm with the **hubness-only control**: an algorithm that beats
chance but not this control is ``marginal``: it ranks hub genes, not specific pairs.

S2G and G2S run in their regression form (``matlab_mode=False``: rows of ``A X = -U`` regressed on the conditions), because the
MATLAB form needs exactly one experiment per gene and so cannot see partial screens or double perturbations; on a full
single-perturbation screen the regression form is also the more accurate of the two (AUPR 0.61 vs 0.47 for S2G, one network).

Double perturbations are used by the algorithms that accept multi-gene experiments (columns of the repository's (X, U) pair
with two non-zero entries in U); inspre, mean difference and the scoring of the ACE estimate stay on single perturbations.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.perturbseq.ace import ACEEstimate, log_normalise

MIN_LIFT = 1.5          # an algorithm must beat the base rate by this factor on average ...
ALPHA = 0.05            # ... and win the corrected paired test against the random ranking at this level
RANDOM = "random"
# Effect estimates use median-ratio size factors: total-count normalisation turns a knockdown of a highly expressed gene
# into effects on genes that do not depend on it (grn_sim.perturbseq.ace.log_normalise; tests/test_perturbation_dependencies.py).
NORMALISATION = "median_ratio"
HUBNESS = "gene_features"      # non-causal control: how much a gene moves anything x how much a gene moves under anything
K = 3

PATHWAYS = {
    "replogle_like": "Modular (Perturb-seq calibrated)",
    "thesis": "Scale-free (thesis kinetics)",
    "cascade": "Signalling cascade",
    "feedforward": "Feed-forward loops",
    "lineage": "Lineage master regulators",
    "regulon": "Overlapping regulons",
}
MODES = {"knockdown": "CRISPRi knockdown", "knockout": "CRISPR knockout", "input": "Activation (thesis input; CRISPRa for the statistical generators)"}
SELECTIONS = {"full": "All genes perturbed", "effective": "Effective genes", "random": "Random genes",
              "expressed": "Most expressed genes", "hubs": "Hub regulators (oracle)"}
GENERATORS = {
    "mechanistic": "Mechanistic (Hill kinetics, as in panel 1)",
    "linear_sem": "Linear SEM",
    "nonlinear_scm": "Nonlinear additive-noise SCM",
    "nb_copula": "Negative-binomial copula",
    "boolean": "Probabilistic Boolean network",
}
NOISE_LEVELS = {"low": 0.5, "medium": 1.0, "high": 2.0}      # multiples of each generator's calibrated noise
FACTORS = ("generator", "noise", "pathway", "n_genes", "mode", "pairs", "selection")

# Registered algorithms scored from the cells' effect estimates. Their output orientation on screen data was fixed once on
# noise-free effects of separate networks (seeds 1001-1003, mechanistic and linear-SEM data, 60 genes): these are scored
# transposed (A[target, regulator]); every other one as returned. AtlasShift needs source atlases (it is compared in the
# cross-atlas section) and ``by_chance`` duplicates the random ranking, so both are left out here.
REGISTRY_EXTRA = ("fast", "vote", "score_vote", "genie3", "notears", "cellbox", "netdeconv", "dmdgrn", "causalgrn", "arachne",
                  "deeprig", "sfinn")
# Dropped after the admission test (tests/test_vcc_algorithm_admission.py: an algorithm must beat the hubness-only control on
# noise-free effects): CPA-Lite (AUPR 0.049 vs 0.07 for the control on noise-free 100-gene screens) and GRNBoost on effect
# profiles (0.046-0.086, control level). GRNBoost is kept on single cells, the input it is designed for.
DROPPED = {"cpa_lite": "fails the admission test: no better than the hubness-only control even on noise-free data",
           "grnboost_profiles": "GRNBoost on per-perturbation effect profiles is at the hubness level; the cell-level variant is used"}
GRNBOOST_CELLS = 1500
TRANSPOSED_ON_SCREENS = frozenset({"cellbox", "netdeconv", "dmdgrn", "causalgrn", "arachne", "deeprig", "sfinn"})


@dataclass(frozen=True)
class Setting:
    generator: str = "mechanistic"  # how the data are produced (see GENERATORS)
    noise: str = "medium"           # low | medium | high (see NOISE_LEVELS)
    pathway: str = "replogle_like"
    n_genes: int = 100
    mode: str = "knockdown"
    pairs: int = 0                  # number of double perturbations (0 = none)
    selection: str = "full"        # full | effective | random | expressed | hubs
    fraction: float = 0.5           # share of genes perturbed when selection != full
    cells: int = 40                 # cells per perturbation

    def key(self) -> str:
        return "|".join(str(v) for v in (self.generator, self.noise, self.pathway, self.n_genes, self.mode, self.pairs,
                                         self.selection, self.fraction, self.cells))

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Setting":
        return cls(**{k: row.get(k, f.default) for k, f in cls.__dataclass_fields__.items()})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -- double perturbations ------------------------------------------------------------------------------------------------

def pair_columns(adata: Any, ace: ACEEstimate, control_label: str = "non-targeting",
                 min_cells: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, U) columns for the double perturbations of ``adata``: X[:, c] the shift of every gene (log-normalised mean against the
    non-targeting cells), U[:, c] with the two perturbed genes' own shifts in their rows. Shapes (n, n_pairs); empty
    arrays when the screen has no double perturbation with enough cells.
    """
    names = list(ace.gene_names)
    index = {g: i for i, g in enumerate(names)}
    lab = adata.obs["perturbation"].astype(str).values
    wanted = sorted({x for x in np.unique(lab) if "+" in x})
    n = len(names)
    if not wanted:
        return np.zeros((n, 0)), np.zeros((n, 0))
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    Y = log_normalise(Xd, method=NORMALISATION, reference=Xd[lab == control_label].mean(axis=0))
    m0 = Y[lab == control_label].mean(axis=0)
    cols_x, cols_u = [], []
    for name in wanted:
        a, b = name.split("+", 1)
        sel = lab == name
        if a not in index or b not in index or sel.sum() < min_cells:
            continue
        shift = (Y[sel].mean(axis=0) - m0).astype(float)
        u = np.zeros(n)
        u[index[a]], u[index[b]] = shift[index[a]], shift[index[b]]
        cols_x.append(shift)
        cols_u.append(u)
    if not cols_x:
        return np.zeros((n, 0)), np.zeros((n, 0))
    return np.stack(cols_x, axis=1), np.stack(cols_u, axis=1)


# -- algorithms ----------------------------------------------------------------------------------------------------------

@dataclass
class Algorithm:
    name: str
    label: str
    score: Callable[..., np.ndarray]          # (ace, X, U, rng) -> S[regulator, target]
    uses_pairs: bool = False                  # accepts double perturbations as extra (X, U) columns
    family: str = ""
    uses_cells: bool = False                  # scores single cells (log-normalised, cells x genes) instead of (X, U)


def _legacy(factory: Callable[[], Any], target_regulator: bool) -> Callable[..., np.ndarray]:
    def run(ace: ACEEstimate, X: np.ndarray, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        res = factory().fit(X, U)
        A = np.abs(res.A_est)
        return A.T if target_regulator else A
    return run


def algorithm_library(max_genes: int = 250) -> List[Algorithm]:
    """The algorithms of the benchmark. Every one runs from the cells' effect estimates in seconds."""
    from grn_sim.inference.g2s import GeneralToSpecific
    from grn_sim.inference.inspre import InspreAlgorithm
    from grn_sim.inference.interventional_baselines import (GeneFeatureBaseline, MeanDifference,
                                                            PearsonCorrelation)
    from grn_sim.inference.nir import NIRAlgorithm
    from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
    from grn_sim.inference.s2g import SimpleToGeneral

    def inspre(ace: ACEEstimate, X: np.ndarray, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.abs(InspreAlgorithm(K).fit_ace(ace, alpha=None).A_est)

    def rand(ace: ACEEstimate, X: np.ndarray, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(size=(X.shape[0], X.shape[0]))

    return [
        Algorithm(RANDOM, "Random ranking (chance)", rand, family="control"),
        Algorithm("mean_difference", "Mean difference (ACE)", _legacy(lambda: MeanDifference(K), False), family="interventional baseline"),
        Algorithm("inspre", "inspre (inverse of ACE)", inspre, family="interventional"),
        Algorithm("s2g", "S2G (simple-to-general)", _legacy(lambda: SimpleToGeneral(matlab_mode=False), True), True, "mechanistic"),
        Algorithm("nir", "NIR (network identification by regression)", _legacy(lambda: NIRAlgorithm(k_interactions=K), True), True, "mechanistic"),
        Algorithm("ridge_cv", "Ridge CV (self allowed)",
                  _legacy(lambda: KaggleMultiOutputRidge(k_interactions=K, allow_self_regulation=True), True), True, "mechanistic"),
        Algorithm("g2s", "G2S (general-to-specific, LOO)",
                  _legacy(lambda: GeneralToSpecific(drop_rule="loo", max_interactions=2 * K, matlab_mode=False), True), True,
                  "mechanistic"),
        Algorithm("pearson", "Pearson correlation", _legacy(lambda: PearsonCorrelation(K), False), True, "association"),
        Algorithm("grnboost", "GRNBoost (tree importances on single cells)", _grnboost_cells, family="association",
                  uses_cells=True),
        Algorithm("gene_features", "Gene hubness only (no pair signal)", _legacy(lambda: GeneFeatureBaseline(K), False), family="control"),
    ] + [_registered(key) for key in REGISTRY_EXTRA]


FAMILIES = {"fast": "mechanistic", "vote": "mechanistic", "score_vote": "mechanistic", "netdeconv": "association",
            "arachne": "association", "genie3": "association", "dmdgrn": "dynamical", "notears": "causal",
            "causalgrn": "causal", "cellbox": "deep learning", "deeprig": "deep learning", "sfinn": "deep learning",
            "cpa_lite": "deep learning"}


def _grnboost_cells(ace: ACEEstimate, X: np.ndarray, U: np.ndarray, rng: np.random.Generator,
                    cells: Optional[np.ndarray] = None) -> np.ndarray:
    """GRNBoost2-style: per target gene, boosted-tree importances of the other genes over single cells (S[regulator, target])."""
    from grn_sim.inference.interventional_baselines import GRNBoostLite

    if cells is None:
        raise ValueError("GRNBoost needs the cells of the screen")
    W = GRNBoostLite(K, n_jobs=2, seed=int(rng.integers(0, 10 ** 6))).fit(cells.T, np.zeros((cells.shape[1], 1))).A_est
    return np.abs(W).T                                                    # A_est is [target, regulator]


def _registered(key: str) -> Algorithm:
    """A registry algorithm on single perturbations: built for partial screens, fitted on (X, sign U), oriented per the table."""
    from grn_sim.inference import registry

    spec = registry.get(key)

    def run(ace: ACEEstimate, X: np.ndarray, U: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        algo = registry.build_partial(key, registry.BuildContext(k_est=K, use_reg=False, seed=int(rng.integers(0, 10 ** 6))))
        A = np.abs(np.nan_to_num(np.real(algo.fit(X, (-1.0 if spec.negate_u_in_additive else 1.0) * U).A_est)))
        return A.T if key in TRANSPOSED_ON_SCREENS else A

    return Algorithm(key, spec.long_name or spec.label, run, False, FAMILIES.get(key, "other"))


# -- one simulation ------------------------------------------------------------------------------------------------------

def _cell_matrix(adata: Any, seed: int, n_cells: int = GRNBOOST_CELLS) -> np.ndarray:
    """A random subsample of the screen's cells, log-normalised like the effect estimates (cells x genes)."""
    Xd = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    lab = adata.obs["perturbation"].astype(str).values
    Y = log_normalise(Xd, method=NORMALISATION, reference=Xd[lab == "non-targeting"].mean(axis=0))
    idx = np.random.default_rng(seed + 31).choice(len(Y), size=min(n_cells, len(Y)), replace=False)
    return np.asarray(Y[np.sort(idx)], dtype=float)


def simulate_screen(setting: Setting, seed: int, adjacency: Optional[np.ndarray] = None, network: Any = None) -> Any:
    """
    The screen of one simulation. ``mechanistic`` builds the preset network with its Hill kinetics (the panel-1 engine; pass
    ``network`` to use a given one); the statistical generators use the same preset's graph for the same seed (or
    ``adjacency``) with their calibrated parameters scaled by the noise level.
    """
    from grn_sim.perturbseq.design import ScreenDesign
    from grn_sim.perturbseq.pipeline import build_network, run_screen

    if setting.generator not in GENERATORS:
        raise ValueError(f"generator must be one of {', '.join(GENERATORS)}")
    if setting.noise not in NOISE_LEVELS:
        raise ValueError(f"noise must be one of {', '.join(NOISE_LEVELS)}")
    factor = NOISE_LEVELS[setting.noise]
    n_pert = max(2, int(round(setting.fraction * setting.n_genes)))
    design = ScreenDesign.full() if setting.selection == "full" else ScreenDesign.partial(n_pert, setting.selection)
    if setting.generator == "mechanistic":
        net = network if network is not None else build_network(setting.pathway, setting.n_genes, seed=seed)
        return run_screen(net, mode=setting.mode, cells_per_condition=setting.cells, seed=seed, design=design,
                          pairs=setting.pairs or None, guide_efficiency_mean=0.92,
                          cell_options={"extrinsic_gene_sd": 0.15 * factor, "extrinsic_shared_sd": 0.1 * factor})
    from grn_sim.statgen import preset_graph, run_stat_screen
    from grn_sim.statgen.calibration import params_for

    A = adjacency if adjacency is not None else preset_graph(setting.pathway, setting.n_genes, seed=seed)
    return run_stat_screen(A, setting.generator, params_for(setting.generator, factor), mode=setting.mode,
                           pairs=setting.pairs or None, cells_per_condition=setting.cells, design=design, seed=seed)


def run_simulation(setting: Setting, seed: int, algorithms: Optional[Sequence[Algorithm]] = None,
                   adjacency: Optional[np.ndarray] = None, network: Any = None) -> List[Dict[str, Any]]:
    """Builds the network, simulates the screen and scores every algorithm; one row per algorithm (see ``simulate_screen``
    for ``adjacency`` / ``network``)."""
    algorithms = list(algorithms) if algorithms is not None else algorithm_library()
    t0 = time.time()
    run = simulate_screen(setting, seed, adjacency=adjacency, network=network)
    t_sim = time.time() - t0
    ace = sb.screen_inputs(run, "cells", seed, normalisation=NORMALISATION)
    Xp, Up = pair_columns(run.adata, ace) if setting.pairs else (np.zeros((setting.n_genes, 0)),) * 2
    X1, U1 = ace.to_xu()
    X2, U2 = (np.hstack([X1, Xp]), np.hstack([U1, Up])) if Xp.shape[1] else (X1, U1)
    n_edges = int((np.asarray(run.screen.adjacency) != 0).sum())
    cells = _cell_matrix(run.adata, seed) if any(a.uses_cells for a in algorithms) else None
    rows: List[Dict[str, Any]] = []
    for alg in algorithms:
        rng = np.random.default_rng(seed * 7919 + 13)
        X, U = (X2, U2) if alg.uses_pairs else (X1, U1)
        t = time.time()
        try:
            S = alg.score(ace, X, U, rng, cells) if alg.uses_cells else alg.score(ace, X, U, rng)
            res = sb.evaluate_scores(run.screen, S)
            status = "ok"
        except Exception as exc:                                    # an algorithm that fails scores as missing, not as chance
            res, status = {}, f"error: {type(exc).__name__}: {exc}"
        rows.append({**setting.to_dict(), "seed": int(seed), "algorithm": alg.name, "status": status,
                     "aupr_direct": res.get("aupr_direct"), "p_at_k_direct": res.get("p_at_k_direct"),
                     "base_direct": res.get("base_direct"), "aupr_total": res.get("aupr_total"),
                     "base_total": res.get("base_total"), "seconds": round(time.time() - t, 2),
                     "n_perturbed": len(ace.nodes), "n_edges": n_edges, "n_pair_columns": int(Xp.shape[1]),
                     "uses_pairs": bool(alg.uses_pairs and Xp.shape[1]), "sim_seconds": round(t_sim, 1)})
    return rows


# -- statistics ----------------------------------------------------------------------------------------------------------

def _bh(p: Sequence[float]) -> List[float]:
    p_arr = np.asarray(p, dtype=float)
    order = np.argsort(p_arr)
    ranked = p_arr[order] * len(p_arr) / (np.arange(len(p_arr)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.minimum(ranked, 1.0)
    return out.tolist()


def _paired_p(diff: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p-value that the paired differences are positive."""
    from scipy.stats import wilcoxon

    d = diff[np.isfinite(diff)]
    if d.size < 3 or np.allclose(d, 0.0):
        return 1.0
    try:
        return float(wilcoxon(d, alternative="greater").pvalue)
    except ValueError:
        return 1.0


def matches(row: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for key, want in filters.items():
        if want in (None, "", "all"):
            continue
        if str(row.get(key)) != str(want):
            return False
    return True


def aggregate(rows: Iterable[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None, n_boot: int = 400) -> Dict[str, Any]:
    """
    Per-algorithm statistics over all simulations that match ``filters`` (setting fields, ``"all"`` = no filter).

    A simulation is one (setting, seed); the random ranking of the same simulation is the paired reference. Returns
    ``{"n_simulations", "n_settings", "algorithms": [...]}`` with, per algorithm: ``n``, ``mean/sd/median/q25/q75`` of
    the direct AUPR, ``base`` (mean base rate), ``lift`` (mean AUPR / mean base rate), ``random_mean``, ``win_rate``
    (simulations where it beat the random ranking), ``p_value`` / ``q_value`` (paired Wilcoxon, BH over the algorithms),
    ``q_hub`` (the same test against the hubness-only control), ``verdict`` (``better`` / ``marginal`` / ``not_better`` /
    ``worse``; ``reference`` for the random ranking, ``control`` for the hubness-only baseline), ``rank`` with its 95% bootstrap
    interval (``rank_low`` / ``rank_high``) and the 95% bootstrap interval of the mean (``ci_low`` / ``ci_high``) and the per-simulation ``values`` for plotting.
    """
    rows = [r for r in rows if matches(r, filters or {}) and r.get("status", "ok") == "ok" and r.get("aupr_direct") is not None]
    sims: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        key = (Setting.from_row(r).key(), int(r["seed"]))
        sims.setdefault(key, {})[r["algorithm"]] = r
    reference = {k: v[RANDOM]["aupr_direct"] for k, v in sims.items() if RANDOM in v}
    names = sorted({a for v in sims.values() for a in v})
    out: List[Dict[str, Any]] = []
    for name in names:
        keys = [k for k, v in sims.items() if name in v and k in reference]
        if not keys:
            continue
        vals = np.array([sims[k][name]["aupr_direct"] for k in keys], dtype=float)
        ref = np.array([reference[k] for k in keys], dtype=float)
        base = np.array([sims[k][name]["base_direct"] for k in keys], dtype=float)
        pk = np.array([sims[k][name]["p_at_k_direct"] for k in keys], dtype=float)
        tot = np.array([sims[k][name]["aupr_total"] for k in keys], dtype=float)
        mean_base = float(base.mean()) or 1e-12
        n = len(vals)
        out.append({
            "algorithm": name, "n": n, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(vals)), "q25": float(np.quantile(vals, 0.25)), "q75": float(np.quantile(vals, 0.75)),
            "base": float(base.mean()), "lift": float(vals.mean() / mean_base), "random_mean": float(ref.mean()),
            "p_at_k": float(pk.mean()), "aupr_total": float(tot.mean()),
            "win_rate": float(np.mean(vals > ref)), "p_value": 1.0 if name == RANDOM else _paired_p(vals - ref),
            "p_worse": 1.0 if name == RANDOM else _paired_p(ref - vals),
            "p_hub": 1.0, "q_hub": 1.0, "hub_mean": None,
            "uses_pairs": bool(any(sims[k][name].get("uses_pairs") for k in keys)),
            "values": [round(float(v), 4) for v in vals],
        })
    tested = [o for o in out if o["algorithm"] not in (RANDOM, HUBNESS)]
    for o, q in zip(tested, _bh([o["p_value"] for o in tested]) if tested else []):
        o["q_value"] = q
    hub_keys = [k for k, v in sims.items() if HUBNESS in v and k in reference]
    for o in tested:
        keys = [k for k in hub_keys if o["algorithm"] in sims[k]]
        if keys:
            a = np.array([sims[k][o["algorithm"]]["aupr_direct"] for k in keys], dtype=float)
            h = np.array([sims[k][HUBNESS]["aupr_direct"] for k in keys], dtype=float)
            o["p_hub"], o["hub_mean"] = _paired_p(a - h), float(h.mean())
    for o, q in zip(tested, _bh([o["p_hub"] for o in tested]) if tested else []):
        o["q_hub"] = q
        if o["q_value"] < ALPHA and o["lift"] >= MIN_LIFT:
            o["verdict"] = "better" if (o["hub_mean"] is None or q < ALPHA) else "marginal"
        elif o["p_worse"] < ALPHA:
            o["verdict"] = "worse"
        else:
            o["verdict"] = "not_better"
    for o in out:
        if o["algorithm"] == RANDOM:
            o["q_value"], o["verdict"] = 1.0, "reference"
        elif o["algorithm"] == HUBNESS:
            o["q_value"], o["verdict"] = o["p_value"], "control"
    out.sort(key=lambda o: (-o["mean"], o["algorithm"]))
    _bootstrap_ranks(out, sims, n_boot)
    return {"n_simulations": len(sims), "n_settings": len({k[0] for k in sims}), "algorithms": out,
            "min_lift": MIN_LIFT, "alpha": ALPHA}


def _bootstrap_ranks(out: List[Dict[str, Any]], sims: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]], n_boot: int) -> None:
    """
    Adds ``rank`` (by mean AUPR), a 95% bootstrap interval of the mean (``ci_low`` / ``ci_high``) and of the rank
    (``rank_low`` / ``rank_high``). Simulations are resampled as a whole, so every algorithm sees the same resampled set and
    the ranks stay paired; simulations an algorithm is missing from count once less for it (its mean is over the rest).
    """
    names = [o["algorithm"] for o in out]
    keys = sorted(sims)
    if not names or not keys:
        return
    M = np.full((len(keys), len(names)), np.nan)
    for i, k in enumerate(keys):
        for j, a in enumerate(names):
            if a in sims[k]:
                M[i, j] = sims[k][a]["aupr_direct"]
    rng = np.random.default_rng(20261110)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    means = np.stack([np.nanmean(M[ix], axis=0) for ix in idx]) if n_boot else np.zeros((0, len(names)))
    means = np.nan_to_num(means, nan=-1.0)
    ranks = (-means).argsort(axis=1).argsort(axis=1) + 1
    for j, o in enumerate(out):
        o["rank"] = j + 1
        if n_boot:
            o["ci_low"], o["ci_high"] = (float(v) for v in np.quantile(means[:, j], [0.025, 0.975]))
            o["rank_low"], o["rank_high"] = (int(v) for v in np.quantile(ranks[:, j], [0.025, 0.975], method="nearest"))


def factor_levels(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """The values every factor takes in ``rows`` (for the filter menus)."""
    levels: Dict[str, set] = {f: set() for f in FACTORS}
    for r in rows:
        for f in FACTORS:
            levels[f].add(r.get(f))
    return {f: sorted(v, key=lambda x: (isinstance(x, str), x)) for f, v in levels.items()}
