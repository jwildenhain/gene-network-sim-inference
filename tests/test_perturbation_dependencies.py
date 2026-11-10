"""
Perturbation dependencies of every data generator (mechanistic and statistical): a perturbation of gene a must move exactly the
genes downstream of a in the true graph, a double perturbation can only interact on genes downstream of both partners, and the
cell-level analysis must not invent dependencies on genes that do not depend on the perturbed one.
"""
import numpy as np
import pytest

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation import vcc_benchmark as vb

GENERATORS = list(vb.GENERATORS)


def reachable(A: np.ndarray) -> np.ndarray:
    """R[a, j]: j is downstream of a (a directed path a -> ... -> j)."""
    R = np.asarray(A) != 0
    P = R.copy()
    for _ in range(len(R)):
        nxt = P | ((P.astype(int) @ R.astype(int)) > 0)
        if np.array_equal(nxt, P):
            break
        P = nxt
    return P


def screen(gen: str, n: int = 40, seed: int = 4, pairs: int = 0, cells: int = 60, measurement: bool = True):
    if gen == "mechanistic":
        from grn_sim.perturbseq.pipeline import build_network, run_screen
        return run_screen(build_network("replogle_like", n, seed=seed), cells_per_condition=cells, seed=seed,
                          guide_efficiency_mean=0.92, pairs=pairs or None, measurement=measurement)
    from grn_sim.statgen import preset_graph, run_stat_screen
    from grn_sim.statgen.calibration import params_for
    return run_stat_screen(preset_graph("replogle_like", n, seed=seed), gen, params_for(gen), pairs=pairs or None,
                           cells_per_condition=cells, seed=seed, measurement=measurement)


@pytest.fixture(scope="module", params=GENERATORS)
def run_with_pairs(request):
    return request.param, screen(request.param, pairs=40)


def test_single_perturbations_move_only_downstream_genes(run_with_pairs):
    gen, run = run_with_pairs
    scr = run.screen
    R = reachable(scr.adjacency)
    for k, c in enumerate(scr.conditions):
        if len(c) != 1:
            continue
        a = c[0]
        moved = np.abs(scr.total_effects[:, k]) > 1e-6
        moved[a] = False
        outside = moved & ~R[a]
        assert not outside.any(), f"{gen}: perturbing {a} moves non-downstream genes {np.flatnonzero(outside)}"


def test_downstream_genes_do_respond(run_with_pairs):
    gen, run = run_with_pairs
    scr = run.screen
    R = reachable(scr.adjacency)
    direct_hits, direct_total = 0, 0
    for k, c in enumerate(scr.conditions):
        if len(c) == 1:
            a = c[0]
            targets = np.flatnonzero(scr.adjacency[a] != 0)
            direct_total += len(targets)
            direct_hits += int((np.abs(scr.total_effects[targets, k]) > 1e-6).sum())
    assert direct_total and direct_hits / direct_total > 0.8, f"{gen}: direct targets rarely respond"
    assert R.sum() > 0


def test_double_perturbations_interact_only_downstream_of_both(run_with_pairs):
    gen, run = run_with_pairs
    scr = run.screen
    R = reachable(scr.adjacency)
    pairs = [k for k, c in enumerate(scr.conditions) if len(c) == 2]
    assert pairs, "no double perturbations were simulated"
    for p_idx, k in enumerate(pairs):
        a, b = scr.conditions[k]
        dev = np.abs(scr.pair_deviation[:, p_idx])
        moved = np.abs(scr.total_effects[:, k]) > 1e-6
        assert not (moved & ~(R[a] | R[b]))[[g for g in range(scr.n_genes) if g not in (a, b)]].any()
        not_both = ~(R[a] & R[b])
        not_both[[a, b]] = False
        assert dev[not_both].max(initial=0.0) < 1e-6, f"{gen}: pair {a}+{b} interacts on genes not downstream of both"


def test_linear_models_have_no_interactions_and_nonlinear_ones_do():
    lin = screen("linear_sem", pairs=40).screen
    non = screen("nonlinear_scm", pairs=40).screen
    assert lin.epistasis.max() < 1e-8
    assert non.epistasis.max() > 1e-3


@pytest.mark.parametrize("gen", GENERATORS)
def test_cells_do_not_show_dependencies_on_independent_genes(gen):
    """Among gene pairs with no path a -> j, at most 5% may come out significant (BH q < 0.05) from the cells."""
    run = screen(gen, n=40, seed=6)
    R = reachable(run.screen.adjacency)
    ace = sb.screen_inputs(run, "cells", 6, normalisation=vb.NORMALISATION)
    sig = ace.qvalue < 0.05
    null = ~R[ace.nodes]
    null[np.arange(len(ace.nodes)), ace.nodes] = False
    assert sig[null].mean() <= 0.05, f"{gen}: {sig[null].mean():.3f} of independent pairs look dependent"
    strong = R[ace.nodes] & (np.abs(run.screen.total_effects[:, [run.screen.single_index[g] for g in ace.nodes]].T) >= 0.5)
    if strong.any():
        assert sig[strong].mean() >= 0.5, f"{gen}: strong true dependencies are missed"


def test_total_count_normalisation_creates_dependencies_that_median_ratio_removes():
    run = screen("mechanistic", n=60, seed=1)
    R = reachable(run.screen.adjacency)
    rates = {}
    for norm in ("total", "median_ratio"):
        ace = sb.screen_inputs(run, "cells", 1, normalisation=norm)
        null = ~R[ace.nodes]
        null[np.arange(len(ace.nodes)), ace.nodes] = False
        rates[norm] = float((ace.qvalue < 0.05)[null].mean())
    assert rates["median_ratio"] <= 0.05 < rates["total"]


def test_double_knockouts_leave_single_perturbation_data_intact_with_true_labels():
    """Adding doubles must not degrade single-knockout effects by itself; the loss in the benchmark comes from guide calling."""
    from grn_sim.perturbseq.ace import estimate_ace

    lib = {a.name: a for a in vb.algorithm_library()}
    scores = {}
    for pairs in (0, 50):
        vals = {True: [], False: []}
        for seed in (1, 2, 3):
            run = vb.simulate_screen(vb.Setting(generator="linear_sem", pairs=pairs), seed)
            for called in (True, False):
                ace = estimate_ace(run.adata, gene_names=list(run.screen.gene_names), use_called_guide=called, min_cells=20,
                                   seed=seed, normalisation=vb.NORMALISATION)
                X, U = ace.to_xu()
                vals[called].append(sb.evaluate_scores(run.screen, lib["mean_difference"].score(ace, X, U, None))["aupr_direct"])
        scores[pairs] = {k: float(np.mean(v)) for k, v in vals.items()}
    assert abs(scores[50][False] - scores[0][False]) < 0.04                 # true labels: no real change
    assert scores[0][True] - scores[50][True] > scores[0][False] - scores[50][False]   # called guides: the dilution
