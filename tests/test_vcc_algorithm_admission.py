"""
Admission test of the Virtual Cell benchmark's algorithms: on noise-free effects (truth input; single cells for cell-level
methods) every algorithm must beat the hubness-only control by ADMISSION_RATIO on every data generator. An algorithm that cannot
do that on perfect data does not infer pairs at all and is dropped (vcc_benchmark.DROPPED); the dropped ones are re-tested here
so the decision stays justified.
"""
import numpy as np
import pytest

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation import vcc_benchmark as vb

ADMISSION_RATIO = 1.5
GENERATORS = ("linear_sem", "mechanistic", "boolean")
SEEDS = (301, 302, 303)


@pytest.fixture(scope="module")
def scores():
    from grn_sim.inference.interventional_baselines import GRNBoostLite
    from grn_sim.inference import registry

    lib = vb.algorithm_library()
    out = {}
    for gen in GENERATORS:
        for seed in SEEDS:
            run = vb.simulate_screen(vb.Setting(generator=gen, n_genes=60), seed)
            ace = sb.screen_inputs(run, "truth", seed)
            X, U = ace.to_xu()
            cells = vb._cell_matrix(run.adata, seed)
            for a in lib:
                rng = np.random.default_rng(seed)
                S = a.score(ace, X, U, rng, cells) if a.uses_cells else a.score(ace, X, U, rng)
                out.setdefault((gen, a.name), []).append(sb.evaluate_scores(run.screen, S)["aupr_direct"])
            # the dropped variants
            cpa = np.abs(registry.build_partial("cpa_lite", registry.BuildContext(k_est=3, use_reg=False)).fit(X, U).A_est)
            out.setdefault((gen, "cpa_lite"), []).append(max(sb.evaluate_scores(run.screen, cpa)["aupr_direct"],
                                                             sb.evaluate_scores(run.screen, cpa.T)["aupr_direct"]))
            prof = np.abs(GRNBoostLite(3, n_jobs=2).fit(X, U).A_est).T
            out.setdefault((gen, "grnboost_profiles"), []).append(sb.evaluate_scores(run.screen, prof)["aupr_direct"])
    return {k: float(np.mean(v)) for k, v in out.items()}


@pytest.mark.parametrize("name", [a.name for a in vb.algorithm_library() if a.family != "control"])
def test_every_benchmarked_algorithm_beats_the_hubness_control_on_perfect_data(scores, name):
    for gen in GENERATORS:
        ratio = scores[(gen, name)] / scores[(gen, vb.HUBNESS)]
        assert ratio >= ADMISSION_RATIO, f"{name} on {gen}: {ratio:.2f}x the hubness control"


@pytest.mark.parametrize("name", sorted(vb.DROPPED))
def test_dropped_algorithms_still_fail_admission(scores, name):
    ratios = [scores[(gen, name)] / scores[(gen, vb.HUBNESS)] for gen in GENERATORS]
    assert min(ratios) < ADMISSION_RATIO, f"{name} now passes admission ({ratios}); consider bringing it back"


def test_dropped_algorithms_are_not_in_the_library():
    names = {a.name for a in vb.algorithm_library()}
    assert not names & set(vb.DROPPED) and "cpa_lite" not in vb.REGISTRY_EXTRA


def test_cellbox_penalty_is_size_scaled_only_above_the_reference(monkeypatch):
    from grn_sim.inference import cellbox

    seen = {}

    def spy(self, X, U, l1, l2, epochs, device="cpu", **_):
        seen[X.shape[0]] = (l1, l2, epochs)
        return np.zeros((X.shape[0], X.shape[0])), {}

    monkeypatch.setattr(cellbox.CellBoxNeuralODE, "_fit_torch", spy)
    for n in (50, 100, 200):
        cellbox.CellBoxNeuralODE(epochs=70, device="cpu").fit(np.random.default_rng(0).normal(size=(n, n)), -np.eye(n))
    assert seen[50] == seen[100] == (0.003, 0.001, 70)
    assert seen[200][0] == pytest.approx(0.003 / 4) and seen[200][2] == cellbox.LARGE_NETWORK_EPOCHS


def test_cellbox_keeps_working_at_200_genes():
    pytest.importorskip("torch")
    lib = {a.name: a for a in vb.algorithm_library()}
    run = vb.simulate_screen(vb.Setting(generator="linear_sem", n_genes=200), 101)
    ace = sb.screen_inputs(run, "cells", 101, normalisation=vb.NORMALISATION)
    X, U = ace.to_xu()
    aupr = sb.evaluate_scores(run.screen, lib["cellbox"].score(ace, X, U, np.random.default_rng(0)))["aupr_direct"]
    assert aupr > 0.2                                    # 0.08 before the size-scaled penalty
