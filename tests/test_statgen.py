"""Statistical data generators (grn_sim.statgen): their own maths, independence from the kinetics, calibration, and the verdicts."""
import numpy as np
import pytest

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.statgen import MODELS, ModelParams, condition_screen, make_model, preset_graph, run_stat_screen
from grn_sim.statgen.calibration import CALIBRATED, params_for
from grn_sim.statgen.models import KNOCKOUT_RESIDUAL, shift

LOG2E = 1.0 / np.log(2.0)


@pytest.fixture(scope="module")
def graph():
    return preset_graph("replogle_like", 40, seed=3)


# -- independence from panel 1's engine ------------------------------------------------------------------------------------

def test_statgen_sources_do_not_use_the_kinetics():
    import pathlib

    import grn_sim.statgen as pkg
    for f in pathlib.Path(pkg.__file__).parent.glob("*.py"):
        text = f.read_text()
        for banned in ("simulator.kinetics", "simulator.steady_state", "simulator.ode_solver", "SteadyStateSolver",
                       "KineticsKernel", "simulate_conditions"):
            assert banned not in text, f"{f.name} uses {banned}"


def test_statistical_screens_run_with_the_kinetics_disabled(graph, monkeypatch):
    from grn_sim.simulator import kinetics, steady_state

    def boom(*a, **k):
        raise AssertionError("the statistical generators must not touch the Hill kinetics")

    monkeypatch.setattr(steady_state.SteadyStateSolver, "solve_with_diagnostics", boom)
    monkeypatch.setattr(kinetics, "production_rates", boom)
    for gen in MODELS:
        run = run_stat_screen(graph, gen, CALIBRATED[gen], cells_per_condition=8, seed=1)
        assert run.adata.n_obs == 8 * (graph.shape[0] + 1)


def test_the_same_seed_gives_the_mechanistic_preset_graph():
    from grn_sim.perturbseq.pipeline import build_network

    for preset in ("replogle_like", "cascade", "regulon"):
        A = preset_graph(preset, 50, seed=8)
        assert np.array_equal(np.sign(A), np.sign(build_network(preset, 50, seed=8).topology.adjacency))


# -- each model's own maths ------------------------------------------------------------------------------------------------

def test_linear_sem_interventional_means_are_analytic(graph):
    m = make_model("linear_sem", graph, ModelParams(effect_scale=0.3, noise=0.2), "knockdown", np.random.default_rng(0))
    M = np.linalg.inv(np.eye(m.n) - m.W)
    for a in (0, 7, 21):
        z = m.log_mean((a,), np.array([[0.8]]))[0]
        assert np.allclose(z, np.log(0.2) * M[a], atol=1e-10)
    scr = condition_screen(m, genes=[3], pairs=None)
    assert np.allclose(scr.total_effects[:, 0], np.log(1 - 0.8) * M[3] * LOG2E, atol=1e-10)


def test_linear_sem_double_perturbations_add_up_with_no_epistasis(graph):
    m = make_model("linear_sem", graph, ModelParams(effect_scale=0.3), "knockdown", np.random.default_rng(1))
    scr = condition_screen(m, pairs=10, seed=2)
    assert scr.epistasis is not None and len(scr.epistasis) == 10
    assert np.max(scr.epistasis) < 1e-8


def test_nonlinear_scm_has_epistasis_and_solves_its_fixed_point(graph):
    m = make_model("nonlinear_scm", graph, ModelParams(effect_scale=0.6), "knockdown", np.random.default_rng(1))
    D = m._delta((4,), np.array([[0.9]]))
    z = m.log_mean((4,), np.array([[0.9]]))
    assert np.max(np.abs(m._g(z) @ m.W + D - z)) < 1e-6
    scr = condition_screen(m, pairs=10, seed=2)
    assert np.max(scr.epistasis) > 1e-3                      # saturation makes double effects non-additive


def test_knockout_removes_nearly_all_production(graph):
    assert shift("knockout", np.array([1.0]))[0] == pytest.approx(np.log(KNOCKOUT_RESIDUAL))
    assert shift("knockdown", np.array([0.0]))[0] == 0.0 and shift("input", np.array([1.0]))[0] > 0
    m = make_model("linear_sem", graph, ModelParams(), "knockout", np.random.default_rng(0))
    scr = condition_screen(m, genes=[5])
    assert scr.total_effects[5, 0] < np.log2(0.05)


def test_nb_copula_marginals_have_the_set_mean_and_dispersion():
    A = np.zeros((6, 6))
    A[0, 1] = A[1, 2] = 1
    m = make_model("nb_copula", A, ModelParams(noise=0.5, shared_noise=0.0), "knockdown", np.random.default_rng(3))
    x = m.sample((), np.zeros((40000, 0)), np.random.default_rng(4))
    mu = np.exp(m.log_base)
    assert np.allclose(x.mean(axis=0), mu, rtol=0.03)
    assert np.allclose(x.var(axis=0) / mu ** 2, m.dispersion, rtol=0.1)
    from scipy.stats import spearmanr

    target = m.chol @ m.chol.T                             # the copula's correlation, implied by the graph
    rho = spearmanr(x).correlation
    assert abs(target[0, 1]) > 0.01 and target[3, 4] == pytest.approx(0.0, abs=1e-9)
    assert rho[0, 1] == pytest.approx(6 / np.pi * np.arcsin(target[0, 1] / 2), abs=0.03)   # Spearman rho of a Gaussian copula
    assert abs(rho[3, 4]) < 0.03


def test_boolean_knockdown_and_noise_behave(graph):
    quiet = make_model("boolean", graph, ModelParams(effect_scale=1.2, noise=0.2), "knockdown", np.random.default_rng(5))
    noisy = make_model("boolean", graph, ModelParams(effect_scale=1.2, noise=4.0), "knockdown", np.random.default_rng(5))
    own = quiet.log_mean((2,), np.array([[1.0]]))[0, 2]
    assert own < np.log(0.2)                                 # a full knockdown removes > 80% of the gene's expression
    def eff(m):
        return np.abs(condition_screen(m, genes=range(10)).total_effects).sum()

    assert eff(noisy) < eff(quiet)                           # flips wash out the propagated effects


@pytest.mark.parametrize("gen", sorted(MODELS))
def test_screens_are_reproducible_per_seed(graph, gen):
    a = run_stat_screen(graph, gen, CALIBRATED[gen], cells_per_condition=6, seed=9)
    b = run_stat_screen(graph, gen, CALIBRATED[gen], cells_per_condition=6, seed=9)
    c = run_stat_screen(graph, gen, CALIBRATED[gen], cells_per_condition=6, seed=10)
    assert (a.adata.X != b.adata.X).nnz == 0 and (a.adata.X != c.adata.X).nnz > 0


def test_partial_designs_and_pairs_work_for_every_generator(graph):
    from grn_sim.perturbseq.design import ScreenDesign

    for gen in MODELS:
        for sel in ("effective", "random", "hubs", "expressed"):
            run = run_stat_screen(graph, gen, CALIBRATED[gen], cells_per_condition=5, design=ScreenDesign.partial(10, sel),
                                  pairs=4, seed=2)
            singles = [c for c in run.screen.conditions if len(c) == 1]
            assert 0 < len(singles) <= 10 and any(len(c) == 2 for c in run.screen.conditions)


# -- calibration and noise levels -----------------------------------------------------------------------------------------

def test_noise_levels_scale_only_the_noise():
    for gen in CALIBRATED:
        lo, mid, hi = (params_for(gen, f) for f in (0.5, 1.0, 2.0))
        assert lo.noise == pytest.approx(mid.noise / 2) and hi.noise == pytest.approx(2 * mid.noise)
        assert lo.effect_scale == mid.effect_scale == hi.effect_scale
    with pytest.raises(ValueError):
        params_for("mechanistic")


@pytest.mark.parametrize("gen", sorted(MODELS))
def test_calibrated_screens_look_like_perturb_seq(gen):
    """Mean over three seeds (single screens vary: the nonlinear model's phenotype share ranges 0.10-0.29 by seed)."""
    keys = ("on_target_knockdown", "fraction_with_phenotype", "affected_pair_density")
    stats = [vb.simulate_screen(vb.Setting(generator=gen, n_genes=100), seed).realism()["statistics"] for seed in (777, 778, 779)]
    st = {k: float(np.mean([s[k] for s in stats])) for k in keys}
    assert 0.75 <= st["on_target_knockdown"] <= 0.95
    assert 0.12 <= st["fraction_with_phenotype"] <= 0.5
    assert 0.01 <= st["affected_pair_density"] <= 0.07


def test_more_noise_makes_the_data_harder():
    lib = {a.name: a for a in vb.algorithm_library()}
    scores = {}
    for noise in ("low", "high"):
        vals = []
        for seed in (31, 32):
            run = vb.simulate_screen(vb.Setting(generator="linear_sem", noise=noise, n_genes=60), seed)
            ace = sb.screen_inputs(run, "cells", seed)
            X, U = ace.to_xu()
            vals.append(sb.evaluate_scores(run.screen, lib["mean_difference"].score(ace, X, U, None))["aupr_direct"])
        scores[noise] = np.mean(vals)
    assert scores["high"] < scores["low"]


# -- verdict machinery is calibrated --------------------------------------------------------------------------------------

def test_pure_noise_algorithms_are_rarely_called_better_than_random():
    rng = np.random.default_rng(0)
    false_calls = 0
    trials = 200
    for _ in range(trials):
        rows = []
        for seed in range(1, 11):
            base = 0.05
            for name in (vb.RANDOM, vb.HUBNESS, "noise_alg"):
                v = base + rng.normal(0, 0.01) + (0.05 if name == vb.HUBNESS else 0.0)
                rows.append({**vb.Setting().to_dict(), "seed": seed, "algorithm": name, "status": "ok", "aupr_direct": v,
                             "p_at_k_direct": 0.1, "base_direct": base, "aupr_total": 0.1})
        verdict = {a["algorithm"]: a["verdict"] for a in vb.aggregate(rows, n_boot=0)["algorithms"]}
        false_calls += verdict["noise_alg"] in ("better", "marginal")
    assert false_calls / trials <= 0.06


def test_a_real_method_on_noise_free_data_is_called_better():
    lib = {a.name: a for a in vb.algorithm_library()}
    rows = []
    for seed in range(1, 9):
        run = run_stat_screen(preset_graph("replogle_like", 30, seed=seed), "linear_sem", CALIBRATED["linear_sem"],
                              cells_per_condition=5, seed=seed)
        ace = sb.screen_inputs(run, "truth", seed)
        X, U = ace.to_xu()
        rng = np.random.default_rng(seed)
        for name in ("s2g", vb.RANDOM, vb.HUBNESS):
            res = sb.evaluate_scores(run.screen, lib[name].score(ace, X, U, rng))
            rows.append({**vb.Setting(generator="linear_sem").to_dict(), "seed": seed, "algorithm": name, "status": "ok",
                         **{k: res[k] for k in ("aupr_direct", "p_at_k_direct", "base_direct", "aupr_total")}})
    by = {a["algorithm"]: a for a in vb.aggregate(rows)["algorithms"]}
    assert by["s2g"]["verdict"] == "better" and by["s2g"]["rank"] == 1


def test_rank_and_mean_intervals_contain_the_point_estimates():
    rows = []
    rng = np.random.default_rng(1)
    for seed in range(1, 21):
        for name, mu in ((vb.RANDOM, 0.05), (vb.HUBNESS, 0.1), ("a", 0.4), ("b", 0.39), ("c", 0.2)):
            rows.append({**vb.Setting().to_dict(), "seed": seed, "algorithm": name, "status": "ok", "base_direct": 0.05,
                         "aupr_direct": mu + rng.normal(0, 0.03), "p_at_k_direct": 0.1, "aupr_total": 0.1})
    by = {a["algorithm"]: a for a in vb.aggregate(rows)["algorithms"]}
    for a in by.values():
        assert a["ci_low"] <= a["mean"] <= a["ci_high"] and a["rank_low"] <= a["rank"] <= a["rank_high"]
    assert by["a"]["rank_high"] >= 2 and by["b"]["rank_low"] <= 1         # a and b cannot be told apart
    assert by["c"]["rank_low"] == by["c"]["rank_high"] == 3                 # c is clearly third


@pytest.mark.parametrize("gen", sorted(MODELS))
def test_every_pathway_gives_bounded_expression(gen):
    """Heavy-tailed weights on acyclic pathways once produced log expression of ~140 (non-finite after measurement)."""
    from grn_sim.evaluation.vcc_benchmark import PATHWAYS
    from grn_sim.statgen.models import MAX_EDGE_WEIGHT

    for pathway in PATHWAYS:
        for seed in (1, 6):
            A = preset_graph(pathway, 100, seed=seed)
            for mode in ("knockdown", "knockout", "input"):
                m = make_model(gen, A, params_for(gen, 2.0), mode, np.random.default_rng(seed + 101))
                assert np.abs(m.W).max() <= MAX_EDGE_WEIGHT + 1e-12
                z = m.log_mean((0,), np.array([[1.0]]))
                assert np.isfinite(z).all() and np.abs(z).max() < 12, f"{gen} {pathway} {mode}: |log fold| {np.abs(z).max():.1f}"
                if gen != "boolean":                       # the Boolean sampler is slow; its states are bounded by design
                    x = m.sample((3,), np.full((50, 1), 0.9), np.random.default_rng(seed))
                    assert np.isfinite(x).all() and x.max() / max(x.min(), 1e-300) < 1e12
