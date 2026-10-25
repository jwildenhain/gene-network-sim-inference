"""Held-out response prediction (benchmarks/hill_response.py): the structure -> Hill refit -> steady-state strategy."""

from functools import lru_cache

import numpy as np
import pytest

from grn_sim.benchmarks import vcc_benchmark as vcc
from grn_sim.benchmarks.hill_response import (
    HillResponseModel,
    hill_factor,
    hill_forward_selection,
    infer_signed_structure,
    linear_refit,
    mean_profile_baseline,
    pearson,
    top_q_support,
)
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator


def experiment(n=12, seed=3, noise=0.0):
    net = GeneNetwork.create(n=n, k_av=1.8, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)
    return PerturbationSimulator(perturbation_factor=1.0, measurement_noise=noise).run(net, seed=seed)


def held_out_score(model, exp, held):
    R = model.response_matrix(exp.U)
    return np.mean([pearson(R[:, j], exp.X[:, j]) for j in held])


# --- the building blocks --------------------------------------------------------------------------

def test_hill_factors_have_the_right_shape():
    eff = np.linspace(0.01, 10, 50)
    act, inh = hill_factor(eff, 1.5, 3.0, +1), hill_factor(eff, 1.5, 3.0, -1)
    assert np.all(np.diff(act) > 0) and np.all(np.diff(inh) < 0)
    assert 1.0 <= act.min() and act.max() <= 2.0 and 0.0 <= inh.min() and inh.max() <= 1.0
    assert hill_factor(1.5, 1.5, 3.0, +1) == pytest.approx(1.5) and hill_factor(1.5, 1.5, 3.0, -1) == pytest.approx(0.5)


def test_top_q_support_keeps_the_best_regulators_per_target_and_no_self_edges():
    scores = np.array([[9.0, 1.0, 0.0], [5.0, 9.0, 3.0], [2.0, 4.0, 9.0]])  # scores[l, k]
    S = top_q_support(scores, 1)
    assert S.sum(axis=0).tolist() == [1, 1, 1] and np.trace(S) == 0
    assert S[1, 0] == 1 and S[2, 1] == 1 and S[1, 2] == 1  # best non-self regulator of each target


def test_linear_refit_recovers_known_coefficients_and_ignores_masked_columns():
    rng = np.random.default_rng(0)
    n, m = 4, 30
    F_true = np.zeros((n, n)); F_true[1, 0], F_true[2, 1] = 0.6, -0.4
    Z = rng.normal(size=(n, m))
    X = F_true @ Z
    U = Z - X                                    # so that X + U = Z
    X[:, -5:] = 0.0; U[:, -5:] = 0.0             # masked experiments
    support = (F_true != 0).T.astype(int)         # support[l, k]
    assert np.allclose(linear_refit(support, X, U, ridge=1e-9), F_true, atol=1e-4)


def test_structure_inference_picks_the_orientation_that_explains_the_training_data():
    exp = experiment(n=10, seed=5)
    truth = np.abs(exp.true_matrix).astype(float)          # [regulator, target]
    A_good = (truth + 0.01).astype(float)
    S_a, _ = infer_signed_structure(A_good, None, exp.X, exp.U, q=2)
    S_b, _ = infer_signed_structure(A_good.T.copy(), None, exp.X, exp.U, q=2)
    assert np.array_equal(S_a != 0, S_b != 0), "a transposed score matrix leads to the same structure"
    tp = ((S_a != 0) & (truth != 0)).sum()
    assert tp / max(1, (S_a != 0).sum()) > 0.6


def test_mean_profile_baseline_is_leave_one_out_for_training_columns():
    X = np.arange(12, dtype=float).reshape(3, 4)
    R = mean_profile_baseline(X, [0, 1, 2])
    assert np.allclose(R[:, 3], X[:, :3].mean(axis=1))                 # held-out column: mean of the training columns
    assert np.allclose(R[:, 0], X[:, 1:3].mean(axis=1))                # training column 0: mean of the others


def test_vectorised_solver_matches_a_plain_loop():
    exp = experiment(n=8, seed=7)
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    model = HillResponseModel(exp.true_matrix.astype(int), x_abs, exp.U, exp.reference_steady_state)

    def loop_production(eff):
        out = np.empty(model.n)
        for k in range(model.n):
            value = np.exp(model._log_c[k])
            for p in range(model._idx.shape[1]):
                if model._valid[k, p]:
                    value *= hill_factor(eff[model._idx[k, p]], model._K[k, p], model._m[k, p], model._sign[k, p])
            out[k] = value
        return out

    eff = exp.reference_steady_state * 1.3
    assert np.allclose(model.production(eff), loop_production(eff))


# --- the strategy itself ---------------------------------------------------------------------------

def test_training_perturbations_are_reproduced_with_the_true_structure():
    exp = experiment(n=10, seed=2)
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    model = HillResponseModel(exp.true_matrix.astype(int), x_abs, exp.U, exp.reference_steady_state)
    R = model.response_matrix(exp.U)
    assert np.mean([pearson(R[:, j], exp.X[:, j]) for j in range(10)]) > 0.9


def test_held_out_perturbations_are_predicted_from_the_other_experiments_with_the_true_structure():
    """The measured oracle result: r about 0.90 for 4 held-out perturbations of 16 genes without noise."""
    scores = []
    for seed in (1, 2, 3):
        exp = experiment(n=16, seed=seed)
        held = [12, 13, 14, 15]; seen = [j for j in range(16) if j not in held]
        x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
        model = HillResponseModel(exp.true_matrix.astype(int), x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state)
        scores.append(held_out_score(model, exp, held))
    assert np.mean(scores) > 0.6, scores


def test_held_out_columns_are_never_used_for_fitting():
    exp = experiment(n=12, seed=4)
    held = [10, 11]; seen = list(range(10))
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    base = HillResponseModel(exp.true_matrix.astype(int), x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state)
    tampered = x_abs.copy(); tampered[:, held] = 123.0                   # garbage in the held-out experiments
    other = HillResponseModel(exp.true_matrix.astype(int), tampered[:, seen], exp.U[:, seen], exp.reference_steady_state)
    assert np.allclose(base.response_matrix(exp.U), other.response_matrix(exp.U))


def rewired_experiment(n, seed, noise):
    """The benchmark's target context: a scale-free network with 15% of its edges moved."""
    net = GeneNetwork.create(n=n, k_av=1.8, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)
    vcc._rewire_edges(net, 0.15, np.random.default_rng(seed + 999), shift=1)
    return PerturbationSimulator(perturbation_factor=1.0, measurement_noise=noise).run(net, seed=seed + 999)


@lru_cache(maxsize=None)
def paired_held_out_scores(seeds, n=16, k_held=4, noise=0.01):
    """Held-out Pearson r on identical rewired networks: null, oracle (true structure), forward search, G2S structure."""
    out = {"null": [], "oracle": [], "search": [], "g2s": []}
    for seed in seeds:
        exp = rewired_experiment(n, seed, noise)
        held = list(range(n - k_held, n)); seen = [j for j in range(n) if j not in held]
        x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
        x_seen, u_seen, x_ctrl = x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state

        def score(R):
            return float(np.mean([pearson(R[:, j], exp.X[:, j]) for j in held]))

        out["null"].append(score(mean_profile_baseline(exp.X, seen)))
        out["oracle"].append(held_out_score(HillResponseModel(exp.true_matrix.astype(int), x_seen, u_seen, x_ctrl), exp, held))
        out["search"].append(held_out_score(HillResponseModel(hill_forward_selection(x_seen, u_seen, x_ctrl), x_seen, u_seen, x_ctrl), exp, held))
        Xt, Ut = exp.X.copy(), exp.U.copy(); Xt[:, held] = 0; Ut[:, held] = 0
        res = vcc._build_runners(2)["g2s"].fit(Xt, Ut)
        signed, _ = infer_signed_structure(res.A_est, res.adjacency_pred, Xt, Ut, q=2)
        out["g2s"].append(held_out_score(HillResponseModel(signed, x_seen, u_seen, x_ctrl), exp, held))
    return {k: np.array(v) for k, v in out.items()}


def test_paired_study_the_true_structure_is_far_above_the_null():
    """
    24-network study (scripts/holdout_study.py; N=16, 4 held out, noise 0.01): null +0.015, oracle +0.475 (23/24 wins),
    forward search +0.167 (17/24 over the null), G2S structure +0.054 (10/24). Six networks are far too few to rank the
    search against the null (the null alone ranges from 0.01 to 0.40 here), so only the large gaps are asserted.
    """
    r = paired_held_out_scores(tuple(range(1, 7)))
    assert r["oracle"].mean() > r["null"].mean() + 0.2, (r["oracle"], r["null"])
    assert r["oracle"].mean() > r["search"].mean(), "the true structure stays an upper bound"


def test_a_structure_taken_from_one_algorithm_is_not_claimed_to_beat_the_null():
    """Recorded negative result: G2S's own structure + Hill is within noise of the null (paired difference +0.04 +/- 0.03)."""
    r = paired_held_out_scores(tuple(range(1, 7)))
    assert abs(r["g2s"].mean() - r["null"].mean()) < 0.25, "this changed: update docs/HOLDOUT_RESEARCH.md and the README"


def test_forward_search_recovers_a_good_part_of_the_true_structure():
    exp = experiment(n=16, seed=3, noise=0.0)
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    signed = hill_forward_selection(x_abs, exp.U, exp.reference_steady_state)
    truth = exp.true_matrix != 0
    found = signed != 0
    assert np.trace(found) == 0 and found.sum(axis=0).max() <= 3
    assert (found & truth).sum() / max(1, found.sum()) > 0.5, "precision of the selected regulators"
    assert (found & truth).sum() / max(1, truth.sum()) > 0.4, "recall of the true regulators"


def test_degenerate_structures_do_not_crash():
    exp = experiment(n=6, seed=1)
    x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
    empty = HillResponseModel(np.zeros((6, 6), dtype=int), x_abs, exp.U, exp.reference_steady_state)
    R = empty.response_matrix(exp.U)
    assert R.shape == (6, 6) and np.all(np.isfinite(R))
    assert np.allclose(R, R[:, [0]]), "without regulators every perturbation gets the same (mean-level) prediction"
    assert pearson(np.zeros(5), np.arange(5.0)) == 0.0 and pearson(np.array([np.nan, 1, 2]), np.arange(3.0)) == 0.0
