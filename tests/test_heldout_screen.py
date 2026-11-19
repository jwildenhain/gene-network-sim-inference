"""Held-out-perturbation benchmark: metric definitions, predictor behaviour on structured data, the cross-validation loop."""
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from grn_sim.evaluation import heldout_screen as hs


def test_metrics_of_a_perfect_and_a_flipped_prediction():
    truth = np.array([5.0, 1.0, -1.0, 0.0, 0.1])      # gene 0 is the perturbed gene itself: ignored
    ok = hs.score_one(truth * 2, truth, gene=0, threshold=0.5)
    assert ok["pearson_delta"] == pytest.approx(1.0) and ok["direction"] == 1.0 and ok["top_overlap"] == 1.0
    assert ok["n_affected"] == 2
    flipped = hs.score_one(-truth, truth, gene=0, threshold=0.5)
    assert flipped["pearson_delta"] == pytest.approx(-1.0) and flipped["direction"] == 0.0
    assert flipped["deg_mse_skill"] < 0


def test_a_zero_prediction_has_no_direction_or_ranking_and_zero_skill():
    truth = np.array([3.0, 1.0, -1.0, 0.0])
    s = hs.score_one(np.zeros(4), truth, gene=0, threshold=0.5)
    assert s["pearson_delta"] == 0.0 and s["deg_mse_skill"] == pytest.approx(0.0)
    assert np.isnan(s["direction"]) and np.isnan(s["top_overlap"])


def test_a_silent_perturbation_only_has_the_correlation():
    s = hs.score_one(np.array([0.0, 0.2, -0.1]), np.array([4.0, 0.1, 0.0]), gene=0, threshold=1.0)
    assert s["n_affected"] == 0 and np.isnan(s["deg_mse_skill"]) and np.isnan(s["direction"]) and not np.isnan(s["pearson_delta"])


def _structured(n=40, m=30, seed=0):
    """Responses of a modular world: gene g sits in one of 4 modules and perturbing it moves gene j by F[module(g), module(j)]."""
    rng = np.random.default_rng(seed)
    module = rng.integers(0, 4, size=n)
    F = 2.0 * np.linalg.qr(rng.normal(size=(4, 4)))[0]       # orthogonal: modules are told apart by their profiles
    effect = F[:, module]                                  # (4, n): the response profile of each module
    nodes = rng.choice(n, size=m, replace=False)
    R = np.stack([effect[module[g]] for g in nodes]) + 0.02 * rng.normal(size=(m, n))
    R[np.arange(m), nodes] = -2.0
    return module, effect, nodes, R


def test_neighbour_and_linear_predictors_find_the_module_that_the_mean_cannot():
    module, effect, nodes, R = _structured()
    test = np.array([g for g in range(40) if g not in set(nodes)][:6])
    truth = np.stack([effect[module[g]] for g in test])
    rng = np.random.default_rng(0)
    corr = {}
    for m in ("train_mean", "random_response", "coexpression", "linear"):
        pred = hs.predict(m, R, nodes, test, rng)
        assert pred.shape == truth.shape
        corr[m] = np.mean([np.corrcoef(p, t)[0, 1] for p, t in zip(pred, truth)])
    assert corr["coexpression"] > 0.5 and corr["linear"] > 0.5
    assert corr["coexpression"] > corr["train_mean"] + 0.3 and corr["linear"] > corr["random_response"] + 0.3


def test_control_predicts_zero_and_unknown_predictors_are_rejected():
    _, _, nodes, R = _structured()
    assert not hs.predict("control", R, nodes, [1, 2], np.random.default_rng(0)).any()
    with pytest.raises(ValueError):
        hs.predict("oracle", R, nodes, [1], np.random.default_rng(0))


def _screen_and_ace(seed=0):
    module, effect, nodes, R = _structured(seed=seed)
    n = R.shape[1]
    conditions = [(int(g),) for g in nodes]
    total = np.stack([effect[module[g]] for g in nodes], axis=1)         # (n, m) noise-free
    screen = SimpleNamespace(conditions=conditions, total_effects=total, effect_threshold=0.3, n_genes=n)
    return screen, SimpleNamespace(nodes=nodes, lfc=R)


def test_cross_validation_scores_every_perturbation_once_and_ranks_structure_above_nothing():
    screen, ace = _screen_and_ace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = hs.evaluate_heldout(screen, ace, n_folds=5, seed=3)
    assert res["n_perturbations"] == 30
    for m in hs.PREDICTORS:
        assert res[m]["all"]["n"] == 30 and res[m]["silent"]["n"] + res[m]["active"]["n"] == 30
    assert res["linear"]["all"]["pearson_delta"] > res["train_mean"]["all"]["pearson_delta"] + 0.3
    assert res["control"]["all"]["deg_mse_skill"] == pytest.approx(0.0, abs=1e-9)
    again = hs.evaluate_heldout(screen, ace, n_folds=5, seed=3)
    assert json.dumps(again, sort_keys=True) == json.dumps(res, sort_keys=True)     # NaN != NaN, so compare the serialised form


def test_too_few_perturbations_for_the_folds_is_an_error():
    screen, ace = _screen_and_ace()
    ace.nodes, ace.lfc = ace.nodes[:6], ace.lfc[:6]
    with pytest.raises(ValueError):
        hs.evaluate_heldout(screen, ace, n_folds=5)
