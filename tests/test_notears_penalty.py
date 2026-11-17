"""NOTEARS' L1 penalty is relative to the data: no empty graph from an overflow, no dense graph below the sparsity transition."""
import warnings

import numpy as np
import pytest

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.inference.notears import NOTEARSAlgorithm


def _screen(generator, n, seed=2026):
    run = vb.simulate_screen(vb.Setting(generator=generator, n_genes=n), seed)
    ace = sb.screen_inputs(run, "cells", seed, normalisation=vb.NORMALISATION)
    X, U = ace.to_xu()
    return run, X, U


def _fit(algorithm, X, U):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return algorithm.fit(X, U)


@pytest.mark.parametrize("generator", ["linear_sem", "nb_copula", "nonlinear_scm"])
def test_statistical_screens_give_a_sparse_graph_that_beats_chance(generator):
    run, X, U = _screen(generator, 100)
    res = _fit(NOTEARSAlgorithm(), X, U)
    assert np.all(np.isfinite(res.A_est)) and not res.diagnostics["diverged"]
    nonzero = int((np.abs(res.A_est) > 1e-6).sum())
    assert 0 < nonzero <= 5 * 100
    scores = sb.evaluate_scores(run.screen, np.abs(res.A_est))
    assert scores["aupr_direct"] > 3 * scores["base_direct"], scores


def test_an_overflowing_fit_is_refitted_with_a_larger_penalty_instead_of_returning_nothing():
    """nonlinear SCM at 200 genes: the first (smallest) penalty overflows Tr(exp(W o W)); the next one converges."""
    run, X, U = _screen("nonlinear_scm", 200)
    res = _fit(NOTEARSAlgorithm(), X, U)
    fits = res.diagnostics["fits"]
    assert len(fits) >= 2 and not fits[0]["finite"] and fits[-1]["finite"]
    assert res.diagnostics["lambda1"] == pytest.approx(fits[-1]["lambda1"])
    scores = sb.evaluate_scores(run.screen, np.abs(res.A_est))
    assert scores["aupr_direct"] > 3 * scores["base_direct"], scores


def test_a_fixed_penalty_is_used_as_given_with_one_fit():
    _, X, U = _screen("linear_sem", 50)
    res = _fit(NOTEARSAlgorithm(lambda1=0.02), X, U)
    assert [f["lambda1"] for f in res.diagnostics["fits"]] == [0.02]


def test_the_penalty_scales_with_the_data():
    """The same screen in other units gives the same graph: the penalty follows the data's own lambda_max."""
    _, X, U = _screen("linear_sem", 50)
    base = _fit(NOTEARSAlgorithm(), X, U)
    scaled = _fit(NOTEARSAlgorithm(), 3.0 * X, 3.0 * U)
    assert scaled.diagnostics["lambda_max"] == pytest.approx(9.0 * base.diagnostics["lambda_max"], rel=1e-6)
    assert scaled.diagnostics["lambda1"] == pytest.approx(9.0 * base.diagnostics["lambda1"], rel=1e-6)
