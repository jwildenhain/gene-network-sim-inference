"""Every registered algorithm on awkward inputs: tiny networks, a constant gene, a control column, duplicated conditions.

Each must return finite n x n matrices on both data models; the algorithms that solve a square problem must refuse a partial
screen or replicates with a readable message (the studio and ``registry.build_partial`` use ``PartialAdapter`` for those).
"""
import warnings

import numpy as np
import pytest

from grn_sim.inference import registry
from grn_sim.inference.partial import PartialAdapter

SQUARE_ONLY = [s.key for s in registry.specs() if s.partial == "adapter" and s.key in ("s2g", "g2s", "vote", "score_vote", "netdeconv")]


def _screen(n, m=None, control=False, constant=None, duplicate=False, partial=None, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.25)
    np.fill_diagonal(A, 0.0)
    m = m or n
    U = np.zeros((n, m))
    for c in range(m):
        U[c % (partial or n), c] = 1.0
    if control:
        U[:, -1] = 0.0
    X = np.linalg.solve(np.eye(n) + 0.5 * A.T, U) + 0.01 * rng.normal(size=(n, m)) + 1.0
    if constant is not None:
        X[constant] = 1.0
    if duplicate:
        X[:, 1], U[:, 1] = X[:, 0], U[:, 0]
    return X, U


CASES = {
    "three_genes": lambda: _screen(3),
    "twelve_genes": lambda: _screen(12),
    "control_column": lambda: _screen(12, control=True),
    "constant_gene": lambda: _screen(12, constant=3),
    "duplicate_condition": lambda: _screen(12, duplicate=True),
}


@pytest.mark.parametrize("key", registry.keys())
@pytest.mark.parametrize("case", list(CASES))
def test_finite_square_outputs(key, case):
    X, U = CASES[case]()
    n = X.shape[0]
    for use_reg in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = registry.build(key, registry.BuildContext(k_est=2, use_reg=use_reg)).fit(X.copy(), U.copy())
        assert np.asarray(res.A_est).shape == (n, n) and np.asarray(res.adjacency_pred).shape == (n, n)
        assert np.all(np.isfinite(np.real(np.asarray(res.A_est)))), f"{key} returned a non-finite estimate on {case} (reg={use_reg})"


@pytest.mark.parametrize("key", SQUARE_ONLY)
@pytest.mark.parametrize("shape", [dict(n=12, m=5, partial=5), dict(n=10, m=30), dict(n=20, m=8, partial=8)])
def test_square_only_algorithms_refuse_a_partial_screen_readably(key, shape):
    X, U = _screen(**shape)
    with pytest.raises(ValueError, match="needs a square screen"):
        registry.build(key, registry.BuildContext(k_est=2)).fit(X, U)


@pytest.mark.parametrize("key", SQUARE_ONLY)
def test_the_partial_adapter_runs_what_the_algorithm_refuses(key):
    X, U = _screen(n=12, m=6, partial=6)
    res = registry.build_partial(key, registry.BuildContext(k_est=2)).fit(X, U)
    assert isinstance(registry.build_partial(key), PartialAdapter)
    assert np.asarray(res.A_est).shape == (12, 12) and np.all(np.isfinite(np.asarray(res.A_est)))


@pytest.mark.parametrize("cls_name", ["SimpleToGeneral", "GeneralToSpecific"])
def test_the_regression_form_still_takes_extra_experiments(cls_name):
    """Tab 4 adds double perturbations as extra (X, U) columns and fits the non-MATLAB form on them: more columns than genes."""
    from grn_sim.inference.g2s import GeneralToSpecific
    from grn_sim.inference.s2g import SimpleToGeneral

    X, U = _screen(n=12, m=30)
    algo = {"SimpleToGeneral": SimpleToGeneral, "GeneralToSpecific": GeneralToSpecific}[cls_name](matlab_mode=False)
    res = algo.fit(X, U)
    assert np.asarray(res.A_est).shape == (12, 12) and np.all(np.isfinite(np.asarray(res.A_est)))
