"""Equivalence of the fast G2S engine (downdates + certified decisions) with the reference pinv loop,
for both drop rules ("exact" = gen2spec.m dropanother, the default; "loo" = the LOO-score shortcut)."""
import itertools

import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.inference import _g2s_fast as fast
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.simulator.experiment import PerturbationSimulator


def _sim(n, seed, topo=TopologyType.RANDOM):
    net = GeneNetwork.create(n=n, k_av=2, net_type=topo, seed=seed)
    e = PerturbationSimulator(2.0).run(net, seed=seed)
    return e.X, e.U


def _datasets():
    rng = np.random.default_rng(7)
    n = 30
    yield "sim_random", _sim(n, 11)                       # Hill data: column-rank deficient X
    yield "sim_power_law", _sim(n, 12, TopologyType.POWER_LAW)
    yield "randn", (rng.standard_normal((n, n)), rng.standard_normal((n, n)))
    X = rng.standard_normal((n, n))
    X[:, 5] = X[:, 7]
    X[:, 9] = 2 * X[:, 3] - X[:, 1]
    yield "collinear", (X, 0.1 * rng.standard_normal((n, n)))
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    yield "cond1e9", (Q1 @ np.diag(np.logspace(0, -9, n)) @ Q2, rng.standard_normal((n, n)))
    U = rng.standard_normal((n, n))
    U[3] = 0.0                                           # unperturbed row: y falls back to X(i,:)
    yield "zero_u_row", (rng.standard_normal((n, n)), U)


DATASETS = list(_datasets())


def _assert_same(ref, res):
    np.testing.assert_array_equal(ref.A_est != 0, res.A_est != 0)
    np.testing.assert_allclose(res.A_est, ref.A_est, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(ref.adjacency_pred, res.adjacency_pred)


@pytest.mark.parametrize("rule", ["exact", "loo"])
@pytest.mark.parametrize("name,data", DATASETS, ids=[d[0] for d in DATASETS])
def test_fast_matches_reference(name, data, rule):
    X, U = data
    for crit, mm, cap in itertools.product(["aic", "mdl"], [True, False], [None, 3]):
        kw = dict(criterion=crit, matlab_mode=mm, max_interactions=cap, drop_rule=rule)
        ref = GeneralToSpecific(backend="reference", **kw).fit(X, U)
        res = GeneralToSpecific(backend="numpy", **kw).fit(X, U)
        _assert_same(ref, res)


@pytest.mark.parametrize("rule,cap", [("exact", None), ("exact", 4), ("loo", 4)])
def test_backends_identical(rule, cap):
    X, U = _sim(40, 21)
    rng = np.random.default_rng(3)
    X2 = rng.standard_normal((40, 40))
    for Xi in (X, X2):
        kw = dict(max_interactions=cap, drop_rule=rule)
        ref = GeneralToSpecific(backend="reference", **kw).fit(Xi, U)
        a = GeneralToSpecific(backend="numpy", **kw).fit(Xi, U)
        b = GeneralToSpecific(backend="processes", n_jobs=3, **kw).fit(Xi, U)
        _assert_same(ref, a)
        _assert_same(ref, b)
        np.testing.assert_array_equal(a.A_est, b.A_est)
        assert b.diagnostics["backend"] == "processes"


def test_numpy_downdate_without_numba(monkeypatch):
    rng = np.random.default_rng(5)
    X, U = rng.standard_normal((40, 40)), rng.standard_normal((40, 40))
    monkeypatch.setitem(fast._KERNEL, "fn", None)
    for rule in ("exact", "loo"):
        ref = GeneralToSpecific(backend="reference", max_interactions=3, drop_rule=rule).fit(X, U)
        res = GeneralToSpecific(backend="numpy", max_interactions=3, drop_rule=rule).fit(X, U)
        _assert_same(ref, res)
        assert res.diagnostics["downdate_steps"] > 0


def test_wide_and_low_rank_bases_use_reference_semantics():
    rng = np.random.default_rng(9)
    n = 40
    low = rng.standard_normal((n, 28)) @ rng.standard_normal((28, n))     # rank 28 < 40 columns
    U = rng.standard_normal((n, n))
    G = GeneralToSpecific(max_interactions=4)
    for i in range(3):
        y = U[i].copy()
        r_ind, r_a = G._gen2spec_reference(y, low)
        f_ind, f_a = G._gen2spec(y, low)
        np.testing.assert_array_equal(r_ind, f_ind)
        np.testing.assert_allclose(f_a, r_a, atol=1e-8)
    wide = rng.standard_normal((30, 45))                                  # N < M: pinv is min-norm
    y = rng.standard_normal(30)
    r_ind, r_a = G._gen2spec_reference(y, wide)
    f_ind, f_a = G._gen2spec(y, wide)
    np.testing.assert_array_equal(r_ind, f_ind)
    np.testing.assert_allclose(f_a, r_a, atol=1e-8)


def test_pinv_replica_is_bitwise_numpy():
    rng = np.random.default_rng(1)
    for shape in [(20, 20), (30, 12), (12, 30)]:
        A = rng.standard_normal(shape)
        P, _ = fast.pinv_with_svd(A)
        np.testing.assert_array_equal(P, np.linalg.pinv(A))


def test_small_basis_options_and_bad_arguments():
    rng = np.random.default_rng(2)
    X, U = rng.standard_normal((10, 10)), rng.standard_normal((10, 10))
    res = GeneralToSpecific(drop_rule="loo").fit(X, U)
    assert res.diagnostics["backend"] == "reference"       # loo rule: M <= fast_loo_threshold is exact
    ref = GeneralToSpecific(backend="reference").fit(X, U)
    _assert_same(ref, GeneralToSpecific().fit(X, U))
    _assert_same(ref, res)                                 # both rules agree on small bases
    with pytest.raises(ValueError):
        GeneralToSpecific(backend="gpu")
    with pytest.raises(ValueError):
        GeneralToSpecific(drop_rule="fast")


def test_exact_rule_differs_from_loo_only_inside_dependencies():
    """Full-column-rank X: both rules coincide; rank-deficient X: the LOO shortcut can differ."""
    rng = np.random.default_rng(4)
    X, U = rng.standard_normal((32, 32)), rng.standard_normal((32, 32))
    _assert_same(GeneralToSpecific(drop_rule="loo").fit(X, U), GeneralToSpecific().fit(X, U))


def _rank_deficient(seed=0, n=24, rank=14):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, rank)) @ rng.standard_normal((rank, n))
    U = np.diag(rng.uniform(0.5, 2.0, n))
    return X, U


def test_tie_break_budget_only_acts_on_rank_deficient_steps():
    X, U = _rank_deficient()
    exact = GeneralToSpecific(backend="numpy", exact_budget_flops=None).fit(X, U)
    assert exact.diagnostics["tie_breaks"] == 0 and exact.diagnostics["matlab_exact"]
    cheap = GeneralToSpecific(backend="numpy", exact_budget_flops=0).fit(X, U)
    assert cheap.diagnostics["tie_breaks"] > 0 and not cheap.diagnostics["matlab_exact"]
    assert cheap.diagnostics["exact_residual_evals"] == 0
    assert np.all(np.isfinite(cheap.A_est))
    # Full-column-rank data never needs the exact evaluations, so a zero budget changes nothing.
    rng = np.random.default_rng(3)
    Xf, Uf = rng.standard_normal((20, 20)), np.eye(20)
    a = GeneralToSpecific(backend="numpy", exact_budget_flops=None).fit(Xf, Uf)
    b = GeneralToSpecific(backend="numpy", exact_budget_flops=0).fit(Xf, Uf)
    assert b.diagnostics["tie_breaks"] == 0 and np.array_equal(a.A_est, b.A_est)


def test_tie_break_is_identical_across_backends():
    X, U = _rank_deficient(seed=7, n=40, rank=25)
    a = GeneralToSpecific(backend="numpy", exact_budget_flops=0).fit(X, U)
    b = GeneralToSpecific(backend="processes", n_jobs=4, exact_budget_flops=0).fit(X, U)
    assert np.array_equal(a.A_est, b.A_est) and a.diagnostics["tie_breaks"] == b.diagnostics["tie_breaks"]


def test_negative_budget_is_rejected():
    with pytest.raises(ValueError):
        GeneralToSpecific(exact_budget_flops=-1)
