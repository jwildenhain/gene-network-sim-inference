"""With K known every algorithm keeps K regulators per target; the magnitude thresholds only apply without K."""
import numpy as np
import pytest

from grn_sim.inference import registry
from grn_sim.inference.cellbox import CellBoxNeuralODE
from grn_sim.inference.cpa_lite import CPALite
from grn_sim.inference.genie3 import GENIE3Algorithm
from grn_sim.inference.notears import NOTEARSAlgorithm
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge

N, M, K = 30, 30, 3


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(N, M)) * 1e-3, rng.normal(size=(N, M)) * 1e-3   # tiny effects: far below any 0.05 threshold


@pytest.mark.parametrize("make", [
    lambda: GENIE3Algorithm(n_estimators=10, k_interactions=K, seed=1),
    lambda: CPALite(epochs=5, k_interactions=K),
    lambda: KaggleMultiOutputRidge(k_interactions=K),
    lambda: CellBoxNeuralODE(epochs=5, k_interactions=K),
], ids=["genie3", "cpa_lite", "ridge_cv", "cellbox"])
def test_k_regulators_per_target_whatever_the_magnitudes(make, data):
    X, U = data
    res = make().fit(X, U)
    per_target = (res.adjacency_pred != 0).sum(axis=1)
    assert per_target.max() <= K and per_target.sum() >= K * N * 0.9, per_target   # K each (a zero weight may drop out)


def test_without_k_the_threshold_still_applies(data):
    X, U = data
    res = GENIE3Algorithm(n_estimators=10, seed=1).fit(X, U)
    assert (res.adjacency_pred != 0).sum() < N * N


def test_registry_gives_these_algorithms_k_on_hill_data_and_none_otherwise():
    for key in ("genie3", "cpa_lite", "cellbox"):
        on = registry.build(key, registry.BuildContext(k_est=4, use_reg=True, is_sos=False))
        off = registry.build(key, registry.BuildContext(k_est=4, use_reg=False, is_sos=False))
        assert on.k_interactions == 4 and off.k_interactions is None, key


def test_notears_default_penalty_does_not_empty_the_graph():
    assert NOTEARSAlgorithm().lambda1 <= 0.001   # 0.02 shrank whole networks to 17-45 edges at 200-300 genes


def test_s2g_is_capped_at_k_on_hill_data_so_it_is_comparable_with_the_others():
    on = registry.build("s2g", registry.BuildContext(k_est=4, use_reg=True, is_sos=False))
    off = registry.build("s2g", registry.BuildContext(k_est=4, use_reg=False, is_sos=False))
    assert on.max_interactions == 4 and off.max_interactions is None


def test_s2g_never_predicts_more_than_its_cap_per_gene():
    from grn_sim.inference.s2g import SimpleToGeneral
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 25))
    U = X @ rng.normal(size=(25, 25)) * 0.1 + rng.normal(size=(25, 25)) * 0.01   # many real dependencies
    res = SimpleToGeneral(max_interactions=2).fit(X, U)
    assert (res.adjacency_pred != 0).sum(axis=0).max() <= 2   # column i holds the regulators of gene i
