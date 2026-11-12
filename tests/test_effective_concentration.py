"""Co-expression methods get X + U on Hill regulator-input data (the perturbation is invisible in X there)."""
import numpy as np
import pytest

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import registry
from grn_sim.inference.base import EffectiveConcentration
from grn_sim.simulator.model_engine import run_simulation_with_engine

WRAPPED = ("genie3", "netdeconv", "arachne", "causalgrn", "pearson", "grnboost")
NOT_WRAPPED = ("gene_features", "mean_difference", "atlasshift", "sfinn", "deeprig", "dmdgrn", "notears", "inspre")


def test_registry_wraps_only_the_algorithms_that_gain_and_only_on_hill_data():
    on = registry.BuildContext(k_est=3, use_reg=True, is_sos=False)
    off = registry.BuildContext(k_est=3, use_reg=False, is_sos=False)
    for key in WRAPPED:
        assert isinstance(registry.build(key, on), EffectiveConcentration), key
        assert not isinstance(registry.build(key, off), EffectiveConcentration), key
    for key in NOT_WRAPPED:
        assert not isinstance(registry.build(key, on), EffectiveConcentration), key


def test_the_wrapper_passes_x_plus_u_and_delegates_attributes():
    seen = {}

    class Probe:
        name = "probe"
        k_interactions = 7

        def fit(self, X, U, **kw):
            seen["X"] = X
            from grn_sim.inference.base import InferenceResult
            return InferenceResult("probe", np.zeros((2, 2)), np.zeros((2, 2), dtype=int), 0.0, {"a": 1})

    wrapped = EffectiveConcentration(Probe())
    res = wrapped.fit(np.ones((2, 3)), 2 * np.ones((2, 3)))
    assert np.array_equal(seen["X"], 3 * np.ones((2, 3)))
    assert wrapped.k_interactions == 7 and wrapped.name == "probe"
    assert res.diagnostics == {"a": 1, "input": "effective concentration X + U"}


@pytest.mark.parametrize("key", ["pearson", "arachne"])
def test_effective_concentration_improves_ranking_on_simulated_hill_data(key):
    from grn_sim.perturbseq.pipeline import build_network
    net = build_network("replogle_like", 100, seed=11)   # the calibrated networks the gain was measured on
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.01, seed=11)
    truth = net.topology.adjacency != 0
    X, U = np.asarray(exp.X, float), np.asarray(exp.U, float)
    ctx = registry.BuildContext(k_est=3, use_reg=True, is_sos=False)
    wrapped = registry.build(key, ctx)
    plain = wrapped.inner
    a_eff = max(compute_aupr(truth, np.abs(wrapped.fit(X, U).A_est)), compute_aupr(truth, np.abs(wrapped.fit(X, U).A_est.T)))
    a_raw = max(compute_aupr(truth, np.abs(plain.fit(X, U).A_est)), compute_aupr(truth, np.abs(plain.fit(X, U).A_est.T)))
    assert a_eff > 1.3 * a_raw, (key, a_raw, a_eff)
