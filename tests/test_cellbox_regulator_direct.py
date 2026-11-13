"""CellBox on Hill regulator-input data: X ~ phi(W (X + U)) / alpha instead of the additive-u model."""
import numpy as np
import pytest

pytest.importorskip("torch")

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import registry
from grn_sim.inference.cellbox import CellBoxNeuralODE
from grn_sim.perturbseq.pipeline import build_network
from grn_sim.simulator.model_engine import run_simulation_with_engine


@pytest.fixture(scope="module")
def hill_data():
    net = build_network("replogle_like", 100, seed=11)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.01, seed=11)
    return net.topology.adjacency != 0, np.asarray(exp.X, float), np.asarray(exp.U, float)


def test_direct_fit_recovers_edges_where_the_additive_route_was_at_chance(hill_data):
    truth, X, U = hill_data
    old = CellBoxNeuralODE(perturbation_mode="regulator", regulator_direct=False, k_interactions=3).fit(X, U)
    new = CellBoxNeuralODE(perturbation_mode="regulator", k_interactions=3).fit(X, U)
    assert new.diagnostics["regulator_direct"] is True and new.diagnostics["perturbation_mode"] == "regulator"
    a_old = compute_aupr(truth, np.abs(old.A_est), ignore_diagonal=True)
    a_new = compute_aupr(truth, np.abs(new.A_est), ignore_diagonal=True)
    assert a_new > 3 * a_old and a_new > 0.25, (a_old, a_new)
    assert (new.adjacency_pred != 0).sum(axis=0).max() <= 3      # K regulators per target: columns after the transposition


def test_additive_mode_is_unchanged(hill_data):
    _, X, U = hill_data
    res = CellBoxNeuralODE(perturbation_mode="additive").fit(X, U)
    assert "regulator_direct" not in res.diagnostics


def test_registry_builds_the_regulator_mode_on_hill_data():
    algo = registry.build("cellbox", registry.BuildContext(k_est=3, use_reg=True, is_sos=False))
    assert algo.perturbation_mode == "regulator" and algo.regulator_direct is True
