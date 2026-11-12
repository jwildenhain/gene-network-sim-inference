"""inspre on Hill-simulator data (regulator-input perturbation): the knockdown normalisation returned an empty graph."""
import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import registry
from grn_sim.inference.inspre import InspreAlgorithm
from grn_sim.simulator.model_engine import run_simulation_with_engine


def _hill_data(n=60, seed=3):
    net = GeneNetwork.create(n=n, k_av=1.8, seed=seed)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.01, seed=seed)
    return (net.topology.adjacency != 0), np.asarray(exp.X, float), np.asarray(exp.U, float)


def test_regulator_mode_recovers_edges_where_the_knockdown_normalisation_finds_none():
    truth, X, U = _hill_data()
    k = 3
    old = InspreAlgorithm(k_interactions=k).fit(X, U)
    new = InspreAlgorithm(k_interactions=k, perturbation_mode="regulator").fit(X, U)
    assert new.diagnostics["perturbation_mode"] == "regulator"
    aupr_old = compute_aupr(truth, old.A_est, ignore_diagonal=True)
    aupr_new = compute_aupr(truth, new.A_est, ignore_diagonal=True)
    assert (new.adjacency_pred != 0).sum() > 0.3 * truth.sum()
    assert aupr_new > 2 * aupr_old and aupr_new > 0.15, (aupr_old, aupr_new)


def test_registry_builds_the_regulator_variant_for_hill_data_and_the_additive_one_otherwise():
    assert registry.build("inspre", registry.BuildContext(k_est=3, use_reg=True, is_sos=False)).perturbation_mode == "regulator"
    assert registry.build("inspre", registry.BuildContext(k_est=3, use_reg=False, is_sos=False)).perturbation_mode == "additive"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        InspreAlgorithm(perturbation_mode="knockdown")
