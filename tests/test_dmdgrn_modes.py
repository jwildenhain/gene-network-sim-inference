"""DMD-GRN: regulator-input mode on Hill data, and the output orientation on additive data."""
import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import orientation, registry
from grn_sim.inference.dmdgrn import DMDGRNAlgorithm
from grn_sim.simulator.model_engine import run_simulation_with_engine


def test_regulator_mode_recovers_edges_that_the_additive_inversion_misses():
    from grn_sim.perturbseq.pipeline import build_network
    net = build_network("replogle_like", 100, seed=11)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.01, seed=11)
    truth = net.topology.adjacency != 0
    X, U = np.asarray(exp.X, float), np.asarray(exp.U, float)
    old = DMDGRNAlgorithm(k_interactions=3).fit(X, U)
    new = DMDGRNAlgorithm(k_interactions=3, perturbation_mode="regulator").fit(X, U)
    assert new.diagnostics["perturbation_mode"] == "regulator"
    a_old = compute_aupr(truth, np.abs(old.A_est), ignore_diagonal=True)
    a_new = compute_aupr(truth, np.abs(new.A_est), ignore_diagonal=True)
    assert a_new > 3 * a_old and a_new > 0.2, (a_old, a_new)


def test_registry_uses_regulator_mode_on_hill_data_only_and_declares_the_orientation():
    assert registry.build("dmdgrn", registry.BuildContext(k_est=3, use_reg=True, is_sos=False)).perturbation_mode == "regulator"
    assert registry.build("dmdgrn", registry.BuildContext(k_est=3, use_reg=False, is_sos=False)).perturbation_mode == "additive"
    assert registry.get("dmdgrn").regulator_input and "dmdgrn" in orientation._ADDITIVE_TARGET_REGULATOR


def test_additive_output_is_target_regulator_and_scoring_orients_it():
    """On additive (linear) data the raw output is A[target, regulator]: transposed it scores far better."""
    gains = []
    for seed in (1, 2, 3):
        net = GeneNetwork.create(n=40, k_av=1.8, seed=seed)
        exp = run_simulation_with_engine(network=net, engine="linear_jacobian", perturbation_factor=1.0,
                                         measurement_noise=0.01, seed=seed)
        truth = net.topology.adjacency != 0
        A = np.abs(DMDGRNAlgorithm(k_interactions=3).fit(np.asarray(exp.X, float), np.asarray(exp.U, float)).A_est)
        oriented = orientation.orient_prediction("dmdgrn", A, orientation.ADDITIVE_GARDNER, False)
        gains.append(compute_aupr(truth, oriented) / max(compute_aupr(truth, A), 1e-9))
    assert min(gains) > 1.5, gains


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        DMDGRNAlgorithm(perturbation_mode="knockdown")
