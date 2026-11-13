"""GENIE3 / GRNBoost on X + U (Hill data) are [regulator, target] as returned; on additive data they stay [target, regulator]."""
import numpy as np
import pytest

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import orientation, registry
from grn_sim.perturbseq.pipeline import build_network
from grn_sim.simulator.model_engine import run_simulation_with_engine


@pytest.fixture(scope="module")
def hill():
    net = build_network("replogle_like", 100, seed=11)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.02, seed=7)
    return net.topology.adjacency != 0, np.asarray(exp.X, float), np.asarray(exp.U, float)


def test_the_declarations_follow_the_measured_orientation():
    assert "genie3" not in orientation._TARGET_ROW_ALGORITHMS and "grnboost" not in orientation._TARGET_ROW_ALGORITHMS
    assert "causalgrn" in orientation._TARGET_ROW_ALGORITHMS                      # unchanged: it does return [target, regulator]
    assert "genie3" in orientation._ADDITIVE_TARGET_REGULATOR and "grnboost" in orientation._ADDITIVE_TARGET_REGULATOR


@pytest.mark.parametrize("key", ["genie3", "grnboost"])
def test_hill_output_scores_better_as_returned_than_transposed(hill, key):
    truth, X, U = hill
    A = np.abs(registry.build(key, registry.BuildContext(k_est=3, use_reg=True, is_sos=False)).fit(X, U).A_est)
    as_is, transposed = compute_aupr(truth, A, ignore_diagonal=True), compute_aupr(truth, A.T, ignore_diagonal=True)
    assert as_is > 1.5 * transposed, (as_is, transposed)
    oriented = orientation.orient_prediction(key, A, orientation.HILL, False)
    assert np.array_equal(oriented, A)                                            # the scoring path leaves it as it is
