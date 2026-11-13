"""CPA-Lite on Hill regulator-input data: X ~ Decoder(Encoder(X + U)) with a latent size that can hold a sparse network."""
import numpy as np
import pytest

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference import cpa_lite as cpa_module
from grn_sim.inference import registry
from grn_sim.inference.cpa_lite import CPALite
from grn_sim.perturbseq.pipeline import build_network
from grn_sim.simulator.model_engine import run_simulation_with_engine


@pytest.fixture(scope="module")
def hill_data():
    net = build_network("replogle_like", 100, seed=11)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.01, seed=11)
    return net.topology.adjacency != 0, np.asarray(exp.X, float), np.asarray(exp.U, float)


def _aupr(truth, res):
    return compute_aupr(truth, np.abs(res.A_est), ignore_diagonal=True)


def test_regulator_mode_recovers_edges_the_four_dimensional_bottleneck_cannot(hill_data):
    truth, X, U = hill_data
    old = CPALite(epochs=40, k_interactions=3).fit(X, U)
    new = CPALite(epochs=40, k_interactions=3, perturbation_mode="regulator").fit(X, U)
    assert new.diagnostics["perturbation_mode"] == "regulator" and new.diagnostics["latent_dim"] >= 50
    best_old = max(_aupr(truth, old), compute_aupr(truth, np.abs(old.A_est.T), ignore_diagonal=True))   # either orientation
    assert _aupr(truth, new) > 3 * best_old
    assert _aupr(truth, new) > 0.25
    assert (new.adjacency_pred != 0).sum(axis=0).max() <= 3      # K regulators per target (columns: output is [regulator, target])


def test_the_closed_form_fallback_without_torch_agrees(hill_data, monkeypatch):
    truth, X, U = hill_data
    monkeypatch.setattr(cpa_module, "HAS_TORCH", False)
    res = CPALite(k_interactions=3, perturbation_mode="regulator").fit(X, U)
    assert res.diagnostics["backend"] == "reduced_rank_ls" and _aupr(truth, res) > 0.2


def test_additive_mode_is_unchanged(hill_data):
    _, X, U = hill_data
    res = CPALite(epochs=5).fit(X, U)
    assert "perturbation_mode" not in res.diagnostics and res.diagnostics["latent_dim"] == 4


def test_registry_builds_regulator_mode_on_hill_data_only():
    assert registry.build("cpa_lite", registry.BuildContext(k_est=3, use_reg=True, is_sos=False)).perturbation_mode == "regulator"
    assert registry.build("cpa_lite", registry.BuildContext(k_est=3, use_reg=False, is_sos=False)).perturbation_mode == "additive"
    with pytest.raises(ValueError):
        CPALite(perturbation_mode="knockdown")
