"""
Unit tests for the 6 newly added GRN reverse engineering algorithms:
- ARACNE (Algorithm for the Reconstruction of Accurate Cellular Networks)
- CausalGRN (Partial Correlation & Interventional Do-Calculus)
- DMD-GRN (Dynamic Mode Decomposition & Koopman Operator)
- DeepRIG (Deep Graph Autoencoder with GCN & Inner-Product Decoder)
- SFINN (Shared Factor Interaction Neural Network with GCN + FNN)
- NOTEARS (Continuous DAG Learning via Non-Convex Augmented Lagrangian)
"""

import numpy as np
import pytest

from grn_sim.inference import (
    AracneAlgorithm,
    CausalGRNAlgorithm,
    DMDGRNAlgorithm,
    DeepRIGAlgorithm,
    SFINNAlgorithm,
    NOTEARSAlgorithm,
)
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.simulator.model_engine import SimulationModelingEngine, run_simulation_with_engine
from grn_sim.evaluation.metrics import evaluate_inference


@pytest.fixture
def synthetic_data():
    """Generates an 8-gene synthetic network and perturbation response."""
    net = GeneNetwork.create(n=8, k_av=1.8, net_type=TopologyType.RANDOM, seed=42)
    exp = run_simulation_with_engine(net, engine=SimulationModelingEngine.HILL_NONLINEAR, seed=42)
    return exp


def test_arachne_algorithm(synthetic_data):
    exp = synthetic_data
    model = AracneAlgorithm(eps=0.10)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "arachne"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert 0.0 <= eval_res.coverage_pct <= 100.0
    assert 0.0 <= eval_res.false_positives_pct <= 100.0
    assert "total_reconstructed_edges" in res.diagnostics


def test_causalgrn_algorithm(synthetic_data):
    exp = synthetic_data
    model = CausalGRNAlgorithm(pcorr_threshold=0.08, alpha=0.05)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "causalgrn"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert 0.0 <= eval_res.coverage_pct <= 100.0
    assert "total_reconstructed_edges" in res.diagnostics


def test_dmdgrn_algorithm(synthetic_data):
    exp = synthetic_data
    model = DMDGRNAlgorithm(rank_r=4, threshold=0.05)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "dmdgrn"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    assert "total_reconstructed_edges" in res.diagnostics


def test_deeprig_algorithm(synthetic_data):
    exp = synthetic_data
    model = DeepRIGAlgorithm(hidden_dim=16, latent_dim=8, seed=42)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "deeprig"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    assert "total_reconstructed_edges" in res.diagnostics


def test_sfinn_algorithm(synthetic_data):
    exp = synthetic_data
    model = SFINNAlgorithm(embed_dim=8, hidden_dim=16, seed=42)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "sfinn"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    assert "total_reconstructed_edges" in res.diagnostics


def test_notears_algorithm(synthetic_data):
    exp = synthetic_data
    model = NOTEARSAlgorithm(lambda1=0.02, max_iter=25, w_threshold=0.05)
    res = model.fit(exp.X, exp.U)

    assert res.algorithm_name == "notears"
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)
    assert set(np.unique(res.adjacency_pred)).issubset({-1, 0, 1})
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert 0.0 <= eval_res.coverage_pct <= 100.0
    assert "total_reconstructed_edges" in res.diagnostics
