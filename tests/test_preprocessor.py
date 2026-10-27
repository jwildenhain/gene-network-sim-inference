"""
Unit tests for the GRN Data Preprocessor and algorithm adaptation layer.
"""

import numpy as np
import pytest

from grn_sim.inference.preprocessor import GRNDataPreprocessor, PreparedGRNData
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


@pytest.fixture
def synthetic_experiment():
    net = GeneNetwork.create(n=8, k_av=1.8, net_type=TopologyType.RANDOM, seed=123)
    exp = run_simulation_with_engine(net, engine=SimulationModelingEngine.HILL_NONLINEAR, seed=123)
    return exp


def test_grn_data_preprocessor_structure(synthetic_experiment):
    exp = synthetic_experiment
    prep = GRNDataPreprocessor(shrink_alpha=0.15)
    data = prep.prepare(exp.X, exp.U, reference_steady_state=exp.reference_steady_state)

    assert isinstance(data, PreparedGRNData)
    assert data.Delta_X.shape == exp.X.shape
    assert data.cov_shrunk.shape == (8, 8)
    assert data.corr_shrunk.shape == (8, 8)
    assert data.prec_mat.shape == (8, 8)
    assert data.partial_corr.shape == (8, 8)
    assert data.sensitivity_mat.shape == (8, 8)
    assert np.all(np.diag(data.corr_shrunk) == 1.0)
    assert np.all(np.diag(data.partial_corr) == 0.0)
    assert "delta_x_norm" in data.diagnostics


def test_all_algorithms_with_and_without_prep(synthetic_experiment):
    exp = synthetic_experiment
    models = [
        AracneAlgorithm,
        CausalGRNAlgorithm,
        DMDGRNAlgorithm,
        DeepRIGAlgorithm,
        SFINNAlgorithm,
        NOTEARSAlgorithm,
    ]

    for model_cls in models:
        # Fit with data prep enabled (default)
        m_prep = model_cls(use_data_prep=True)
        res_prep = m_prep.fit(exp.X, exp.U)
        assert res_prep.adjacency_pred.shape == exp.true_matrix.shape
        assert np.all(np.diag(res_prep.adjacency_pred) == 0)
        assert res_prep.diagnostics["use_data_prep"] is True

        # Fit with data prep disabled (backward compatibility)
        m_raw = model_cls(use_data_prep=False)
        res_raw = m_raw.fit(exp.X, exp.U)
        assert res_raw.adjacency_pred.shape == exp.true_matrix.shape
        assert np.all(np.diag(res_raw.adjacency_pred) == 0)
        assert res_raw.diagnostics["use_data_prep"] is False
