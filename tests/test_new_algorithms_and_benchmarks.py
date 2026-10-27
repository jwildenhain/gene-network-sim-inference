import numpy as np
import pytest

from grn_sim.inference.cellbox import CellBoxNeuralODE
from grn_sim.inference.genie3 import GENIE3Algorithm
from grn_sim.inference.network_deconv import NetworkDeconvolution
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.cpa_lite import CPALite
from grn_sim.simulator.model_engine import (
    SimulationModelingEngine,
    run_simulation_with_engine,
)
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.benchmarks.benchmark_datasets import (
    list_benchmark_datasets,
    load_benchmark_dataset,
)
from grn_sim.evaluation.metrics import evaluate_inference


@pytest.fixture
def synthetic_experiment():
    """Creates a small 8-gene synthetic perturbation experiment."""
    net = GeneNetwork.create(n=8, k_av=1.8, net_type=TopologyType.RANDOM, seed=123)
    exp = run_simulation_with_engine(net, engine=SimulationModelingEngine.HILL_NONLINEAR, seed=123)
    return exp


def test_cellbox_neural_ode_fit(synthetic_experiment):
    exp = synthetic_experiment
    model = CellBoxNeuralODE(epochs=30, l1_reg=0.01)
    res = model.fit(exp.X, exp.U)
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert eval_res.coverage_pct >= 0.0
    assert "final_loss" in res.diagnostics
    assert res.diagnostics["backend"] in ("pytorch", "scipy")


def test_genie3_algorithm_fit(synthetic_experiment):
    exp = synthetic_experiment
    model = GENIE3Algorithm(n_estimators=25, seed=42)
    res = model.fit(exp.X, exp.U)
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert eval_res.inferred_connections_pct > 0.0


def test_network_deconvolution_fit(synthetic_experiment):
    exp = synthetic_experiment
    model = NetworkDeconvolution(beta=0.85)
    res = model.fit(exp.X, exp.U)
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert np.all(np.diag(res.adjacency_pred) == 0)


def test_ridge_cv_fit(synthetic_experiment):
    exp = synthetic_experiment
    model = KaggleMultiOutputRidge(k_interactions=2)
    res = model.fit(exp.X, exp.U)
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    eval_res = evaluate_inference(exp.true_matrix, res.adjacency_pred)
    assert eval_res.inferred_connections_pct > 0.0


def test_cpa_lite_fit(synthetic_experiment):
    """Whichever backend this environment has: PyTorch (reports a training loss) or the SVD fallback."""
    exp = synthetic_experiment
    model = CPALite(latent_dim=4, epochs=25)
    res = model.fit(exp.X, exp.U, seed=42)
    assert res.adjacency_pred.shape == exp.true_matrix.shape
    assert res.A_est is not None
    assert res.diagnostics["backend"] in ("pytorch", "svd_pca")
    if res.diagnostics["backend"] == "pytorch":
        assert "final_loss" in res.diagnostics


def test_cpa_lite_works_without_pytorch(synthetic_experiment, monkeypatch):
    """PyTorch is an optional extra, so the SVD fallback must give a complete result on its own."""
    from grn_sim.inference import cpa_lite
    monkeypatch.setattr(cpa_lite, "HAS_TORCH", False)
    exp = synthetic_experiment
    res = CPALite(latent_dim=4, epochs=25).fit(exp.X, exp.U, seed=42)
    assert res.diagnostics == {"latent_dim": 4, "backend": "svd_pca"}
    assert res.A_est.shape == exp.true_matrix.shape and np.all(np.isfinite(res.A_est))


@pytest.mark.parametrize("engine", [
    SimulationModelingEngine.HILL_NONLINEAR,
    SimulationModelingEngine.CELLBOX_NEURAL_ODE,
    SimulationModelingEngine.LINEAR_JACOBIAN,
    SimulationModelingEngine.PERTURB_SEQ_COUNTS,
])
def test_all_modeling_engines(engine):
    net = GeneNetwork.create(n=8, k_av=1.8, net_type=TopologyType.POWER_LAW, seed=42)
    exp = run_simulation_with_engine(net, engine=engine, seed=42)
    assert exp.X.shape == (8, 8)
    assert exp.U.shape == (8, 8)
    assert not np.any(np.isnan(exp.X))
    assert not np.any(np.isinf(exp.X))
    assert exp.reference_steady_state.shape == (8,)


def test_benchmark_catalog_and_loaders():
    catalog = list_benchmark_datasets()
    assert len(catalog) >= 4
    ids = {d["id"] for d in catalog}
    assert {"ecoli_sos", "dream4_10gene", "norman_2019_tf16", "replogle_2022_p53"}.issubset(ids)

    for ds_id in ids:
        data = load_benchmark_dataset(ds_id)
        assert "X" in data
        assert "U" in data
        assert "true_matrix" in data
        assert "gene_names" in data
        N = len(data["gene_names"])
        assert data["X"].shape == (N, N)
        assert data["U"].shape == (N, N)
        assert data["true_matrix"].shape == (N, N)
        if data.get("has_ground_truth", True) and ds_id != "norman_2019_real_105gene":
            assert np.sum(data["true_matrix"] != 0) > 0


def test_benchmark_studio_pipeline_all_12_algorithms():
    from grn_sim.web.server import _build_benchmark_studio
    studio = _build_benchmark_studio("dream4_10gene")
    assert studio["status"] == "success"
    assert studio["dataset_id"] == "dream4_10gene"
    assert len(studio["gene_names"]) == 10

    # Ensure all 12 algorithms evaluated
    tbl = studio["table_4_1"]
    expected_algs = {
        "vote", "cellbox", "genie3", "s2g", "g2s", "netdeconv",
        "ridge_cv", "cpa_lite", "atlasshift", "nir", "fast", "by_chance"
    }
    for alg in expected_algs:
        assert alg in tbl, f"Algorithm {alg} missing from benchmark studio table"
        assert "coverage_pct" in tbl[alg]
        assert "inferred_connections_pct" in tbl[alg]
        assert "total_errors" in tbl[alg]
        assert alg in studio["svgs_dot"]
        assert studio["svgs_dot"][alg].startswith("<svg") or "<svg" in studio["svgs_dot"][alg]


@pytest.mark.parametrize("module_name, class_name, kwargs, fallback_backend", [
    ("cellbox", "CellBoxNeuralODE", {"epochs": 5}, "scipy"),
    ("cpa_lite", "CPALite", {"latent_dim": 4, "epochs": 5}, "svd_pca"),
])
def test_an_installed_but_unimportable_pytorch_falls_back_like_a_missing_one(
    module_name, class_name, kwargs, fallback_backend, synthetic_experiment, monkeypatch
):
    """PyTorch is imported when a model is fitted; a broken install must not take the algorithm down."""
    import importlib
    module = importlib.import_module(f"grn_sim.inference.{module_name}")
    cls = getattr(module, class_name)

    def broken(*args, **kwargs):
        raise ImportError("torch is installed but cannot be imported")

    monkeypatch.setattr(module, "HAS_TORCH", True)
    monkeypatch.setattr(cls, "_fit_torch", broken)
    res = cls(**kwargs).fit(synthetic_experiment.X, synthetic_experiment.U, seed=1)
    assert res.diagnostics["backend"] == fallback_backend


def test_importing_the_package_does_not_load_the_heavy_optional_libraries():
    """`import grn_sim` loads neither PyTorch nor anndata (seconds of import time); they load when first used."""
    import subprocess
    import sys
    code = "import sys, grn_sim; print(sorted(m for m in ('torch', 'anndata') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
