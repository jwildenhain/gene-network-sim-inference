"""CellBox runs on the CPU or, when one is usable, on the GPU, with the same result."""
import numpy as np
import pytest

from grn_sim.inference import cellbox as cb
from grn_sim.inference.cellbox import CellBoxNeuralODE, resolve_device

torch = pytest.importorskip("torch")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="no usable CUDA GPU")


def _data(n=24, seed=3):
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < 0.1) * rng.normal(size=(n, n))
    U = np.eye(n)
    X = np.linalg.solve(3 * np.eye(n) - A, U) + rng.normal(scale=0.01, size=(n, n))
    return X, U


def _gpu_visible(monkeypatch, usable):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: usable)


def test_cpu_is_chosen_when_asked_and_reported(monkeypatch):
    monkeypatch.delenv("GRN_SIM_DEVICE", raising=False)
    X, U = _data()
    res = CellBoxNeuralODE(epochs=10, device="cpu").fit(X, U)
    assert res.diagnostics["device"] == "cpu" and res.diagnostics["backend"] == "pytorch"


def test_auto_uses_the_gpu_only_for_big_networks(monkeypatch):
    monkeypatch.delenv("GRN_SIM_DEVICE", raising=False)
    _gpu_visible(monkeypatch, True)
    assert resolve_device("auto", cb.GPU_MIN_GENES - 1) == "cpu"
    assert resolve_device("auto", cb.GPU_MIN_GENES) == "cuda"
    assert resolve_device(None, 2000) == "cuda"
    _gpu_visible(monkeypatch, False)
    assert resolve_device("auto", 2000) == "cpu"


def test_environment_variable_sets_the_default(monkeypatch):
    _gpu_visible(monkeypatch, True)
    monkeypatch.setenv("GRN_SIM_DEVICE", "cpu")
    assert resolve_device(None, 5000) == "cpu"
    assert resolve_device("cuda", 10) == "cuda"                       # an explicit argument beats the variable
    monkeypatch.setenv("GRN_SIM_DEVICE", "gpu")                       # "gpu" is accepted for "cuda"
    assert resolve_device(None, 10) == "cuda"


def test_unknown_device_is_refused():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("tpu", 100)
    with pytest.raises(ValueError):
        CellBoxNeuralODE(epochs=5, device="tpu").fit(*_data())


def test_an_explicit_gpu_request_without_a_gpu_fails_clearly(monkeypatch):
    _gpu_visible(monkeypatch, False)
    with pytest.raises(RuntimeError, match="no usable GPU"):
        CellBoxNeuralODE(epochs=5, device="cuda").fit(*_data())


def test_auto_falls_back_to_the_cpu_when_the_gpu_run_fails(monkeypatch):
    monkeypatch.delenv("GRN_SIM_DEVICE", raising=False)
    monkeypatch.setattr(cb, "resolve_device", lambda requested, n: "cpu" if requested == "cpu" else "cuda")   # no real GPU needed
    real = CellBoxNeuralODE._fit_torch

    def flaky(self, X, U, l1, l2, epochs, device_name="cpu", target=None):
        if device_name == "cuda":
            raise RuntimeError("CUDA out of memory (simulated)")
        return real(self, X, U, l1, l2, epochs, device_name, target=target)

    monkeypatch.setattr(CellBoxNeuralODE, "_fit_torch", flaky)
    res = CellBoxNeuralODE(epochs=10).fit(*_data())
    assert res.diagnostics["device"] == "cpu" and "out of memory" in res.diagnostics["gpu_fallback"]
    with pytest.raises(RuntimeError, match="out of memory"):           # but an explicit request is not hidden
        CellBoxNeuralODE(epochs=10, device="cuda").fit(*_data())


def test_the_cpu_result_is_unchanged_by_the_device_code():
    """A seeded CPU fit is reproducible and gives a non-trivial network."""
    X, U = _data()
    torch.manual_seed(0)
    a = CellBoxNeuralODE(epochs=30, device="cpu").fit(X, U)
    torch.manual_seed(0)
    b = CellBoxNeuralODE(epochs=30, device="cpu").fit(X, U)
    assert np.array_equal(a.A_est, b.A_est) and np.any(a.adjacency_pred != 0)


@needs_gpu
def test_gpu_and_cpu_agree():
    X, U = _data(40)
    torch.manual_seed(1)
    cpu = CellBoxNeuralODE(epochs=60, device="cpu").fit(X, U)
    torch.manual_seed(1)
    gpu = CellBoxNeuralODE(epochs=60, device="cuda").fit(X, U)
    assert gpu.diagnostics["device"] == "cuda"
    assert np.corrcoef(cpu.A_est.ravel(), gpu.A_est.ravel())[0, 1] > 0.999
    assert np.mean(np.sign(cpu.A_est) == np.sign(gpu.A_est)) > 0.99


@needs_gpu
def test_a_worker_pool_still_works_after_the_gpu_was_used():
    """CUDA is started in this process; the fork-based pools of the other algorithms must not hang afterwards."""
    from grn_sim.inference.s2g import SimpleToGeneral
    X, U = _data(30)
    CellBoxNeuralODE(epochs=5, device="cuda").fit(X, U)
    res = SimpleToGeneral(n_jobs=2).fit(X, U)
    assert res.adjacency_pred.shape == (30, 30)
