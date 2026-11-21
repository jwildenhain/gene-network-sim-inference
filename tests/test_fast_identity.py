"""FAST's first estimate is I + A, so its top-K ranking wastes a slot per row on the diagonal (see ``FastSVD``)."""
import numpy as np

from grn_sim.inference import registry
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.llc import LLCMean


def _additive_data(n: int = 24, k: int = 2, seed: int = 0):
    """A X = -U with a sparse zero-diagonal A (k regulators per row) and a full-rank X."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        cols = rng.choice([j for j in range(n) if j != i], size=k, replace=False)
        A[i, cols] = rng.choice([-1.0, 1.0], size=k) * rng.uniform(0.5, 1.0, size=k)
    X = rng.normal(size=(n, n))
    return A, X, -A @ X


def _recovered(A_est: np.ndarray, A: np.ndarray) -> int:
    return int(np.sum((A_est != 0) & (A != 0)))


def test_the_matlab_ranking_is_the_class_default_and_the_studio_removes_the_identity():
    assert FastSVD().rank_without_identity is False
    fast = registry.build("fast", registry.BuildContext(k_est=3, use_reg=True))
    assert fast.rank_without_identity is True
    assert registry.build("fast", registry.BuildContext(k_est=3, use_reg=False)).rank_without_identity is True


def test_without_the_identity_the_diagonal_is_never_selected_and_more_edges_are_found():
    A, X, U = _additive_data()
    plain = FastSVD(k_interactions=2).fit(X, U).A_est
    fixed = FastSVD(k_interactions=2, rank_without_identity=True).fit(X, U).A_est
    assert np.count_nonzero(np.diag(plain)) >= 10                # one slot per row goes to a self-edge
    assert np.count_nonzero(np.diag(fixed)) == 0
    assert _recovered(fixed, A) > _recovered(plain, A) + 10      # 48 true edges: nearly all with the option, about half without
    assert _recovered(fixed, A) >= 44


def test_the_option_leaves_the_sos_diagonal_correction_alone():
    A, X, U = _additive_data(n=9, k=2, seed=3)
    a = FastSVD(k_interactions=2, add_diagonal_identity=True).fit(X, U).A_est
    b = FastSVD(k_interactions=2, add_diagonal_identity=True, rank_without_identity=True).fit(X, U).A_est
    assert np.array_equal(a, b)


def test_the_option_also_applies_to_the_adaptive_model_order():
    A, X, U = _additive_data(seed=5)
    res = FastSVD(k_interactions=3, adaptive_k=True, rank_without_identity=True).fit(X, U)
    assert np.count_nonzero(np.diag(res.A_est)) == 0 and res.diagnostics["rank_without_identity"] is True


def test_llc_mean_is_registered_and_follows_the_data_model():
    assert "llc_mean" in registry.keys() and registry.get("llc_mean").partial == "native"
    assert registry.build("llc_mean", registry.BuildContext(use_reg=True)).perturbation_mode == "regulator"
    assert registry.build("llc_mean", registry.BuildContext(use_reg=False)).perturbation_mode == "additive"
    assert any(d["key"] == "llc_mean" for d in registry.describe())


def test_llc_mean_regulator_mode_works_on_the_response_of_x_plus_u():
    rng = np.random.default_rng(1)
    n = 12
    F = np.zeros((n, n))
    for j in range(n):
        F[rng.choice([i for i in range(n) if i != j], size=2, replace=False), j] = rng.uniform(0.2, 0.4, 2) * rng.choice([-1, 1], 2)
    # X = c F^T (X + U) with c = 1: X = (I - F^T)^-1 F^T U, one single-gene perturbation per column
    U = np.eye(n) * 1.0
    X = np.linalg.solve(np.eye(n) - F.T, F.T @ U)
    res = LLCMean(k_interactions=2, ridge=1e-6, perturbation_mode="regulator").fit(X, U)
    assert res.A_est.shape == (n, n) and res.diagnostics["perturbation_mode"] == "regulator"
    true = np.abs(F) > 0
    hits = sum(true[i, j] for i in range(n) for j in np.argsort(-np.abs(res.A_est[i]))[:2])
    assert hits >= 0.6 * true.sum()


def test_the_diagnosis_page_matches_its_data():
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    page, data = root / "docs" / "results" / "fast_diagnosis.md", root / "docs" / "results" / "fast_diagnosis.json"
    if not data.exists():                                       # the Docker image ships no docs/
        import pytest
        pytest.skip("docs/results is not available (installed package or Docker image)")
    spec = importlib.util.spec_from_file_location("diagnose_fast", root / "scripts" / "diagnose_fast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert page.read_text() == mod.render(json.loads(data.read_text())) + "\n", "run python scripts/diagnose_fast.py --render docs/results/fast_diagnosis.md"
