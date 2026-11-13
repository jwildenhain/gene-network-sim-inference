"""Long algorithms report a fraction done; jobs show it as part of their progress."""
import numpy as np
import pytest

from grn_sim.inference import _g2s_fast as fast
from grn_sim.inference import _parallel
from grn_sim.web import jobs as J


def _problem(n=40):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, n))
    U = rng.normal(size=(n, n))
    p = fast.G2SParams(criterion="mdl", min_interactions=1, max_interactions=5, drop_rule="loo")
    return X, U, p


def test_serial_and_pool_report_fraction_up_to_one():
    X, U, p = _problem()
    basis = fast.make_shared_basis(X)
    for n_jobs in (1, 3):
        seen = []
        with _parallel.progress_to(seen.append):
            fast.solve_genes(U, basis, p, list(range(U.shape[0])), n_jobs=n_jobs)
        assert seen, n_jobs
        assert all(0.0 <= a <= b <= 1.0 for a, b in zip(seen, seen[1:]))
        if n_jobs == 1:
            assert seen[-1] == 1.0 and len(seen) == U.shape[0]


def test_no_hook_is_a_noop():
    _parallel.report_progress(0.5)


def test_hook_errors_do_not_break_the_fit():
    X, U, p = _problem(20)
    basis = fast.make_shared_basis(X)

    def boom(_):
        raise RuntimeError("x")

    with _parallel.progress_to(boom):
        fast.solve_genes(U, basis, p, list(range(20)), n_jobs=1)


def test_job_progress_includes_the_fraction():
    job = J.Job("job-1", "t", None, "run-1", None, ["g2s", "s2g"])
    ctx = J.JobContext(type("M", (), {"_lock": __import__("threading").RLock()})(), job)
    ctx.fraction(0.5)
    assert job.to_dict()["progress_pct"] == 25.0        # (0 + 0.5) of 2 steps
    ctx.progress(advance=True)
    assert job.sub_fraction == 0.0 and job.to_dict()["progress_pct"] == 50.0
    ctx.fraction(5.0)
    assert job.to_dict()["progress_pct"] < 100.0


# every algorithm that can run for minutes reports how far it is (cpa_lite in regulator mode is a closed-form solve, 1 s at
# 1000 genes; the other fast ones - mean_difference, atlasshift, pearson, sfinn, dmdgrn, inspre - are seconds at 2000 genes)
LONG_RUNNING = ["s2g", "g2s", "nir", "vote", "score_vote", "genie3", "grnboost", "causalgrn", "ridge_cv",
                "cellbox", "notears", "arachne"]


@pytest.fixture(scope="module")
def small_hill():
    from grn_sim.perturbseq.pipeline import build_network
    from grn_sim.simulator.model_engine import run_simulation_with_engine
    net = build_network("replogle_like", 40, seed=5)
    exp = run_simulation_with_engine(network=net, engine="hill_nonlinear", perturbation_factor=2.0,
                                     measurement_noise=0.02, seed=3)
    return np.asarray(exp.X, float), np.asarray(exp.U, float)


@pytest.mark.parametrize("key", LONG_RUNNING)
def test_long_running_algorithms_report_progress(key, small_hill, monkeypatch):
    from grn_sim.inference import registry
    monkeypatch.setenv("GRN_SIM_NO_RESULT_CACHE", "1")
    X, U = small_hill
    seen = []
    with _parallel.progress_to(seen.append):
        registry.build(key, registry.BuildContext(k_est=3, use_reg=True, is_sos=False)).fit(X, U)
    assert len(seen) >= 2, (key, seen)
    assert all(0.0 <= v <= 1.0 for v in seen)
    assert max(seen) > min(seen)


def test_progress_slice_maps_into_its_share():
    seen = []
    with _parallel.progress_to(seen.append):
        with _parallel.progress_slice(0.2, 0.6):
            _parallel.report_progress(0.5)
    assert seen == [pytest.approx(0.4)]
