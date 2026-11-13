"""Long algorithms report a fraction done; jobs show it as part of their progress."""
import numpy as np

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
