"""Heavy requests arriving together must not deadlock the server (threadpoolctl library walk vs Numba compile)."""
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pytest

from grn_sim.inference import _native
from grn_sim.inference.g2s import GeneralToSpecific


def _skip_if_blas_is_multithreaded():
    """These two tests would hang for good (the very bug they guard against) if OpenBLAS got its thread pool."""
    _native.warm_up()
    controller = _native._controller()
    threads = max([lib.num_threads for lib in controller.select(user_api="blas").lib_controllers] or [1])
    if threads > 1:
        pytest.skip(f"OpenBLAS runs {threads} threads: NumPy was imported before OPENBLAS_NUM_THREADS=1 was set "
                    "(export it in the environment that starts pytest)")


def _get(url):
    with urllib.request.urlopen(url, timeout=240) as r:
        return r.status


def test_blas_runs_on_one_thread():
    """The OpenBLAS thread pool must not exist: forking process pools beside it can hang the process for good."""
    import os

    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1", "set OPENBLAS_NUM_THREADS=1 before NumPy is imported"
    _native.warm_up()
    controller = _native._controller()
    assert all(lib.num_threads == 1 for lib in controller.select(user_api="blas").lib_controllers)


def test_the_blas_controller_is_built_once_and_reused():
    _native.warm_up()
    first = _native._controller()
    assert first is not None and first.lib_controllers          # numpy / scipy are loaded by now
    with _native.blas_limit(1):
        pass
    assert _native._controller() is first                        # not rebuilt (no new library walk per call)


def test_the_lock_is_reset_in_a_forked_child():
    import os

    with _native.NATIVE_LOCK:                                    # held here, as it is while a pool is being forked
        pid = os.fork()
        if pid == 0:                                             # the child must be able to take it
            ok = _native.NATIVE_LOCK.acquire(False)
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0


def test_concurrent_g2s_fits_finish():
    _skip_if_blas_is_multithreaded()
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, n))
    U = rng.normal(size=(n, n))
    errors, results = [], []

    def run():
        try:
            results.append(GeneralToSpecific(max_interactions=4, drop_rule="loo").fit(X, U).adjacency_pred.sum())
        except Exception as exc:                                # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    [t.start() for t in threads]
    [t.join(180) for t in threads]
    assert not any(t.is_alive() for t in threads), "a G2S fit never finished (deadlock)"
    assert not errors and len(set(results)) == 1                # all six agree


def test_heavy_requests_together_do_not_deadlock(app_url):
    _skip_if_blas_is_multithreaded()
    urls = [app_url + p for p in (
        "/api/benchmark/thesis_figures", "/api/benchmark/dataset?id=ecoli_sos", "/api/benchmark/dataset?id=dream4_10gene",
        "/api/benchmark/sos", "/api/benchmark/thesis_figures", "/api/benchmark/dataset?id=norman_2019_tf16",
        "/api/state", "/api/history")]
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = [pool.submit(_get, u) for u in urls]
        done = [f.result(timeout=300) for f in as_completed(futures, timeout=300)]
    assert done == [200] * len(urls)
