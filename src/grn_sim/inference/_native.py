"""
Thread-safety for the native-library parts of the algorithms (BLAS limits, Numba compilation, process pools).

The web server runs requests on threads. ``threadpoolctl`` finds the loaded BLAS libraries by walking the process's
shared objects (``dl_iterate_phdr``, with a Python callback), and Numba's first compilation loads native modules
(``dlopen``). Done at the same moment on two threads - as when a page asks for several heavy results at once - the
interpreter lock and the system loader lock are taken in opposite orders and the server stops forever. So:

* the BLAS controller is built once (``blas_limit`` reuses it), under ``NATIVE_LOCK``;
* the Numba kernel compiles under the same lock, and ``warm_up()`` does both before the first request is served;
* process pools are created and forked under the same lock, so no thread forks while another is loading libraries.

The lock is reset in forked workers: they inherit it in whatever state the forking thread held it.
"""

import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional


class _ForkSafeRLock:
    """A re-entrant lock that starts unlocked in a forked child."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> "_ForkSafeRLock":
        self._lock.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._lock.release()

    def acquire(self, *a: Any, **k: Any) -> bool:
        return self._lock.acquire(*a, **k)

    def release(self) -> None:
        self._lock.release()

    def _reset(self) -> None:
        self._lock = threading.RLock()


NATIVE_LOCK = _ForkSafeRLock()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=NATIVE_LOCK._reset)

_CONTROLLER: Optional[Any] = None


def _controller() -> Optional[Any]:
    """The cached ``ThreadpoolController`` (None when threadpoolctl is missing); rebuilt if it saw no library."""
    global _CONTROLLER
    cached = _CONTROLLER
    if cached is False or (cached is not None and cached.lib_controllers):
        return cached or None                                  # the usual case: no lock, no library walk
    with NATIVE_LOCK:
        if _CONTROLLER is None or (_CONTROLLER is not False and not _CONTROLLER.lib_controllers):
            try:
                import numpy  # noqa: F401  (the BLAS libraries are only visible once these are loaded)
                import scipy.linalg  # noqa: F401
                from threadpoolctl import ThreadpoolController
                _CONTROLLER = ThreadpoolController()
            except Exception:  # pragma: no cover - threadpoolctl missing
                _CONTROLLER = False
        return _CONTROLLER or None


@contextmanager
def blas_limit(limits: Optional[int] = 1, user_api: Optional[str] = "blas") -> Iterator[None]:
    """``threadpool_limits(limits, user_api)`` without re-walking the loaded libraries on every call."""
    controller = _controller()
    if controller is None:
        yield
        return
    with controller.limit(limits=limits, user_api=user_api):
        yield


def warm_up() -> None:
    """Does the one-time native work (BLAS controller, Numba kernel) in the calling thread, before requests are served."""
    _controller()
    try:
        from grn_sim.inference import _g2s_fast
        _g2s_fast._hh_kernel()
    except Exception:  # pragma: no cover - numba missing or broken: the pure-numpy path is used
        pass
