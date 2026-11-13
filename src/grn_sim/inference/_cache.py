"""A small result cache so ensembles do not repeat the expensive member fits the studio has just run.

``vote`` and ``score_vote`` fit S2G, G2S (and NIR) themselves, and so do the stand-alone ``g2s`` / ``nir`` jobs of the same
network: at 1000 genes that is 825 s of G2S three times. Fits are keyed on the data (a hash of X and U), the algorithm's
settings and the call's keyword arguments, so a hit is the same computation; hits return a copy whose
``diagnostics["cached_seconds"]`` is what the fit originally took, which the studio adds to the reported run time (an
algorithm's cost is its stand-alone cost, not what the cache saved). Set ``GRN_SIM_NO_RESULT_CACHE=1`` to disable.
"""
import collections
import copy
import functools
import hashlib
import os
import threading
from typing import Any, Callable

import numpy as np

MAX_ENTRIES = 3          # a 2000-gene result is about 50 MB
_CACHE: "collections.OrderedDict[tuple, Any]" = collections.OrderedDict()
_LOCK = threading.Lock()


def enabled() -> bool:
    return os.environ.get("GRN_SIM_NO_RESULT_CACHE", "") in ("", "0")


def clear() -> None:
    with _LOCK:
        _CACHE.clear()


def cached_seconds(results) -> float:
    """Seconds the members of an ensemble got from the cache (to be added to the ensemble's run time)."""
    return float(sum(r.diagnostics.get("cached_seconds", 0.0) for r in results))


def result_cache(fit: Callable) -> Callable:
    @functools.wraps(fit)
    def wrapper(self, X, U, **kwargs):
        if not enabled():
            return fit(self, X, U, **kwargs)
        digest = hashlib.blake2b(digest_size=16)
        for a in (np.asarray(X), np.asarray(U)):
            digest.update(str(a.shape).encode())
            digest.update(np.ascontiguousarray(a).tobytes())
        settings = tuple(sorted((k, repr(v)) for k, v in vars(self).items()))
        key = (type(self).__name__, settings, repr(sorted(kwargs.items())), digest.digest())
        with _LOCK:
            hit = _CACHE.get(key)
            if hit is not None:
                _CACHE.move_to_end(key)
        if hit is not None:
            out = copy.deepcopy(hit)
            out.diagnostics = {**out.diagnostics, "cache_hit": True, "cached_seconds": float(hit.time_elapsed)}
            return out
        result = fit(self, X, U, **kwargs)
        with _LOCK:
            _CACHE[key] = copy.deepcopy(result)
            while len(_CACHE) > MAX_ENTRIES:
                _CACHE.popitem(last=False)
        return result

    return wrapper
