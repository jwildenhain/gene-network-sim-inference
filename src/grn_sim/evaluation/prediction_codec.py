"""Compact storage of predicted adjacency matrices in the archive.

A dense ``N x N`` matrix of 0 / +-1 as JSON costs about 12 MB at 2000 genes and every algorithm stores one; the
predictions hold a few thousand non-zero entries. Above ``SPARSE_ABOVE_GENES`` they are stored as coordinate lists;
smaller networks keep the plain nested lists older archives (and the sweep scripts) use. ``decode_prediction`` reads both.
"""
from typing import Any, Dict, List, Union

import numpy as np

SPARSE_ABOVE_GENES = 60

Encoded = Union[List[List[int]], Dict[str, Any]]


def encode_prediction(pred: np.ndarray, sparse_above: int = SPARSE_ABOVE_GENES) -> Encoded:
    arr = np.asarray(pred)
    if arr.ndim != 2 or arr.shape[0] <= sparse_above:
        return arr.tolist()
    rows, cols = np.nonzero(arr)
    return {"sparse": True, "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "rows": rows.tolist(), "cols": cols.tolist(), "vals": arr[rows, cols].astype(int).tolist()}


def decode_prediction(stored: Any) -> np.ndarray:
    """The integer matrix for either storage format."""
    if isinstance(stored, dict) and stored.get("sparse"):
        out = np.zeros(tuple(stored["shape"]), dtype=int)
        out[np.asarray(stored["rows"], dtype=int), np.asarray(stored["cols"], dtype=int)] = np.asarray(stored["vals"], dtype=int)
        return out
    return np.array(stored, dtype=int)
