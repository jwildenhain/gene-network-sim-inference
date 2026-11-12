"""Predicted matrices are stored sparsely for large networks and read back exactly."""
import json

import numpy as np

from grn_sim.evaluation.prediction_codec import decode_prediction, encode_prediction


def _pred(n, density=0.002, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) < density) * rng.choice([-1, 1], size=(n, n))


def test_large_matrix_round_trips_and_is_much_smaller():
    pred = _pred(500)
    stored = encode_prediction(pred)
    assert stored["sparse"] is True
    assert np.array_equal(decode_prediction(json.loads(json.dumps(stored))), pred)
    assert len(json.dumps(stored)) < len(json.dumps(pred.tolist())) / 20


def test_small_matrix_keeps_the_plain_format_and_old_archives_still_read():
    pred = _pred(30, 0.1)
    assert encode_prediction(pred) == pred.tolist()
    assert np.array_equal(decode_prediction(pred.tolist()), pred)


def test_empty_and_dense_extremes():
    assert np.array_equal(decode_prediction(encode_prediction(np.zeros((100, 100), dtype=int))), np.zeros((100, 100), dtype=int))
    full = np.ones((70, 70), dtype=int)
    assert np.array_equal(decode_prediction(encode_prediction(full)), full)
