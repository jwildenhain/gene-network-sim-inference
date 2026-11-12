"""A deadlocked pool worker is killed and its work finished in the parent instead of hanging the caller."""
import os
import time

import pytest

from grn_sim.inference import _parallel

MAIN_PID = os.getpid()


def _square_hang_in_child(payload, i):
    """Index 3 never returns in a forked worker (a deadlock stand-in); in the parent it computes normally."""
    if i == 3 and os.getpid() != MAIN_PID:
        time.sleep(3600)
    return i * i


def _chunk(chunk):
    if 3 in chunk and os.getpid() != MAIN_PID:
        time.sleep(3600)
    return [c * c for c in chunk]


def _boom(chunk):
    raise ValueError("bad chunk")


def test_run_chunks_reports_a_stall_and_returns_the_finished_chunks():
    t0 = time.time()
    done, stalled = _parallel.run_chunks(_chunk, [[0, 1], [2, 3], [4, 5]], 3, stall_seconds=2.0)
    assert stalled and time.time() - t0 < 60
    assert done == {0: [0, 1], 2: [16, 25]}  # the chunk holding index 3 never finished


def test_run_chunks_without_a_stall_returns_everything():
    done, stalled = _parallel.run_chunks(_chunk, [[0, 1], [2], [4, 5]], 3, stall_seconds=5.0)
    assert not stalled and done == {0: [0, 1], 1: [4], 2: [16, 25]}


def test_map_indices_finishes_the_missing_chunk_in_the_parent(monkeypatch):
    monkeypatch.setattr(_parallel, "STALL_IDLE_SECONDS", 2.0)
    out = _parallel.map_indices(_square_hang_in_child, None, 40, n_jobs=4, min_parallel=8)
    assert out == [i * i for i in range(40)]


def test_worker_exceptions_still_propagate():
    with pytest.raises(ValueError, match="bad chunk"):
        _parallel.run_chunks(_boom, [[0], [1]], 2, stall_seconds=5.0)
