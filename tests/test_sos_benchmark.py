"""
Unit and integration tests for E. coli SOS pathway benchmark replication.
"""

from grn_sim.benchmarks.sos_pathway import run_sos_benchmark, load_sos_dataset, SOS_GROUND_TRUTH


def test_load_sos_dataset():
    X, U, sX, sU = load_sos_dataset()
    assert X.shape == (9, 9)
    assert U.shape == (9, 9)
    assert sX.shape == (9, 9)
    assert sU.shape == (9, 9)
    assert SOS_GROUND_TRUTH.shape == (9, 9)


def test_run_sos_benchmark():
    metrics = run_sos_benchmark(k_connections=5, by_chance_runs=10)
    for key in ["s2g", "g2s", "fast", "NIR", "voting", "by_chance"]:
        assert key in metrics
        m = metrics[key]
        assert 0.0 <= m.coverage_pct <= 100.0
        assert 0.0 <= m.false_positives_pct <= 100.0
        assert m.total_errors >= 0
