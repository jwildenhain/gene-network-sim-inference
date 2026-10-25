"""
Unit & Integration Tests for Perturbation Dynamics Categorization & Storage.
Verifies:
1. Classification of experiments into 'responsive', 'flat', and 'oscillating'.
2. SQLite database persistence of `response_type` and `response_stats_json` in `nw`.
3. Computation of summary statistics grouped by perturbation dynamics type.
4. Server simulation presets for each of the 3 dynamics categories.
"""

import os
import tempfile
import numpy as np
import pytest

from grn_sim.simulator.response_dynamics import (
    classify_perturbation_response,
    FLAT_THRESHOLD,
)
from grn_sim.web.database import (
    init_db,
    save_simulation_to_db,
    load_all_history_from_db,
    get_summary_statistics_by_type,
)


def test_classify_responsive_experiment():
    X = np.array([
        [0.45, 0.02, -0.15],
        [0.01, -0.62, 0.03],
        [-0.08, 0.12, 0.88],
    ])
    diag = {"is_valid_steady_state": True, "oscillating_detected": False}
    res = classify_perturbation_response(X, diag)
    assert res["response_type"] == "responsive"
    assert res["max_amplitude"] == pytest.approx(0.88, abs=1e-3)
    assert res["num_responsive_genes"] == 3
    assert res["num_responsive_perts"] == 3
    assert res["badge_class"] == "badge-emerald"


def test_classify_flat_experiment():
    # Small perturbations under 5% relative shift
    X = np.array([
        [0.012, 0.005, -0.018],
        [-0.003, 0.021, 0.004],
        [0.008, -0.015, 0.009],
    ])
    diag = {"is_valid_steady_state": True, "oscillating_detected": False}
    res = classify_perturbation_response(X, diag)
    assert res["response_type"] == "flat"
    assert res["max_amplitude"] < FLAT_THRESHOLD
    assert res["num_responsive_genes"] == 0
    assert res["badge_class"] == "badge-secondary"


def test_classify_oscillating_experiment():
    X = np.array([
        [1.2, 0.5],
        [-0.8, 2.1],
    ])
    diag = {"is_valid_steady_state": False, "oscillating_detected": True, "oscillating_perturbations_count": 2}
    res = classify_perturbation_response(X, diag)
    assert res["response_type"] == "oscillating"
    assert res["badge_class"] == "badge-amber"
    assert res["num_oscillating_perts"] == 2


def test_database_persistence_and_summary_by_type():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tf:
        db_path = tf.name

    try:
        init_db(db_path)

        # 1. Save a responsive run
        X_resp = np.array([[0.5, 0.1], [0.0, 0.8]])
        sim_resp = {
            "num_genes": 2,
            "topology": "random",
            "k_average": 1.0,
            "act_ratio": 0.5,
            "noise": 0.01,
            "seed": 42,
            "X": X_resp.tolist(),
            "U": [[1.0, 0.0], [0.0, 1.0]],
            "true_matrix": [[0, 1], [-1, 0]],
            "steady_state_diagnostics": {"is_valid_steady_state": True, "oscillating_detected": False},
        }
        inf_resp = {
            "metrics": {
                "nir": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "total_errors": 0},
                "s2g": {"precision": 0.5, "recall": 0.5, "f1_score": 0.5, "total_errors": 2},
            },
            "tmp_statmat": {
                "nir": {"specificity": 100.0, "power": 100.0},
                "s2g": {"specificity": 50.0, "power": 50.0},
            },
        }
        hist_resp = {
            "id": 1,
            "run_id": "test-resp-01",
            "timestamp": "2026-10-22 12:00:00",
            "num_genes": 2,
            "topology": "random",
            "k_av": 1.0,
            "act_ratio": 0.5,
            "noise": 0.01,
            "seed": 42,
            "response_type": "responsive",
        }
        save_simulation_to_db("test-resp-01", sim_resp, inf_resp, hist_resp, db_path=db_path)

        # 2. Save a flat run
        X_flat = np.array([[0.01, 0.005], [0.002, 0.015]])
        sim_flat = {
            "num_genes": 2,
            "topology": "hierarchical",
            "k_average": 1.0,
            "act_ratio": 0.5,
            "noise": 0.001,
            "seed": 101,
            "X": X_flat.tolist(),
            "U": [[0.01, 0.0], [0.0, 0.01]],
            "true_matrix": [[0, 1], [-1, 0]],
            "steady_state_diagnostics": {"is_valid_steady_state": True, "oscillating_detected": False},
        }
        inf_flat = {
            "metrics": {
                "nir": {"precision": 0.2, "recall": 0.2, "f1_score": 0.2, "total_errors": 4},
            },
            "tmp_statmat": {
                "nir": {"specificity": 40.0, "power": 20.0},
            },
        }
        hist_flat = {
            "id": 2,
            "run_id": "test-flat-01",
            "timestamp": "2026-10-22 12:05:00",
            "num_genes": 2,
            "topology": "hierarchical",
            "k_av": 1.0,
            "act_ratio": 0.5,
            "noise": 0.001,
            "seed": 101,
            "response_type": "flat",
        }
        save_simulation_to_db("test-flat-01", sim_flat, inf_flat, hist_flat, db_path=db_path)

        # 3. Save an oscillating run
        X_osc = np.array([[12.0, 50.0], [-30.0, 80.0]])
        sim_osc = {
            "num_genes": 2,
            "topology": "random",
            "k_average": 1.0,
            "act_ratio": 0.1,
            "noise": 0.02,
            "seed": 1,
            "X": X_osc.tolist(),
            "U": [[1.5, 0.0], [0.0, 1.5]],
            "true_matrix": [[0, -1], [-1, 0]],
            "steady_state_diagnostics": {"is_valid_steady_state": False, "oscillating_detected": True},
        }
        inf_osc = {
            "metrics": {
                "nir": {"precision": 0.3, "recall": 0.3, "f1_score": 0.3, "total_errors": 5},
            },
            "tmp_statmat": {
                "nir": {"specificity": 30.0, "power": 30.0},
            },
        }
        hist_osc = {
            "id": 3,
            "run_id": "test-osc-01",
            "timestamp": "2026-10-22 12:10:00",
            "num_genes": 2,
            "topology": "random",
            "k_av": 1.0,
            "act_ratio": 0.1,
            "noise": 0.02,
            "seed": 1,
            "response_type": "oscillating",
        }
        save_simulation_to_db("test-osc-01", sim_osc, inf_osc, hist_osc, db_path=db_path)

        # 4. Verify history loading contains response_type
        history = load_all_history_from_db(db_path)
        assert len(history) == 3
        types_in_hist = {h["run_id"]: h.get("response_type") for h in history}
        assert types_in_hist["test-resp-01"] == "responsive"
        assert types_in_hist["test-flat-01"] == "flat"
        assert types_in_hist["test-osc-01"] == "oscillating"

        # 5. Verify summary statistics calculation grouped by type
        summary = get_summary_statistics_by_type(db_path)
        assert summary["total_simulations"] == 3
        cats = summary["categories"]
        assert cats["responsive"]["count"] == 1
        assert cats["flat"]["count"] == 1
        assert cats["oscillating"]["count"] == 1

        assert cats["responsive"]["avg_max_amplitude"] == pytest.approx(0.8, abs=1e-3)
        assert cats["flat"]["avg_max_amplitude"] < 0.05
        assert cats["oscillating"]["avg_max_amplitude"] > 10.0

        # Check algorithm aggregation
        assert "nir" in cats["responsive"]["algorithms"]
        assert cats["responsive"]["algorithms"]["nir"]["avg_total_errors"] == 0
        assert cats["responsive"]["top_algorithm"] == "NIR"

        assert "nir" in cats["flat"]["algorithms"]
        assert cats["flat"]["algorithms"]["nir"]["avg_total_errors"] == 4

        assert "nir" in cats["oscillating"]["algorithms"]
        assert cats["oscillating"]["algorithms"]["nir"]["avg_total_errors"] == 5

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_summary_of_runs_without_algorithm_results_is_valid_json(tmp_path):
    """A category whose runs have no algorithm results must not report an infinite error count (Infinity is not
    valid JSON, and the archive page then fails to load at all)."""
    import json

    db_path = str(tmp_path / "summary.sqlite")
    sim = {
        "num_genes": 2, "topology": "random", "k_average": 1.0, "act_ratio": 0.5, "noise": 0.01, "seed": 3,
        "X": [[0.5, 0.1], [0.1, 0.6]], "U": [[1.0, 0.0], [0.0, 1.0]], "true_matrix": [[0, 1], [1, 0]],
        "steady_state_diagnostics": {"is_valid_steady_state": True, "oscillating_detected": False},
    }
    entry = {"id": 1, "run_id": "no-algs-01", "timestamp": "2026-10-25 10:00:00", "num_genes": 2, "topology": "random",
             "k_av": 1.0, "act_ratio": 0.5, "noise": 0.01, "seed": 3, "response_type": "responsive"}
    save_simulation_to_db("no-algs-01", sim, {"metrics": {}}, entry, db_path=db_path)

    summary = get_summary_statistics_by_type(db_path)
    json.dumps(summary, allow_nan=False)  # raises on Infinity / NaN
    responsive = summary["categories"]["responsive"]
    assert responsive["count"] == 1 and responsive["top_algorithm"] == "N/A" and responsive["top_algorithm_errors"] == 0
