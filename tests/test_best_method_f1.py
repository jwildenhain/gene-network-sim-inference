"""The archive's "best method" is the algorithm with the highest F1, not the one with the fewest total errors."""

import copy
import json

import pytest

from grn_sim.web import server
from grn_sim.web.database import (
    _rebase_best_on_f1,
    get_summary_statistics_by_type,
    load_all_history_from_db,
    save_simulation_to_db,
)

# The audit's case: on 18 true edges an empty graph makes 18 errors, which beat a useful algorithm's 27.
SUMMARY = {
    "notears":   {"power": 0.0,  "ppv": 0.0,  "specificity": 100.0, "total_errors": 18},
    "arachne":   {"power": 5.6,  "ppv": 33.0, "specificity": 98.0,  "total_errors": 19},
    "causalgrn": {"power": 61.0, "ppv": 55.0, "specificity": 93.0,  "total_errors": 27},
    "genie3":    {"power": 94.0, "ppv": 14.0, "specificity": 13.0,  "total_errors": 112},
}


def _legacy_entry():
    return {"run_id": "legacy", "best_algorithm": "NOTEARS", "best_alg_key": "notears", "best_total_errors": 18,
            "best_power_pct": 0.0, "best_specificity_pct": 100.0, "metrics_summary": copy.deepcopy(SUMMARY)}


def test_legacy_entries_are_rebased_to_the_highest_f1():
    entry = _legacy_entry()
    _rebase_best_on_f1(entry)
    assert entry["best_alg_key"] == "causalgrn" and entry["best_algorithm"] == "CAUSALGRN"
    assert entry["best_total_errors"] == 27 and entry["best_power_pct"] == 61.0
    assert entry["f1_score"] == pytest.approx(2 * 0.61 * 0.55 / (0.61 + 0.55), abs=1e-4)
    assert entry["best_basis"] == "f1"


def test_fewest_errors_breaks_an_f1_tie_and_entries_marked_f1_are_left_alone():
    tie = {"metrics_summary": {"a": {"power": 50.0, "ppv": 50.0, "total_errors": 9},
                               "b": {"power": 50.0, "ppv": 50.0, "total_errors": 4}}}
    _rebase_best_on_f1(tie)
    assert tie["best_alg_key"] == "b"
    done = {"best_basis": "f1", "best_alg_key": "x", "metrics_summary": copy.deepcopy(SUMMARY)}
    _rebase_best_on_f1(done)
    assert done["best_alg_key"] == "x"  # new entries are exactly as written
    _rebase_best_on_f1({"best_alg_key": "x"})  # nothing to derive from: no crash, no change


def test_loading_the_archive_shows_f1_based_best_methods_for_old_runs(tmp_path):
    db = str(tmp_path / "a.sqlite")
    sim = {"num_genes": 2, "topology": "random", "k_average": 1.0, "act_ratio": 0.5, "noise": 0.01, "seed": 1,
           "X": [[0.5, 0.1], [0.1, 0.6]], "U": [[1.0, 0.0], [0.0, 1.0]], "true_matrix": [[0, 1], [1, 0]],
           "steady_state_diagnostics": {"is_valid_steady_state": True, "oscillating_detected": False}}
    save_simulation_to_db("legacy", sim, {"metrics": {}}, _legacy_entry(), db_path=db)
    (entry,) = load_all_history_from_db(db)
    assert entry["best_alg_key"] == "causalgrn" and entry["f1_score"] > 0.5
    # the stored row was not rewritten
    import sqlite3
    stored = json.loads(sqlite3.connect(db).execute("select history_entry_json from nw").fetchone()[0])
    assert stored["best_alg_key"] == "notears"


def test_summary_cards_name_the_top_algorithm_by_mean_f1(tmp_path):
    db = str(tmp_path / "s.sqlite")
    sim = {"num_genes": 2, "topology": "random", "k_average": 1.0, "act_ratio": 0.5, "noise": 0.01, "seed": 1,
           "X": [[0.5, 0.1], [0.1, 0.6]], "U": [[1.0, 0.0], [0.0, 1.0]], "true_matrix": [[0, 1], [1, 0]],
           "steady_state_diagnostics": {"is_valid_steady_state": True, "oscillating_detected": False}}
    metrics = {a: {"precision": s["ppv"] / 100, "recall": s["power"] / 100, "signed_ppv": 0.0, "total_errors": s["total_errors"],
                   "f1_score": 2 * s["power"] * s["ppv"] / (s["power"] + s["ppv"]) / 100 if s["power"] + s["ppv"] else 0.0}
               for a, s in SUMMARY.items()}
    inferred = {"metrics": metrics, "tmp_statmat": {a: {"specificity": s["specificity"], "power": s["power"]} for a, s in SUMMARY.items()}}
    entry = {"id": 1, "run_id": "r1", "timestamp": "2026-11-02 10:00:00", "num_genes": 2, "topology": "random", "k_av": 1.0,
             "act_ratio": 0.5, "noise": 0.01, "seed": 1, "response_type": "responsive"}
    save_simulation_to_db("r1", sim, inferred, entry, db_path=db)
    cat = get_summary_statistics_by_type(db)["categories"]["responsive"]
    assert cat["top_algorithm"] == "CAUSALGRN"  # NOTEARS has the fewest errors but recovers nothing
    assert cat["top_algorithm_f1"] > 0.5 and cat["top_algorithm_errors"] == 27


def test_new_runs_pick_the_best_method_by_f1_and_say_so():
    server._execute_simulation({"num_genes": 12, "seed": 15, "topology": "random", "k_av": 1.5, "noise": 0.02,
                                "kinetics_engine": "cellbox_neural_ode"})
    server._execute_inference(["notears", "arachne", "causalgrn", "g2s", "by_chance"])
    entry = server.SIMULATION_HISTORY[0]
    metrics = server.SIMULATION_RUN_STORE[entry["run_id"]]["inferred_payload"]["metrics"]
    expected = max(metrics, key=lambda k: (metrics[k]["f1_score"], -metrics[k]["total_errors"]))
    assert entry["best_alg_key"] == expected and entry["best_basis"] == "f1"
    assert entry["best_total_errors"] == metrics[expected]["total_errors"]
    fewest = min(metrics, key=lambda k: metrics[k]["total_errors"])
    if fewest != expected:  # the case this change is about
        assert metrics[expected]["f1_score"] > metrics[fewest]["f1_score"]


def test_an_incomplete_summary_is_left_as_stored():
    entry = {"best_alg_key": "vote", "best_power_pct": 80, "metrics_summary": {"vote": {"ppv": 50}, "nir": {"power": 90}}}
    before = copy.deepcopy(entry)
    _rebase_best_on_f1(entry)
    assert entry == before
