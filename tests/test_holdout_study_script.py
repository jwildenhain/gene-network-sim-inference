"""scripts/holdout_study.py: the reproducible paired study behind docs/HOLDOUT_RESEARCH.md (tiny settings, offline)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "holdout_study.py"
spec = importlib.util.spec_from_file_location("holdout_study", SCRIPT)
hs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hs)

TINY = dict(genes=8, held_out=2, noise=0.01, networks=2, algorithms=("g2s",))


@pytest.fixture(scope="module")
def activation():
    return hs.run_study("input", **TINY)


@pytest.fixture(scope="module")
def knockdown():
    return hs.run_study("knockdown", **TINY)


def test_every_predictor_is_scored_once_per_network_on_both_views(activation):
    for view in ("all", "off_target"):
        assert set(activation[view]) == {"null", "oracle", "hill_search", "g2s"}
        assert all(len(v) == TINY["networks"] and np.all(np.isfinite(v)) for v in activation[view].values())


def test_the_target_only_baseline_exists_only_for_knockdown_data(activation, knockdown):
    assert "self_only" not in activation["all"] and "self_only" in knockdown["all"]
    assert all(v == 0.0 for v in knockdown["off_target"]["self_only"]), "no downstream skill by construction"
    assert np.mean(knockdown["all"]["self_only"]) > 0.4


def test_structure_recovery_of_the_search_is_reported(activation):
    s = activation["structure"]
    assert len(s["precision"]) == len(s["recall"]) == TINY["networks"]
    assert all(0.0 <= v <= 1.0 for v in s["precision"] + s["recall"]) and "precision" in hs.format_report(activation, "t")


def test_the_study_is_deterministic(activation):
    assert hs.run_study("input", **TINY) == activation


def test_summary_pairs_every_predictor_with_the_null(knockdown):
    summary = hs.summarise(knockdown["all"])
    assert summary["null"]["minus_null"] == 0.0 and summary["null"]["wins"] == 0
    assert all(s["n"] == TINY["networks"] for s in summary.values())
    report = hs.format_report(knockdown, "title")
    assert "OFF-TARGET GENES ONLY" in report and "self_only" in report


def test_command_line_writes_json_and_rejects_bad_input(tmp_path, capsys):
    out = tmp_path / "study.json"
    assert hs.main(["--genes", "8", "--held-out", "2", "--networks", "2", "--algorithms", "g2s", "--json", str(out)]) == 0
    assert "minus null" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["settings"]["networks"] == 2 and "all" in data["scores"]
    with pytest.raises(SystemExit):
        hs.main(["--algorithms", "not_an_algorithm"])
    with pytest.raises(SystemExit):
        hs.main(["--genes", "8", "--held-out", "8"])
