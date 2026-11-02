"""scripts/transfer_study.py: the reproducible paired study of the cross-context predictors (tiny settings, offline)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "transfer_study.py"
spec = importlib.util.spec_from_file_location("transfer_study", SCRIPT)
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

TINY = dict(mode="input", genes=8, held_out=2, rewire=0.15, networks=2)
PREDICTORS = {"atlas_mean", "atlas_calibrated", "transpert_style", "pseudobulk_ridge", "null", "oracle_hill"}


@pytest.fixture(scope="module")
def study():
    return ts.run_study(**TINY)


def test_every_predictor_is_scored_once_per_network_on_both_views(study):
    assert set(study["held_out"]) == PREDICTORS
    for parts in study["held_out"].values():
        assert len(parts["all"]) == len(parts["off_target"]) == TINY["networks"]
        assert np.all(np.isfinite(parts["all"])) and np.all(np.isfinite(parts["off_target"]))
    assert set(study["seen"]) == PREDICTORS - {"oracle_hill"}


def test_the_study_is_deterministic(study):
    assert ts.run_study(**TINY) == study


def test_summary_pairs_every_predictor_with_the_source_average(study):
    summary = ts.summarise({name: parts["all"] for name, parts in study["held_out"].items()})
    assert summary[ts.BASELINE]["minus_baseline"] == 0.0 and summary[ts.BASELINE]["wins"] == 0
    assert all(s["n"] == TINY["networks"] for s in summary.values())
    report = ts.format_report(study, "title")
    assert "off-target genes only" in report and "SEEN target perturbations" in report


def test_seed_offset_gives_different_networks(study):
    assert ts.run_study(**TINY, seed_offset=50) != study


def test_command_line_writes_json_and_rejects_bad_input(tmp_path, capsys):
    out = tmp_path / "study.json"
    assert ts.main(["--genes", "8", "--held-out", "2", "--networks", "2", "--json", str(out)]) == 0
    assert "minus atlas_mean" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["settings"]["networks"] == 2 and "held_out" in data["scores"]
    with pytest.raises(SystemExit):
        ts.main(["--genes", "8", "--held-out", "8"])
    with pytest.raises(SystemExit):
        ts.main(["--rewire", "2"])
