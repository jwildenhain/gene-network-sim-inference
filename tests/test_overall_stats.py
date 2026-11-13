"""Tab 5's statistics are rebuilt from the archived sweep networks and stored in the archive."""

import json
import sqlite3

import pytest

from grn_sim.benchmarks import overall_stats as os_
from grn_sim.inference import registry

KEYS = [s.key for s in registry.specs()]


def _fake_metrics(run):
    """Deterministic per-run metrics: f1 depends on the algorithm's position and on the run's axis value."""
    base = {k: (i + 1) / (len(KEYS) + 1) for i, k in enumerate(KEYS)}
    return {k: {"f1": base[k], "total_errors": 10.0 + i, "power": 50.0, "specificity": 90.0, "ppv": 40.0, "signed_ppv": 30.0,
                "fdr": 60.0, "runtime": 0.5 + i} for i, k in enumerate(KEYS)}


@pytest.fixture
def fake_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(os_, "_run_metrics", lambda con, run_id: _fake_metrics(run_id))
    return str(tmp_path / "fake.sqlite")


def test_the_design_is_the_69_networks_of_the_sweep():
    d = os_.design()
    assert len(d) == 69 == len({x["run_id"] for x in d})
    by_axis = {a: sum(x["axis"] == a for x in d) for a in ("size", "conn", "noise", "dyn")}
    assert by_axis == {"size": 18, "conn": 21, "noise": 21, "dyn": 9}


def test_the_payload_covers_every_registered_algorithm_with_the_structure_the_page_reads(fake_archive):
    p = os_.build_payload(fake_archive)
    assert p["total_sweep_runs"] == 69 and p["algorithms_evaluated_count"] == len(KEYS)
    assert p["total_algorithmic_evaluations"] == 69 * len(KEYS)
    assert [x["algorithm_key"] for x in p["algorithm_leaderboard"]][0] == KEYS[-1]          # highest f1, deterministic
    assert {x["algorithm_key"] for x in p["algorithm_leaderboard"]} == set(KEYS)
    assert [x["rank"] for x in p["algorithm_leaderboard"]] == list(range(1, len(KEYS) + 1))
    assert set(p["architecture_summary"]) == {"random", "power_law", "hierarchical"}
    assert all(a["runs_evaluated"] == 23 for a in p["architecture_summary"].values())
    for sweep, axis in (("size_sweep", "sizes"), ("connectivity_sweep", "k_values"), ("noise_sweep", "noise_levels")):
        assert set(p[sweep]["algorithms"]) == set(KEYS)
        assert all(len(c["f1"]) == len(p[sweep][axis]) for c in p[sweep]["algorithms"].values())
    top = p["algorithm_leaderboard"][0]
    assert top["mean_f1"] == round(1 - 1 / (len(KEYS) + 1), 3) and top["runs_evaluated"] == 69
    json.dumps(p)                                                                          # serialisable (no NaN / numpy types)


def test_an_incomplete_archive_is_reported_not_papered_over(monkeypatch, tmp_path):
    def partial(con, run_id):
        m = _fake_metrics(run_id)
        m.pop(KEYS[0])
        return m
    monkeypatch.setattr(os_, "_run_metrics", partial)
    db = str(tmp_path / "p.sqlite")
    with pytest.raises(LookupError, match="lacks algorithms"):
        os_.build_payload(db)
    monkeypatch.setattr(os_, "_run_metrics", lambda con, run_id: None)
    with pytest.raises(LookupError, match="not in the archive"):
        os_.build_payload(db)


def test_the_snapshot_is_stored_in_the_database_and_reused_until_stale(tmp_path, monkeypatch):
    db = str(tmp_path / "t.sqlite")
    monkeypatch.setattr(os_, "_run_metrics", lambda con, run_id: _fake_metrics(run_id))
    first, origin = os_.current(db)
    assert origin == "rebuilt"
    again, origin = os_.current(db)
    assert origin == "stored" and again == first
    con = sqlite3.connect(db)
    assert con.execute("SELECT name, n_runs, n_algorithms FROM overall_stats").fetchall() == [("sweep", 69, len(KEYS))]
    con.close()
    stale = dict(first, algorithms_evaluated_count=len(KEYS) - 1)
    os_.store(stale, db)
    _, origin = os_.current(db)
    assert origin == "rebuilt"                                                             # a snapshot missing algorithms is rebuilt


def test_the_endpoint_serves_the_stored_snapshot(app_url, monkeypatch):
    import urllib.request
    monkeypatch.setattr(os_, "_run_metrics", lambda con, run_id: _fake_metrics(run_id))
    with urllib.request.urlopen(f"{app_url}/api/benchmark/overall_stats?refresh=1", timeout=60) as r:
        body = json.loads(r.read())
    assert body["snapshot"] == "rebuilt" and body["algorithms_evaluated_count"] == len(KEYS) and body["total_sweep_runs"] == 69
