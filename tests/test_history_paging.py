"""The archive history is served as ranked, sorted, paged slim rows (grn_sim.web.history_view)."""
import json
import urllib.error
import urllib.request

import pytest

from grn_sim.web import database as db
from grn_sim.web import history_view as hv


def _entry(i, rtype="responsive", genes=16, f1=0.5, errors=10, power=50.0, topology="random", osc=False):
    return {"run_id": f"run-{i:03d}", "response_type": rtype, "num_genes": genes, "f1_score": f1, "best_total_errors": errors,
            "best_power_pct": power, "best_specificity_pct": 90.0, "topology": topology, "k_av": 1.8, "noise": 0.05,
            "oscillating": osc, "best_algorithm": "VOTE", "best_alg_key": "vote", "timestamp": f"2026-01-{(i % 28) + 1:02d} 10:00:00",
            "metrics_summary": {"vote": {"power": power}}, "response_dynamics": {"big": list(range(200))}}


def _view(entries):
    return hv.ArchiveView(entries)


def _page(view, **kw):
    q = {k: [str(v)] for k, v in kw.items()}
    return hv.page_of(view, hv.parse_params(q))


def test_slim_rows_drop_the_heavy_detail():
    row = hv.slim_row(_entry(1))
    assert "metrics_summary" not in row and "response_dynamics" not in row
    assert len(json.dumps(row)) < 600 and row["rankInSet"] is None and row["setName"] == "All runs"


def test_curated_sets_rank_like_the_page_always_did():
    entries = [_entry(0, f1=0.9, genes=8)]                                        # too small for a curated set
    entries += [_entry(i, f1=0.5 + i / 100, errors=i) for i in range(1, 25)]      # 24 candidates, F1 rising with i
    entries += [_entry(30, f1=0.8, errors=5, power=40.0), _entry(31, f1=0.8, errors=5, power=60.0),   # F1 and errors tie: power decides
                _entry(32, f1=0.8, errors=2, power=10.0)]                         # F1 ties: fewer errors win
    resp = _page(_view(entries), set="responsive", page_size=50)
    ids = [r["run_id"] for r in resp["history"]]
    assert len(ids) == hv.CURATED_SIZE and "run-000" not in ids                        # top 15 only, N >= 12 only
    # the three hand-made 0.8 runs outrank the rising series (0.51 .. 0.74): F1 first, then fewer errors, then more power
    assert ids[:4] == ["run-032", "run-031", "run-030", "run-024"] and ids[-1] == "run-013"
    top = [r["run_id"] for r in _page(_view([_entry(30, f1=0.8, errors=5, power=40.0), _entry(31, f1=0.8, errors=5, power=60.0),
                                             _entry(32, f1=0.8, errors=2, power=10.0)]), set="responsive")["history"]]
    assert top == ["run-032", "run-031", "run-030"]
    assert [r["rankInSet"] for r in resp["history"]] == list(range(1, hv.CURATED_SIZE + 1))
    assert {r["setName"] for r in resp["history"]} == {"Responsive"}


def test_the_default_curated_view_is_the_three_dynamics_sets_in_order():
    entries = [_entry(i, rtype=t, f1=0.4 + i / 100) for i, t in enumerate(["flat", "oscillating", "responsive"] * 3)]
    resp = _page(_view(entries), set="all", page_size=50)
    assert [r["setName"] for r in resp["history"]] == ["Responsive"] * 3 + ["Flat Levels"] * 3 + ["Oscillating"] * 3
    assert resp["view"]["total"] == 9 and resp["view"]["total_pages"] == 1


def test_everything_keeps_archive_order_and_has_no_rank():
    entries = [_entry(i, genes=5 + i) for i in range(30)]                             # small networks count here
    resp = _page(_view(entries), set="everything", page_size=10)
    assert [r["run_id"] for r in resp["history"]] == [f"run-{i:03d}" for i in range(10)]
    assert all(r["rankInSet"] is None for r in resp["history"])


def test_topology_and_overall_sets():
    entries = [_entry(1, topology="modular", f1=0.9), _entry(2, topology="power_law", f1=0.8), _entry(3, topology="modular", f1=0.7)]
    assert [r["run_id"] for r in _page(_view(entries), set="modular")["history"]] == ["run-001", "run-003"]
    assert [r["run_id"] for r in _page(_view(entries), set="power_law")["history"]] == ["run-002"]
    assert [r["run_id"] for r in _page(_view(entries), set="overall_top15")["history"]] == ["run-001", "run-002", "run-003"]
    assert _view([{**_entry(4), "topology": None}]).sets["random"][0]["run_id"] == "run-004"     # a missing topology counts as random


def test_paging_arithmetic():
    view = _view([_entry(i) for i in range(23)])
    p1, p3, past = (_page(view, set="everything", page=n, page_size=10) for n in (1, 3, 99))
    assert (p1["view"]["total"], p1["view"]["total_pages"], p1["view"]["first_row"], p1["view"]["last_row"]) == (23, 3, 1, 10)
    assert len(p3["history"]) == 3 and (p3["view"]["first_row"], p3["view"]["last_row"]) == (21, 23)
    assert past["view"]["page"] == 3 and past["history"] == p3["history"]                      # beyond the end: the last page
    empty = _page(_view([]), set="everything")
    assert empty["history"] == [] and empty["view"]["total_pages"] == 1 and empty["view"]["first_row"] == 0


def test_sorting_covers_the_whole_result_not_just_a_page():
    view = _view([_entry(i, f1=i / 100, errors=100 - i) for i in range(40)])
    first = _page(view, set="everything", sort="f1", order="desc", page_size=10)["history"]
    assert [r["run_id"] for r in first] == [f"run-{i:03d}" for i in range(39, 29, -1)]          # the best of all 40, on page 1
    last = _page(view, set="everything", sort="errors", order="desc", page=4, page_size=10)["history"]
    assert [r["best_total_errors"] for r in last] == sorted(r["best_total_errors"] for r in last)[::-1] and last[-1]["best_total_errors"] == 61
    ranked = _page(_view([_entry(i, f1=0.5) for i in range(20)]), set="everything", sort="rank")
    assert len(ranked["history"]) == 20                                                         # no rank anywhere: all rows, none lost


def test_sorting_by_what_a_column_shows():
    entries = [_entry(1, rtype="flat"), _entry(2, rtype="oscillating", osc=True), _entry(3, rtype="responsive")]
    cats = [r["response_type"] for r in _page(_view(entries), set="everything", sort="category")["history"]]
    assert cats == ["responsive", "flat", "oscillating"]                                        # '⚡ Responsive' < '⚪ Flat Levels' < '〰 Oscillating'
    eq = [r["run_id"] for r in _page(_view(entries), set="everything", sort="equilibrium")["history"]]
    assert eq[0] == "run-002" and set(eq[1:]) == {"run-001", "run-003"}                         # 'Oscillating' < 'Steady State', like the browser's text sort


def test_search_filters_by_run_id():
    view = _view([_entry(i) for i in range(30)])
    resp = _page(view, set="everything", q="run-02", page_size=50)
    assert [r["run_id"] for r in resp["history"]] == [f"run-02{i}" for i in range(10)]
    assert resp["view"]["total"] == 10 and resp["view"]["runs_in_list"] == 30
    assert _page(view, set="everything", q="RUN-029")["history"][0]["run_id"] == "run-029"      # not case sensitive
    assert _page(view, set="everything", q="nothing")["history"] == []


@pytest.mark.parametrize("query, fragment", [
    ({"set": ["nope"]}, "set must be one of"), ({"page": ["0"]}, "page must be at least 1"), ({"page": ["x"]}, "whole numbers"),
    ({"page_size": ["0"]}, "page_size must be between"), ({"page_size": ["10000"]}, "page_size must be between"),
    ({"sort": ["bogus"]}, "sort must be one of"), ({"order": ["sideways"]}, "order must be asc or desc"),
    ({"q": ["x" * 101]}, "at most 100"),
])
def test_bad_parameters_are_refused(query, fragment):
    with pytest.raises(ValueError, match=fragment):
        hv.parse_params(query)


def test_the_view_is_rebuilt_only_when_the_archive_changes(monkeypatch):
    calls, version = [], [("db", 1, 100)]
    monkeypatch.setattr(db, "load_all_history_from_db", lambda: (calls.append(1), [_entry(1)])[1])
    monkeypatch.setattr(hv, "_archive_version", lambda: version[0])
    monkeypatch.setitem(hv._CACHE, "key", None)
    first = hv.get_view()
    assert hv.get_view() is first and hv.get_view() is first and len(calls) == 1               # cached
    version[0] = ("db", 2, 140)                                                                 # the archive file changed
    assert hv.get_view() is not first and len(calls) == 2


def test_the_summary_is_computed_once_per_archive_version(monkeypatch):
    calls, version = [], [("db", 1, 100)]
    monkeypatch.setattr(db, "load_all_history_from_db", lambda: [_entry(1)])
    monkeypatch.setattr(db, "get_summary_statistics_by_type", lambda: (calls.append(1), {"x": 1})[1])
    monkeypatch.setattr(hv, "_archive_version", lambda: version[0])
    monkeypatch.setitem(hv._CACHE, "key", None)
    assert hv.get_summary() == {"x": 1} and hv.get_summary() == {"x": 1} and len(calls) == 1
    version[0] = ("db", 2, 150)
    hv.get_summary()
    assert len(calls) == 2


# ---- over HTTP ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seeded(app_url):
    """An archive with a few runs (the in-process archive starts empty); 12 genes, so the curated views list them."""
    for seed in (11, 12, 13, 14):
        _post(app_url + "/api/simulate", {"num_genes": 12, "seed": seed, "topology": "random", "auto_infer": True,
                                          "algorithms": ["s2g", "vote"]})
    return app_url


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_legacy_form_still_returns_every_run_with_its_detail(seeded):
    app_url = seeded
    status, body = _get(app_url + "/api/history")
    data = json.loads(body)
    assert status == 200 and data["history"] and "summary_by_type" in data
    assert "metrics_summary" in data["history"][0] and "view" not in data


def test_the_paged_form_returns_slim_rows_and_far_fewer_bytes(seeded):
    app_url = seeded
    legacy = _get(app_url + "/api/history")[1]
    status, body = _get(app_url + "/api/history?set=everything&page_size=3")
    page = json.loads(body)
    assert status == 200 and len(page["history"]) == 3 and page["view"]["page_size"] == 3
    assert "metrics_summary" not in page["history"][0] and "summary_by_type" not in page
    assert page["view"]["total"] == page["view"]["runs_in_list"] >= 3
    assert len(body) < len(legacy) / 5                                                          # an order of magnitude less to send
    assert page["database"]["nw_rows"] >= 1 and "active_run_id" in page


def test_bad_parameters_get_a_400_over_http(seeded):
    app_url = seeded
    for q in ("set=zzz", "page=-1", "page_size=999", "sort=drop", "order=up"):
        status, body = _get(app_url + "/api/history?" + q)
        assert status == 400 and "error" in json.loads(body), q


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def test_a_new_run_shows_up_in_the_paged_view(seeded):
    app_url = seeded
    sim = _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 77, "topology": "random", "auto_infer": True, "algorithms": ["s2g", "vote"]})
    run_id = sim["run_id"]
    page = json.loads(_get(app_url + "/api/history?set=everything&page_size=5")[1])
    assert page["history"][0]["run_id"] == run_id                                               # newest first, cache invalidated
    found = json.loads(_get(app_url + f"/api/history?set=everything&q={run_id}")[1])
    assert [r["run_id"] for r in found["history"]] == [run_id]


# ---- algorithm comparison follows the listed runs ------------------------------------------------------------------------
def _json(url):
    return json.loads(_get(url)[1])


def _comparison(app_url, query):
    return _json(app_url + "/api/archive/algorithm_comparison?" + query)


def _counts(payload):
    """Runs behind each algorithm row of the comparison, per category."""
    return {c: [a["runs_count"] for a in v["algorithms"].values()] for c, v in payload["summary"]["categories"].items()}


def test_summary_statistics_can_be_limited_to_some_runs(seeded):
    ids = [r["run_id"] for r in _json(seeded + "/api/history?set=everything&page_size=2")["history"]]
    both = db.get_summary_statistics_by_type(run_ids=ids)
    one = db.get_summary_statistics_by_type(run_ids=ids[:1])
    nothing = db.get_summary_statistics_by_type(run_ids=[])
    assert both["total_simulations"] == 2 and one["total_simulations"] == 1 and nothing["total_simulations"] == 0
    assert all(n == 1 for counts in _counts({"summary": one}).values() for n in counts)
    assert not any(_counts({"summary": nothing}).values())


def test_the_comparison_covers_exactly_the_runs_of_the_view(seeded):
    everything = _comparison(seeded, "set=everything")
    assert everything["runs"] == _json(seeded + "/api/history?set=everything&page_size=1")["view"]["total"]
    for view in ("all", "flat", "responsive", "overall_top15"):
        listed = _json(seeded + f"/api/history?set={view}&page_size=200")["view"]["total"]
        payload = _comparison(seeded, f"set={view}")
        assert payload["runs"] == listed
        assert payload["summary"]["total_simulations"] == listed
        assert all(n <= listed for counts in _counts(payload).values() for n in counts)


def test_the_comparison_follows_the_id_search(seeded):
    ids = [r["run_id"] for r in _json(seeded + "/api/history?set=everything&page_size=3")["history"]]
    one = _comparison(seeded, f"set=everything&q={ids[0]}")
    assert one["runs"] >= 1 and one["q"] == ids[0]
    assert all(n <= one["runs"] for counts in _counts(one).values() for n in counts)
    none = _comparison(seeded, "set=everything&q=no-such-run-anywhere")
    assert none["runs"] == 0 and not any(_counts(none).values())


def test_the_comparison_refuses_bad_parameters(seeded):
    for query in ("set=nonsense", "q=" + "x" * 101):
        status, body = _get(seeded + "/api/archive/algorithm_comparison?" + query)
        assert status == 400 and "error" in json.loads(body)


def test_the_comparison_is_cached_per_set_and_search_text(monkeypatch):
    view = _view([_entry(i, f1=0.5 + i / 100) for i in range(4)])
    calls = []
    monkeypatch.setattr(hv, "get_view", lambda: view)
    monkeypatch.setattr(db, "get_summary_statistics_by_type", lambda **kw: calls.append(kw) or {"categories": {}})
    params = hv.parse_params({"set": ["everything"], "q": ["run-00"]})
    first = hv.get_comparison(params)
    assert hv.get_comparison(params) is first and len(calls) == 1
    assert first["runs"] == 4 and sorted(calls[0]["run_ids"]) == [f"run-{i:03d}" for i in range(4)]
    hv.get_comparison(hv.parse_params({"set": ["everything"], "q": ["run-001"]}))
    assert len(calls) == 2
