"""Archive tab: the algorithm comparison follows the active filter and id search, and the id search says what an id is."""
import json
import re
import urllib.parse
import urllib.request

import pytest

from tests.test_playwright_archive_paging import _api, _open_archive, _run_ids, _status, archive  # noqa: F401

pytest.importorskip("playwright.sync_api")


def _open_comparison(page):
    if not page.is_visible("#archive-alg-breakdown-wrapper"):
        page.click("#btn-toggle-alg-breakdown")
    page.wait_for_selector("#archive-alg-breakdown-wrapper", state="visible")


def _scope(page):
    return page.inner_text("#archive-alg-scope")


def _wait_scope(page, text):
    page.wait_for_function(f"document.getElementById('archive-alg-scope').innerText.includes({json.dumps(text)})", timeout=30000)


def _search(page, text):
    page.fill("#archive-search", text)
    page.wait_for_function(f"archiveView().q === {json.dumps(text.strip())}", timeout=10000)


def _table_rows(page):
    """(category, algorithm, runs) of every comparison row."""
    return page.eval_on_selector_all(
        "#tbl-dynamics-alg-comparison tbody tr",
        "rows => rows.map(r => { const t = r.querySelectorAll('td'); return [t[0].innerText.trim(), t[1].innerText.trim(), Number(t[2].innerText)]; })")


def _expected_rows(app_url, view, q=""):
    qs = urllib.parse.urlencode({"set": view, **({"q": q} if q else {})})
    cats = _api(app_url, "/api/archive/algorithm_comparison?" + qs)["summary"]["categories"]
    return sum(len(c["algorithms"]) for c in cats.values())


def test_comparison_counts_the_runs_of_the_listed_view(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)
    total = _api(archive, "/api/history?set=everything&page_size=1")["view"]["total"]
    _wait_scope(page, f"Averages over the {total} runs")
    assert "All runs" in _scope(page)
    assert len(_table_rows(page)) == _expected_rows(archive, "everything") > 0
    assert all(1 <= n <= total for _, _, n in _table_rows(page))
    page.close()


def test_changing_the_curated_view_changes_the_comparison(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)
    everything_rows = len(_table_rows(page))
    for view in ("flat", "overall_top15", "oscillating", "everything"):
        page.select_option("#archive-filter-type", view)
        expected = _api(archive, f"/api/history?set={view}&page_size=1")["view"]["total"]
        _wait_scope(page, f"Averages over the {expected} run" if expected else "No runs in the current list")
        assert len(_table_rows(page)) == _expected_rows(archive, view)
        if expected == 0:
            assert not _table_rows(page) and "nothing to compare" in _scope(page)
    assert len(_table_rows(page)) == everything_rows                       # back to the full list: back to the full table
    page.close()


def test_a_category_view_only_shows_that_category(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)
    target = _api(archive, "/api/history?set=everything&page_size=1")["history"][0]["response_type"]
    page.select_option("#archive-filter-type", target if target in ("responsive", "flat", "oscillating") else "responsive")
    chosen = page.input_value("#archive-filter-type")
    page.wait_for_function("!document.getElementById('archive-alg-scope').innerText.includes('All runs')", timeout=30000)
    labels = {c for c, _, _ in _table_rows(page)}
    expected = {"responsive": "Responsive", "flat": "Flat", "oscillating": "Oscillating"}[chosen]
    assert all(expected in label for label in labels), labels
    page.close()


def test_the_id_search_narrows_the_comparison_and_clearing_restores_it(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)
    ids = _api(archive, "/api/history?set=everything&page_size=50")["history"]
    wanted = ids[0]["run_id"]
    _search(page, wanted)
    _wait_scope(page, f"run id containing “{wanted}”")
    matching = _api(archive, "/api/history?set=everything&page_size=50&q=" + wanted)["view"]["total"]
    assert f"the {matching} run" in _scope(page) or f"the {matching} runs" in _scope(page) or matching == 1
    assert len(_table_rows(page)) == _expected_rows(archive, "everything", wanted)
    assert all(n <= matching for _, _, n in _table_rows(page))
    _search(page, "")
    page.wait_for_function("!document.getElementById('archive-alg-scope').innerText.includes('containing')", timeout=30000)
    assert len(_table_rows(page)) == _expected_rows(archive, "everything")
    page.close()


def test_a_search_without_matches_empties_the_comparison_and_says_so(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)
    _search(page, "zzz-not-a-run")
    _wait_scope(page, "nothing to compare")
    assert not _table_rows(page)
    page.close()


def test_a_slow_earlier_comparison_cannot_overwrite_a_newer_one(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _open_comparison(page)

    def slow_first(route):
        if "q=slow-search" in route.request.url:
            page.wait_for_timeout(1500)
        route.continue_()

    page.route("**/api/archive/algorithm_comparison*", slow_first)
    page.evaluate("() => { archiveView().q = 'slow-search'; loadAlgorithmComparison(); }")
    page.evaluate("() => { archiveView().q = ''; loadAlgorithmComparison(); }")
    page.wait_for_timeout(2500)
    assert "slow-search" not in _scope(page)
    assert len(_table_rows(page)) == _expected_rows(archive, "everything")
    page.close()


def test_the_comparison_is_current_when_opened_after_filtering(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    first = _api(archive, "/api/history?set=everything&page_size=1")["history"][0]["run_id"]
    _search(page, first)
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Rows')", timeout=30000)
    _open_comparison(page)
    _wait_scope(page, f"run id containing “{first}”")
    page.close()


# ---- the id search: what an id looks like ------------------------------------------------------------------------------
def test_the_search_box_shows_an_example_id_in_the_style_of_real_ones(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive)
    placeholder = page.get_attribute("#archive-search", "placeholder")
    assert re.search(r"iorn-\d+", placeholder), placeholder
    hint = page.inner_text("#archive-search-hint")
    assert "run id" in hint and "iorn-" in hint and "sweep-" in hint
    assert page.get_attribute("#archive-search", "aria-describedby") == "archive-search-hint"
    assert page.is_visible("#archive-search-hint")
    ids = _api(archive, "/api/history?set=everything&page_size=50")["history"]
    assert any(re.fullmatch(r"iorn-\d{6}", h["run_id"]) for h in ids)       # the example format exists in the archive
    page.close()


def test_searching_part_of_an_id_finds_it_ignoring_case(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    target = next(h["run_id"] for h in _api(archive, "/api/history?set=everything&page_size=50")["history"]
                  if re.fullmatch(r"iorn-\d{6}", h["run_id"]))
    for text in (target, target.upper(), target[5:], target[:-2]):
        _search(page, text)
        page.wait_for_function(f"archiveView().q === {json.dumps(text)} && !document.querySelector('#archive-pager-status').innerText.includes('Loading')", timeout=10000)
        page.wait_for_timeout(700)
        assert target in _run_ids(page), text
    page.close()


def test_a_missing_id_explains_itself(archive, browser):  # noqa: F811
    page = _open_archive(browser, archive, page_size=50, view="everything")
    _search(page, "nope-123")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('No run id')", timeout=30000)
    assert "nope-123" in _status(page) and "choose" not in _status(page)      # whole archive: nothing more to suggest
    page.select_option("#archive-filter-type", "flat")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('All runs')", timeout=30000)
    assert "nope-123" in _status(page) and "whole archive" in _status(page)   # a curated view: suggest the full list
    assert _run_ids(page) == []
    page.close()
