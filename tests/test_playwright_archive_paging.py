"""The archive tab loads one page of a ranked view at a time instead of every run with its full detail."""
import json
import urllib.request

import pytest

pytest.importorskip("playwright.sync_api")


def _api(app_url, path):
    with urllib.request.urlopen(app_url + path, timeout=120) as r:
        return json.loads(r.read())


@pytest.fixture(scope="module")
def archive(app_url):
    """An archive with enough runs (12 genes: the curated views only list networks of 12 or more) for several pages of three."""
    for seed in range(21, 27):
        req = urllib.request.Request(app_url + "/api/simulate", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"num_genes": 12, "seed": seed, "topology": "random", "auto_infer": True,
                                                      "algorithms": ["s2g", "vote"]}).encode())
        urllib.request.urlopen(req, timeout=120).read()
    return app_url


def _open_archive(browser, app_url, page_size=3, view="everything"):
    page = browser.new_page(viewport={"width": 1400, "height": 1000})
    page.goto(app_url, wait_until="networkidle")
    page.evaluate("switchMainTab('tab-history')")
    page.wait_for_selector("#tbl-history tbody tr", state="attached", timeout=60000)
    page.evaluate(f"archiveView().pageSize = {page_size}; document.getElementById('archive-filter-type').value = '{view}'; loadHistoryTable();")
    page.wait_for_function(f"document.querySelector('#archive-pager-status').innerText.includes('Page 1 of') && document.querySelectorAll('#tbl-history tbody tr').length <= {page_size}", timeout=30000)
    return page


def _status(page):
    return page.inner_text("#archive-pager-status")


def _run_ids(page):
    return page.eval_on_selector_all("#tbl-history tbody tr code", "els => els.map(e => e.innerText.trim())")


def test_the_tab_asks_for_one_small_page_not_every_run(archive, browser):
    requests, sizes = [], []
    page = browser.new_page()
    page.on("request", lambda r: requests.append(r.url) if "/api/history" in r.url else None)
    page.on("response", lambda r: sizes.append(len(r.body())) if "/api/history?" in r.url and r.ok else None)
    page.goto(archive, wait_until="networkidle")
    page.evaluate("switchMainTab('tab-history')")
    page.wait_for_selector("#tbl-history tbody tr", state="attached", timeout=60000)
    page.wait_for_timeout(500)
    paged = [u for u in requests if "/api/history?" in u]
    assert paged and "set=all" in paged[0] and "page_size=50" in paged[0]
    assert not [u for u in requests if u.rstrip("/").endswith("/api/history")]               # never the full-detail form
    assert max(sizes) < 60_000                                                                # a page, not megabytes
    page.close()


def test_pager_buttons_walk_through_the_pages(archive, browser):
    page = _open_archive(browser, archive, page_size=2)
    total = _api(archive, "/api/history?set=everything&page_size=1")["view"]["total"]
    pages = -(-total // 2)
    assert pages >= 3 and f"of {total}" in _status(page) and f"Page 1 of {pages}" in _status(page)
    assert page.locator("button[aria-label='First page']").is_disabled() and page.locator("button[aria-label='Previous page']").is_disabled()
    first = _run_ids(page)
    page.click("button[aria-label='Next page']")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 2')")
    second = _run_ids(page)
    assert second != first
    assert second == [r["run_id"] for r in _api(archive, "/api/history?set=everything&page=2&page_size=2")["history"]]
    page.click("button[aria-label='Last page']")
    page.wait_for_function(f"document.querySelector('#archive-pager-status').innerText.includes('Page {pages} of {pages}')")
    assert page.locator("button[aria-label='Next page']").is_disabled() and page.locator("button[aria-label='Last page']").is_disabled()
    assert 1 <= len(_run_ids(page)) <= 2
    page.click("button[aria-label='First page']")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 1 of')")
    assert _run_ids(page) == first
    page.close()


def test_sorting_covers_every_page_not_just_the_visible_one(archive, browser):
    page = _open_archive(browser, archive)
    header = page.locator("#tbl-history thead th", has_text="Total Errors")
    header.click()
    page.wait_for_function("document.querySelector('#tbl-history thead th[aria-sort]') !== null")
    page.wait_for_timeout(600)
    expect_asc = [r["run_id"] for r in _api(archive, "/api/history?set=everything&sort=errors&order=asc&page_size=3")["history"]]
    assert _run_ids(page) == expect_asc and header.get_attribute("aria-sort") == "ascending"
    header.click()
    page.wait_for_function("document.querySelector('#tbl-history thead th[aria-sort=descending]') !== null")
    page.wait_for_timeout(600)
    expect_desc = [r["run_id"] for r in _api(archive, "/api/history?set=everything&sort=errors&order=desc&page_size=3")["history"]]
    assert _run_ids(page) == expect_desc
    errs = [float(t) for t in page.eval_on_selector_all("#tbl-history tbody tr td:nth-child(13)", "els => els.map(e => e.innerText.trim())")]
    assert errs == sorted(errs, reverse=True)
    page.close()


def test_search_narrows_the_whole_archive_and_clearing_restores_it(archive, browser):
    page = _open_archive(browser, archive)
    target = _run_ids(page)[0]
    page.fill("#archive-search", target)
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('of 1 ')", timeout=15000)
    assert _run_ids(page) == [target] and "Page 1 of 1" in _status(page)
    page.fill("#archive-search", "no-such-run")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('No run id')", timeout=15000)
    assert "no-such-run" in _status(page) and page.locator("#tbl-history tbody tr").count() == 0
    page.fill("#archive-search", "")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 1 of') && !document.querySelector('#archive-pager-status').innerText.includes('No run')", timeout=15000)
    assert len(_run_ids(page)) == 3
    page.close()


def test_changing_the_view_returns_to_the_first_page(archive, browser):
    page = _open_archive(browser, archive)
    page.click("button[aria-label='Next page']")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 2')")
    page.select_option("#archive-filter-type", "responsive")
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 1 of')", timeout=15000)
    assert "Best 15 Responsive" in page.inner_text("#archive-best15-banner-text")
    assert page.inner_text("#archive-filter-count-badge").endswith("Curated Runs")
    page.close()


def test_a_slow_earlier_page_does_not_overwrite_the_page_asked_for_later(archive, browser):
    page = browser.new_page()
    page.add_init_script("""(() => { const nf = window.fetch.bind(window);
        window.fetch = (input, init) => { const url = typeof input === 'string' ? input : (input && input.url) || '';
            const p = nf(input, init); return url.includes('/api/history?') && url.includes('page=1&') && url.includes('set=everything')
              ? p.then(r => new Promise(res => setTimeout(() => res(r), 2500))) : p; }; })();""")
    page.goto(archive, wait_until="networkidle")
    page.evaluate("switchMainTab('tab-history')")
    page.wait_for_selector("#tbl-history tbody tr", state="attached", timeout=60000)
    # fire and forget (evaluate would otherwise wait for the returned promise and the two requests would never overlap)
    page.evaluate("() => { archiveView().pageSize = 3; document.getElementById('archive-filter-type').value = 'everything'; archiveView().page = 1; loadHistoryTable(); }")
    page.wait_for_timeout(200)
    page.evaluate("() => { archiveView().page = 2; loadHistoryTable(); }")                       # asked for later, answered first
    page.wait_for_function("document.querySelector('#archive-pager-status').innerText.includes('Page 2')", timeout=15000)
    page.wait_for_timeout(3200)                                                                   # the slow page-1 answer arrives now
    assert "Page 2" in _status(page)
    assert _run_ids(page) == [r["run_id"] for r in _api(archive, "/api/history?set=everything&page=2&page_size=3")["history"]]
    page.close()


def test_the_pager_and_search_are_labelled_for_assistive_technology(archive, browser):
    page = _open_archive(browser, archive)
    info = page.evaluate("""() => ({
        nav: document.getElementById('archive-pager').getAttribute('aria-label'),
        navRole: document.getElementById('archive-pager').getAttribute('role'),
        live: document.getElementById('archive-pager-status').getAttribute('aria-live'),
        buttons: [...document.querySelectorAll('#archive-pager button')].map(b => b.getAttribute('aria-label')),
        sizeLabel: document.querySelector('label.archive-pager-size').innerText.trim().split('\\n')[0],
        search: document.querySelector('label[for=archive-search]').innerText })""")
    assert info["nav"] == "Archive pages" and info["navRole"] == "navigation" and info["live"] == "polite"
    assert info["buttons"] == ["First page", "Previous page", "Next page", "Last page"]
    assert info["sizeLabel"].startswith("Rows per page") and info["search"] == "Search runs by id"
    page.close()
