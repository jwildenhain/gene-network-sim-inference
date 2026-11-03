"""Slow API requests show a progress bar with what is loading; quick ones and background polling do not."""

DELAY_MS = 1500


def _delay_init_script(pattern: str, ms: int) -> str:
    # runs before the page's own script, so the page's loading-bar wrapper sits on top of this delayed fetch
    return f"""
    (() => {{
      const nativeFetch = window.fetch.bind(window);
      window.fetch = (input, init) => {{
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const p = nativeFetch(input, init);
        return url.includes({pattern!r}) ? p.then(r => new Promise(res => setTimeout(() => res(r), {ms}))) : p;
      }};
    }})();
    """


def test_slow_request_shows_the_loading_bar_and_clears_it(app_url, browser):
    page = browser.new_page()
    page.add_init_script(_delay_init_script("/api/benchmark/vcc_atlasshift", DELAY_MS))
    page.goto(app_url, wait_until="networkidle")
    assert not page.locator("#global-load").is_visible()                       # idle: nothing shown
    page.evaluate("switchMainTab('tab-vcc'); loadVccBenchmarkData();")
    page.wait_for_selector("#global-load.on", timeout=3000)                     # appears after ~0.3 s
    msg = page.locator("#global-load-msg").inner_text()
    assert "Virtual Cell Challenge" in msg and " s" in msg, msg
    assert page.get_attribute("#global-load", "aria-busy") == "true"
    page.wait_for_selector("#global-load.on", state="detached", timeout=60000)  # gone when the data arrived
    assert page.get_attribute("#global-load", "aria-busy") == "false"
    assert page.locator("#tbl-vcc-summary tbody tr").count() > 2
    page.close()


def test_quick_and_background_requests_do_not_flash_the_bar(app_url, browser):
    page = browser.new_page()
    page.goto(app_url, wait_until="networkidle")
    page.evaluate("window.__barSeen = false; new MutationObserver(() => { if (document.getElementById('global-load').classList.contains('on')) window.__barSeen = true; }).observe(document.getElementById('global-load'), {attributes: true});")
    page.evaluate("fetch('/api/status').then(r => r.json())")                   # quick and polled: never shown
    page.evaluate("fetch('/api/jobs').then(r => r.json())")
    page.wait_for_timeout(800)
    assert page.evaluate("window.__barSeen") is False
    page.close()
