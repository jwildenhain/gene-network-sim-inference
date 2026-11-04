"""A slow earlier request must not draw over the one the user asked for later (newest request wins)."""
import json
import urllib.request

DELAY_MS = 3000


def _delay_script(url_part: str, body_part: str, ms: int) -> str:
    """Init script: responses to requests whose URL / body contain these parts arrive ``ms`` late."""
    return f"""
    (() => {{
      const nativeFetch = window.fetch.bind(window);
      window.fetch = (input, init) => {{
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const body = init && init.body ? String(init.body) : '';
        const p = nativeFetch(input, init);
        return url.includes({url_part!r}) && body.includes({body_part!r}) ? p.then(r => new Promise(res => setTimeout(() => res(r), {ms}))) : p;
      }};
    }})();
    """


def _wait_until(page, js, timeout=60000):
    page.wait_for_function(js, timeout=timeout)


def test_a_slow_run_does_not_overwrite_the_mode_chosen_after_it(app_url, browser):
    page = browser.new_page()
    page.add_init_script(_delay_script("/api/simulate", '"mode":"sos"', DELAY_MS))
    page.goto(app_url, wait_until="networkidle")
    page.evaluate("document.getElementById('sim-mode').value = 'sos'; updateModeUI('sos', true);")           # slow (+3 s)
    page.wait_for_timeout(300)
    page.evaluate("document.getElementById('sim-mode').value = 'benchmark_norman_2019_real_16tf';"
                  "updateModeUI('benchmark_norman_2019_real_16tf', true);")                                  # asked for later
    _wait_until(page, "APP_STATE.sim && APP_STATE.sim.dataset_source === 'norman_2019_real_16tf'")
    page.wait_for_timeout(DELAY_MS + 1500)                                                                    # the SOS answer arrives now
    assert page.evaluate("APP_STATE.sim.dataset_source") == "norman_2019_real_16tf"
    assert "Norman" in page.locator("#sim-mode-info").inner_text()
    assert "Norman" in page.locator("#empirical-banner-name").inner_text()
    btn = page.locator("#btn-run-all")
    assert btn.is_enabled() and "Simulate" in btn.inner_text()                                                # not stuck, label restored
    page.close()


def test_the_busy_button_is_released_only_when_the_last_request_ends(app_url, browser):
    page = browser.new_page()
    page.add_init_script(_delay_script("/api/simulate", '"mode":"sos"', DELAY_MS))
    page.goto(app_url, wait_until="networkidle")
    page.evaluate("document.getElementById('sim-mode').value = 'sos'; updateModeUI('sos', true);")
    page.wait_for_timeout(300)
    page.evaluate("document.getElementById('sim-mode').value = 'benchmark_dream4_10gene'; updateModeUI('benchmark_dream4_10gene', true);")
    page.wait_for_timeout(DELAY_MS + 200)                                                                     # the superseded SOS run just ended
    _wait_until(page, "APP_STATE.sim && APP_STATE.sim.dataset_source === 'dream4_10gene'")
    _wait_until(page, "!document.getElementById('btn-run-all').disabled")
    assert page.evaluate("document.getElementById('btn-run-all')._busy") == 0
    page.close()


def test_opening_two_archive_runs_quickly_shows_the_second(app_url, browser):
    with urllib.request.urlopen(app_url + "/api/history", timeout=120) as r:
        runs = [h["run_id"] for h in json.loads(r.read())["history"]]
    first, second = runs[0], runs[1]
    page = browser.new_page()
    page.add_init_script(_delay_script(f"/api/history/load?run_id={first}", "", DELAY_MS))
    page.goto(app_url, wait_until="networkidle")
    page.evaluate(f"selectArchiveRun({first!r}); selectArchiveRun({second!r});")
    _wait_until(page, f"APP_STATE.archiveDetail && APP_STATE.archiveDetail.run_id === {second!r}")
    page.wait_for_timeout(DELAY_MS + 1500)                                                                    # the first answer arrives now
    assert page.evaluate("APP_STATE.archiveDetail.run_id") == second
    assert page.evaluate("APP_STATE.sim.run_id") == second
    page.close()


def test_switching_benchmark_data_sets_quickly_keeps_the_last_one(app_url, browser):
    page = browser.new_page()
    page.add_init_script(_delay_script("/api/benchmark/dataset?id=ecoli_sos", "", DELAY_MS))
    page.goto(app_url, wait_until="networkidle")
    page.evaluate("switchMainTab('tab-sos'); switchBenchmarkDataset('ecoli_sos'); switchBenchmarkDataset('dream4_10gene');")
    _wait_until(page, "APP_STATE.activeBenchmark === 'dream4_10gene'")
    page.wait_for_timeout(DELAY_MS + 2000)                                                                    # the E. coli answer arrives now
    assert page.evaluate("APP_STATE.activeBenchmark") == "dream4_10gene"
    assert page.evaluate("document.getElementById('pill-ds-dream4_10gene').classList.contains('active')")
    assert not page.evaluate("document.getElementById('pill-ds-ecoli_sos').classList.contains('active')")
    assert page.evaluate("[...document.querySelectorAll('#tab-sos [id^=pill-ds-]')].every(b => !b.disabled)")   # buttons released
    page.close()
