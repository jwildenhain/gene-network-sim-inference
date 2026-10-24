"""
Browser tests for the Virtual Cell Challenge tab of the web studio.

The page is served by the real GRNWebHandler, but every /api/** call is intercepted: the VCC endpoint is
answered with a real benchmark payload (or a crafted failure/delay) and everything else gets a stub 503,
so the tests are fast and never touch the project's SQLite database.
"""

import copy
import json
import urllib.parse

import pytest

pytest.importorskip("playwright.sync_api")

from grn_sim.benchmarks.vcc_benchmark import run_vcc_comparative_benchmark  # noqa: E402

VCC_PATH = "/api/benchmark/vcc_atlasshift"


@pytest.fixture(scope="module")
def payload_a():
    data = run_vcc_comparative_benchmark(num_genes=8, k_av=1.5, seed=42, rewire_fractions=[0.0, 0.2])
    data["summary_table"][0]["coverage_pct"] = 11.1  # marker to tell payloads apart in the DOM
    return data


@pytest.fixture(scope="module")
def payload_b(payload_a):
    data = copy.deepcopy(payload_a)
    data["summary_table"][0]["coverage_pct"] = 99.9
    return data


class VccPage:
    """A page whose VCC requests are answered by `respond(route, query)`; others get a stub 503."""

    def __init__(self, browser, app_url, respond, color_scheme="dark"):
        self.requests = []  # parsed query strings of every VCC request
        self.context = browser.new_context(viewport={"width": 1500, "height": 1000}, color_scheme=color_scheme)
        self.page = self.context.new_page()
        self.page.route("**/api/**", self._route)
        self._respond = respond
        self.page.goto(app_url, wait_until="domcontentloaded")
        self.page.click('[data-tab="tab-vcc"]')

    def _route(self, route):
        url = urllib.parse.urlparse(route.request.url)
        if url.path == VCC_PATH:
            query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            self.requests.append(query)
            self._respond(route, query)
        else:
            route.fulfill(status=503, content_type="application/json", body="{}")

    def close(self):
        self.context.close()

    def first_row_coverage(self):
        return self.page.inner_text("#tbl-vcc-summary tbody tr:first-child td:nth-child(3)")

    def wait_for_table(self):
        self.page.wait_for_selector("#tbl-vcc-summary tbody tr")


def fulfil(route, data, status=200):
    route.fulfill(status=status, content_type="application/json", body=json.dumps(data))


@pytest.fixture
def open_vcc(browser, app_url):
    opened = []

    def opener(respond, **kw):
        vp = VccPage(browser, app_url, respond, **kw)
        opened.append(vp)
        return vp

    yield opener
    for vp in opened:
        vp.close()


def test_compare_model_options_come_from_the_payload(open_vcc, payload_a):
    vp = open_vcc(lambda route, q: fulfil(route, payload_a))
    vp.page.wait_for_function("document.querySelectorAll('#vcc-model-select option').length === 8")

    options = vp.page.eval_on_selector_all(
        "#vcc-model-select option", "els => els.map(e => ({v: e.value, t: e.textContent, d: e.disabled, title: e.title}))"
    )
    assert [o["v"] for o in options] == [m["key"] for m in payload_a["parameters"]["response_models"]]
    cpa = next(o for o in options if o["v"] == "cpa_lite")
    assert cpa["d"] and cpa["t"].endswith("(n/a)") and "Jacobian" in cpa["title"]
    assert not next(o for o in options if o["v"] == "vote")["d"]
    assert vp.page.input_value("#vcc-model-select") == "vote"
    assert vp.page.inner_text("#vcc-target-model-r").startswith("r = ")


def test_unavailable_selection_falls_back_to_first_available_model(open_vcc, payload_a):
    data = copy.deepcopy(payload_a)
    for m in data["parameters"]["response_models"]:
        if m["key"] == "vote":
            m.update(available=False, reason="stubbed failure")
    vp = open_vcc(lambda route, q: fulfil(route, data))
    vp.page.wait_for_function("document.querySelectorAll('#vcc-model-select option').length === 8")
    selected = vp.page.input_value("#vcc-model-select")
    assert selected != "vote"
    assert selected in {m["key"] for m in data["parameters"]["response_models"] if m["available"]}


def test_failed_request_shows_an_error_and_re_enables_the_button(open_vcc):
    vp = open_vcc(lambda route, q: fulfil(route, {"status": "error", "error": "rewire_fraction must be in [0, 1], got 2.0"}, 400))
    vp.page.wait_for_selector("#vcc-load-error", state="visible")
    assert "rewire_fraction must be in [0, 1]" in vp.page.inner_text("#vcc-load-error")
    assert vp.page.is_enabled("#btn-run-vcc")
    assert "Re-Simulate" in vp.page.inner_text("#btn-run-vcc")


def test_error_clears_after_a_successful_reload(open_vcc, payload_a):
    state = {"fail": True}

    def respond(route, q):
        fulfil(route, {"error": "boom"}, 500) if state["fail"] else fulfil(route, payload_a)

    vp = open_vcc(respond)
    vp.page.wait_for_selector("#vcc-load-error", state="visible")
    state["fail"] = False
    vp.page.click("#btn-run-vcc")
    vp.page.wait_for_selector("#vcc-load-error", state="hidden")
    vp.wait_for_table()


def test_stale_response_cannot_overwrite_a_newer_one(open_vcc, payload_a, payload_b):
    held = []

    def respond(route, q):
        if q["rewire_fraction"] == "0.30":
            fulfil(route, payload_b)
        else:
            held.append(route)  # the default-context request stays in flight

    vp = open_vcc(respond)
    vp.page.wait_for_timeout(300)  # let the request issued by the tab click reach the router
    assert held, "the initial 15% request should be pending"

    vp.page.select_option("#vcc-cell-line", "0.30")
    vp.wait_for_table()
    assert "99.9" in vp.first_row_coverage()

    for route in held:  # the superseded request finally "returns"
        try:
            fulfil(route, payload_a)
        except Exception:
            pass  # already aborted by the page: also fine
    vp.page.wait_for_timeout(500)
    assert "99.9" in vp.first_row_coverage(), "stale payload overwrote the newer context"


def test_context_change_reuses_the_cache_but_the_button_recomputes(open_vcc, payload_a):
    vp = open_vcc(lambda route, q: fulfil(route, payload_a))
    vp.wait_for_table()
    vp.requests.clear()

    vp.page.select_option("#vcc-cell-line", "0.30")
    vp.page.wait_for_timeout(300)
    assert vp.requests[-1]["rewire_fraction"] == "0.30" and vp.requests[-1]["refresh"] == "0"

    vp.page.click("#btn-run-vcc")
    vp.page.wait_for_timeout(300)
    assert vp.requests[-1]["refresh"] == "1"


def test_theme_toggle_repaints_all_vcc_canvases(open_vcc, payload_a):
    vp = open_vcc(lambda route, q: fulfil(route, payload_a), color_scheme="dark")
    vp.wait_for_table()
    canvases = ["canvas-vcc-pca", "canvas-vcc-epistasis", "canvas-vcc-target-profile", "canvas-vcc-rewire"]
    corner = "id => Array.from(document.getElementById(id).getContext('2d').getImageData(2, 2, 1, 1).data)"

    before = {c: vp.page.evaluate(corner, c) for c in canvases}
    vp.page.click("#theme-toggle-btn")
    vp.page.wait_for_timeout(400)
    after = {c: vp.page.evaluate(corner, c) for c in canvases}

    assert before.keys() == after.keys()
    for c in canvases:
        assert before[c] != after[c], f"{c} kept its old theme colours after the toggle"


def test_purple_badges_are_styled(open_vcc, payload_a):
    vp = open_vcc(lambda route, q: fulfil(route, payload_a))
    vp.wait_for_table()
    color = vp.page.evaluate(
        "getComputedStyle(document.querySelector('#tab-vcc .badge-purple')).color"
    )
    assert color == "rgb(139, 92, 246)"
