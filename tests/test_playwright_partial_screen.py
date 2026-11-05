"""Studio panel: partial Perturb-seq screens (perturbed genes, selection) and their own statistics categories."""

import os

import pytest
from playwright.sync_api import sync_playwright

from _live import APP_URL, ARTIFACT_DIR

pytestmark = pytest.mark.live_server


def test_partial_screen_fields_query_and_design_statistics():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1500, "height": 2200}).new_page()
        errors = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(APP_URL, wait_until="networkidle")
        # a small current network, so the screens below take seconds whatever the studio simulated last
        reply = page.evaluate("""fetch('/api/simulate', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({num_genes: 30, topology: 'modular', k_av: 2.5, seed: 3, auto_infer: false})}).then(r => r.json())""")
        assert reply.get("status") == "success", reply
        page.evaluate("id => { APP_STATE.sim = Object.assign(APP_STATE.sim || {}, {run_id: id}); }", reply["run_id"])
        page.locator('button[data-tab="tab-vcc"]').click()
        page.wait_for_timeout(1200)

        assert page.locator("#ps-perturbed").is_visible() and page.locator("#ps-selection").is_visible()
        assert page.input_value("#ps-perturbed") == "0" and page.input_value("#ps-selection") == "effective"
        query = page.evaluate("perturbSeqQuery()")
        assert "perturbed=0" in query and "selection=effective" in query

        page.select_option("#ps-preset", "replogle_like")
        page.fill("#ps-cells", "15")

        # full screen first: no partial-screen block, every row scored
        page.click("#btn-ps-realism")
        page.wait_for_selector("#ps-realism-table", timeout=240000)
        assert page.locator("#ps-design").count() == 0
        assert page.locator("#ps-realism-table td", has_text="n/a").count() == 0

        # partial screen with effective selection
        page.fill("#ps-perturbed", "5")
        assert "perturbed=5" in page.evaluate("perturbSeqQuery()")
        page.click("#btn-ps-realism")
        page.wait_for_selector("#ps-design", timeout=240000)
        design = page.inner_text("#ps-design")
        assert "Partial screen: 5 of" in design and "effective" in design
        cats = page.eval_on_selector_all(".ps-design-table", "els => els.map(e => e.dataset.category)")
        assert cats == ["design", "identifiability", "coverage", "blocks"]
        assert page.locator("#ps-realism-table td", has_text="n/a").count() >= 3      # selection-biased statistics are not scored
        page.screenshot(path=os.path.join(ARTIFACT_DIR, "partial_screen_panel.png"), full_page=False)

        # random selection keeps every statistic comparable
        page.select_option("#ps-selection", "random")
        page.click("#btn-ps-realism")
        page.wait_for_function("document.querySelector('#ps-design') && document.querySelector('#ps-design').innerText.includes('(random)')",
                               timeout=240000)
        assert page.locator("#ps-realism-table td", has_text="n/a").count() == 0

        # the algorithm list says which algorithms need the partial-screen adapter
        flags = page.evaluate("fetch('/api/algorithms').then(r => r.json()).then(d => d.algorithms.map(a => [a.key, a.partial]))")
        assert dict(flags)["s2g"] == "adapter" and dict(flags)["inspre"] == "native"
        assert not [e for e in errors if "favicon" not in e], errors
        browser.close()
