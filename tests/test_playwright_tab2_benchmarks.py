"""
End-to-End Playwright Test for Tab 2 Real-World Benchmarks Studio:
1. Availability and usage review of SOS, DREAM4, Norman, and Replogle datasets.
2. Raw dataset download dropdown (TSV, JSON, ZIP bundle).
3. Rich provenance and experimental provenance card with biological mechanism.
4. Dual loading distinction into Tab 1: Wrangled Direct Inference vs. Biophysical Parameter Estimation.
"""

import pytest
from playwright.sync_api import sync_playwright

from _live import APP_URL as BASE_URL
from _live import ARTIFACT_DIR

pytestmark = pytest.mark.live_server
from _live import DOCS_SCREENSHOTS


def test_tab2_benchmarks_provenance_and_downloads():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1050})
        page = context.new_page()

        print("1. Loading phpIoRN application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # 2. Switch to Tab 2
        print("2. Switching to Tab 2 Real-World Benchmarks Studio...")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(1000)
        assert page.is_visible("#tab-sos")

        # 3. Check Dual Action Load Buttons
        assert page.is_visible("#btn-load-wrangled"), "Load Wrangled button should be visible"
        assert page.is_visible("#btn-load-biophysical"), "Fit ODE Parameters button should be visible"
        assert page.is_visible("#btn-download-raw-menu"), "Download Raw Data button should be visible"

        # 4. Check Raw Download Dropdown
        page.click("#btn-download-raw-menu")
        page.wait_for_timeout(400)
        assert page.is_visible("#raw-download-dropdown"), "Raw download dropdown should be open"
        dropdown_text = page.inner_text("#raw-download-dropdown")
        assert "Expression Matrix X" in dropdown_text
        assert "Perturbation Matrix U" in dropdown_text
        assert "Ground Truth Adjacency A" in dropdown_text
        assert "Complete Dataset Package (.zip)" in dropdown_text
        assert "Full JSON Bundle" in dropdown_text

        # Capture screenshot with dropdown open
        page.screenshot(path=str(ARTIFACT_DIR / "tab2_raw_download_menu_open.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "17_tab2_raw_download_menu_open.png"))

        # Close dropdown by clicking title
        page.click("#benchmark-header-title")
        page.wait_for_timeout(300)

        # 5. Verify E. coli SOS Provenance Card
        print("3. Verifying E. coli SOS dataset provenance...")
        title = page.inner_text("#prov-card-title")
        assert "E. coli SOS" in title
        organism = page.inner_text("#prov-organism")
        assert "Escherichia coli" in organism
        tech = page.inner_text("#prov-technology")
        assert "microarray" in tech.lower()
        scale = page.inner_text("#prov-scale")
        assert "9 Genes" in scale and "True Edges" in scale
        wrangled_desc = page.inner_text("#prov-usage-wrangled")
        assert "X" in wrangled_desc and "U" in wrangled_desc
        bio_desc = page.inner_text("#prov-usage-biophysical")
        assert "basal synthesis" in bio_desc or "Hill" in bio_desc

        # 6. Switch to Norman 2019 Real CRISPRa (16 TFs)
        print("4. Switching to Norman 2019 Real CRISPRa (16 TFs)...")
        page.click("#pill-ds-norman_2019_real_16tf")
        page.wait_for_function("document.getElementById('prov-card-title').textContent.includes('Norman')", timeout=10000)
        norman_title = page.inner_text("#prov-card-title")
        assert "Norman" in norman_title
        norman_org = page.inner_text("#prov-organism")
        assert "Homo sapiens" in norman_org
        norman_tech = page.inner_text("#prov-technology")
        assert "CRISPR activation" in norman_tech or "CRISPRa" in norman_tech
        norman_scale = page.inner_text("#prov-scale")
        assert "16 Genes" in norman_scale

        # Capture Norman provenance card screenshot
        page.screenshot(path=str(ARTIFACT_DIR / "tab2_norman_real_provenance_card.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "18_tab2_norman_real_provenance_card.png"))

        # 7. Switch to DREAM4 10-gene Gold Standard
        print("5. Switching to DREAM4 10-gene...")
        page.click("#pill-ds-dream4_10gene")
        page.wait_for_function("document.getElementById('prov-card-title').textContent.includes('DREAM4')", timeout=10000)
        dream_title = page.inner_text("#prov-card-title")
        assert "DREAM4" in dream_title
        dream_scale = page.inner_text("#prov-scale")
        assert "10 Genes" in dream_scale and "True Edges" in dream_scale

        # 8. Switch to Replogle 2022 p53 Circuit
        print("6. Switching to Replogle 2022 p53 circuit...")
        page.click("#pill-ds-replogle_2022_p53")
        page.wait_for_function("document.getElementById('prov-card-title').textContent.includes('Replogle')", timeout=10000)
        rep_title = page.inner_text("#prov-card-title")
        assert "Replogle" in rep_title
        rep_scale = page.inner_text("#prov-scale")
        assert "12 Genes" in rep_scale and "True Edges" in rep_scale

        # Capture Replogle p53 card screenshot
        page.screenshot(path=str(ARTIFACT_DIR / "tab2_replogle_p53_provenance_card.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "19_tab2_replogle_p53_provenance_card.png"))

        # 9. Test Loading into Tab 1 in Mode A: Wrangled Direct Inference
        print("7. Testing Loading into Tab 1 (Wrangled Direct)...")
        page.click("#btn-load-wrangled")
        page.wait_for_timeout(1500)
        assert page.is_visible("#tab-wizard"), "Should transition to Tab 1 Wizard Studio"
        info_text = page.inner_text("#sim-mode-info")
        assert "Real-World Wrangled Data" in info_text or "Wrangled" in info_text
        page.screenshot(path=str(ARTIFACT_DIR / "tab1_loaded_wrangled_direct.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "20_tab1_loaded_wrangled_direct.png"))

        # 10. Return to Tab 2 and Test Loading in Mode B: Biophysical Parameter Estimation
        print("8. Testing Loading into Tab 1 (Biophysical Parameter Estimation)...")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(800)
        page.click("#pill-ds-ecoli_sos")
        page.wait_for_timeout(1000)
        page.click("#btn-load-biophysical")
        page.wait_for_timeout(1500)
        assert page.is_visible("#tab-wizard"), "Should transition to Tab 1 Wizard Studio"
        bio_info_text = page.inner_text("#sim-mode-info")
        assert "Biophysical Parameter Estimation" in bio_info_text or "estimated directly" in bio_info_text
        page.screenshot(path=str(ARTIFACT_DIR / "tab1_loaded_biophysical_fit.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "21_tab1_loaded_biophysical_fit.png"))

        browser.close()
        print("✅ Tab 2 Benchmarks Studio verification test passed successfully!")


if __name__ == "__main__":
    test_tab2_benchmarks_provenance_and_downloads()
