"""
Playwright End-to-End Test for Tab 4 (Previous Simulations Archive) Step 4 Panel:
- Perturbation ODE Time-Course vs Expression Matrix (X) vs Shift Bar Profile (ΔX) toggle.
- Condition dropdown switching.
- Data review for clear level changes (responsive), flat (flat), and oscillations (oscillating).
"""

from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8085"
ARTIFACT_DIR = Path("/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed")
DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def test_archive_step4_toggle_and_trajectory_review():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # Navigate to Tab 4 History
        print("2. Navigating to Tab 4 Previous Simulations Archive...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(800)
        assert page.is_visible("#tab-history")

        # -------------------------------------------------------------
        # TEST 1: RESPONSIVE RUN (CLEAR LEVEL CHANGES)
        # -------------------------------------------------------------
        print("3. Selecting a Responsive Run (Clear Level Changes)...")
        page.select_option("#archive-filter-type", "responsive")
        page.wait_for_timeout(600)

        first_row_btn = page.query_selector("#tbl-history tbody tr button")
        if first_row_btn:
            first_row_btn.click()
        else:
            page.click("#tbl-history tbody tr")
        page.wait_for_timeout(1000)

        # Verify Step 4 panel is visible
        assert page.is_visible("#panel-hist-step4"), "Step 4 panel must be visible in archive detail inspector"
        assert page.is_visible("#canvas-hist-pert"), "Step 4 canvas must be visible"

        # Check toggle buttons
        assert page.is_visible("#btn-hist-pert-mode-timecourse")
        assert page.is_visible("#btn-hist-pert-mode-matrix")
        assert page.is_visible("#btn-hist-pert-mode-bar")

        # By default, timecourse should be active
        assert "active" in page.get_attribute("#btn-hist-pert-mode-timecourse", "class")

        # Check trajectory diagnosis banner for clear level changes
        diag_text = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Responsive Diagnosis: {diag_text}")
        assert "Clear Level Changes" in diag_text or "Responsive" in diag_text

        # Capture screenshot of time-course
        page.screenshot(path=str(ARTIFACT_DIR / "tab4_step4_timecourse_responsive.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "30_tab4_step4_timecourse_responsive.png"))

        # -------------------------------------------------------------
        # TEST 2: TOGGLE TO EXPRESSION MATRIX (X)
        # -------------------------------------------------------------
        print("4. Toggling to Expression Matrix (X)...")
        page.click("#btn-hist-pert-mode-matrix")
        page.wait_for_timeout(400)
        assert "active" in page.get_attribute("#btn-hist-pert-mode-matrix", "class")
        assert "active" not in page.get_attribute("#btn-hist-pert-mode-timecourse", "class")
        # Condition dropdown should be hidden in matrix view
        assert not page.is_visible("#select-hist-pert-gene")

        # Capture screenshot of expression matrix heatmap
        page.screenshot(path=str(ARTIFACT_DIR / "tab4_step4_expression_matrix.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "31_tab4_step4_expression_matrix.png"))

        # -------------------------------------------------------------
        # TEST 3: TOGGLE TO SHIFT BAR PROFILE (ΔX)
        # -------------------------------------------------------------
        print("5. Toggling to Shift Bar Profile (ΔX)...")
        page.click("#btn-hist-pert-mode-bar")
        page.wait_for_timeout(400)
        assert "active" in page.get_attribute("#btn-hist-pert-mode-bar", "class")
        assert page.is_visible("#select-hist-pert-gene")

        # -------------------------------------------------------------
        # TEST 4: FLAT DYNAMICS RUN
        # -------------------------------------------------------------
        print("6. Selecting a Flat Dynamics Run (No Visible Perturbation)...")
        page.select_option("#archive-filter-type", "flat")
        page.wait_for_timeout(600)
        page.click("#tbl-history tbody tr:first-child button:first-child")
        page.wait_for_timeout(1000)

        # Switch to timecourse
        page.click("#btn-hist-pert-mode-timecourse")
        page.wait_for_timeout(400)

        diag_text_flat = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Flat Diagnosis: {diag_text_flat}")
        assert "Flat" in diag_text_flat

        page.screenshot(path=str(ARTIFACT_DIR / "tab4_step4_timecourse_flat.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "32_tab4_step4_timecourse_flat.png"))

        # -------------------------------------------------------------
        # TEST 5: OSCILLATING DYNAMICS RUN (LIMIT CYCLE)
        # -------------------------------------------------------------
        print("7. Selecting an Oscillating Run (Limit Cycle Dynamics)...")
        page.select_option("#archive-filter-type", "oscillating")
        page.wait_for_timeout(600)
        page.click("#tbl-history tbody tr:first-child button:first-child")
        page.wait_for_timeout(1000)

        diag_text_osc = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Oscillating Diagnosis: {diag_text_osc}")
        assert "Oscillat" in diag_text_osc

        page.screenshot(path=str(ARTIFACT_DIR / "tab4_step4_timecourse_oscillating.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "33_tab4_step4_timecourse_oscillating.png"))

        browser.close()
        print("All Tab 4 Step 4 toggle & trajectory review tests passed successfully!")


if __name__ == "__main__":
    test_archive_step4_toggle_and_trajectory_review()
