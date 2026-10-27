"""
Playwright End-to-End Test for Tab 4 Step 4:
- Side-by-Side dual graph view: Perturbation ODE Time-Course (left) & Expression Matrix Heatmap (right).
- Toggle to Shift Bar Profile: 3x3 multi-perturbation grid showing up to 9 perturbation response profiles simultaneously.
- Pagination across 3x3 grid for networks with N > 9.
- Captures high-resolution screenshots for responsive, flat, and oscillating runs.
"""

import shutil
import pytest
from playwright.sync_api import sync_playwright

from _live import APP_URL as BASE_URL
from _live import ARTIFACT_DIR

pytestmark = pytest.mark.live_server
from _live import DOCS_SCREENSHOTS


def test_archive_side_by_side_and_3x3_grid():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1200})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # Navigate to Tab 4 History
        print("2. Navigating to Tab 4 Previous Simulations Archive...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(800)
        assert page.is_visible("#tab-history")

        # -------------------------------------------------------------
        # TEST 1: DEFAULT DUAL SIDE-BY-SIDE VIEW (RESPONSIVE RUN)
        # -------------------------------------------------------------
        print("3. Loading Representative Responsive Run...")
        page.select_option("#archive-filter-type", "responsive")
        page.wait_for_timeout(600)
        first_row_btn = page.query_selector("#tbl-history tbody tr button")
        if first_row_btn:
            first_row_btn.click()
        else:
            page.click("#tbl-history tbody tr")
        page.wait_for_timeout(1000)

        # Scroll into view of step 4
        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(600)

        # Verify dual container is visible
        assert page.is_visible("#hist-step4-dual-container"), "Dual container must be visible by default"
        assert page.is_visible("#canvas-hist-pert-timecourse"), "Time-course canvas must be visible on left"
        assert page.is_visible("#canvas-hist-pert-matrix"), "Expression matrix canvas must be visible on right"

        # Verify grid container is initially hidden
        assert not page.is_visible("#hist-step4-grid-container"), "3x3 grid container must be hidden in dual mode"

        # Check target label and switch condition
        dual_label = page.inner_text("#hist-dual-target-label")
        print(f"Initial Dual Target Label: {dual_label}")
        assert "G1" in dual_label

        page.select_option("#select-hist-pert-gene", "1")  # Target G2
        page.wait_for_timeout(500)
        dual_label_g2 = page.inner_text("#hist-dual-target-label")
        print(f"Updated Dual Target Label: {dual_label_g2}")
        assert "G2" in dual_label_g2

        # Reset to G1
        page.select_option("#select-hist-pert-gene", "0")
        page.wait_for_timeout(400)

        # Capture Screenshot: Dual Side-by-Side (Responsive)
        shot1 = DOCS_SCREENSHOTS / "37_tab4_step4_dual_side_by_side_responsive.png"
        page.screenshot(path=str(shot1))
        shutil.copy(shot1, ARTIFACT_DIR / "37_tab4_step4_dual_side_by_side_responsive.png")
        print(f"Saved Screenshot 1: {shot1}")

        # -------------------------------------------------------------
        # TEST 2: TOGGLE TO 3x3 MULTI-PERTURBATION SHIFT BAR GRID
        # -------------------------------------------------------------
        print("4. Toggling to 3x3 Multi-Perturbation Shift Bar Grid (#btn-hist-pert-mode-bar)...")
        page.click("#btn-hist-pert-mode-bar")
        page.wait_for_timeout(800)

        # Verify 3x3 grid container is now visible and dual container is hidden
        assert page.is_visible("#hist-step4-grid-container"), "3x3 grid container must be visible after toggle"
        assert not page.is_visible("#hist-step4-dual-container"), "Dual container must be hidden in grid mode"
        assert page.is_visible("#canvas-hist-pert-grid"), "3x3 grid canvas must be visible"

        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        page_label = page.inner_text("#hist-grid-page-label")
        print(f"3x3 Grid Page Label: {page_label}")
        assert "Conditions 1–9" in page_label

        # Capture Screenshot: 3x3 Multi-Perturbation Grid
        shot2 = DOCS_SCREENSHOTS / "38_tab4_step4_3x3_shift_bar_grid.png"
        page.screenshot(path=str(shot2))
        shutil.copy(shot2, ARTIFACT_DIR / "38_tab4_step4_3x3_shift_bar_grid.png")
        print(f"Saved Screenshot 2: {shot2}")

        # Test pagination (iorn-rep-responsive has N=10, so 2 pages)
        print("Testing grid pagination to page 2...")
        page.click("#btn-hist-grid-next")
        page.wait_for_timeout(500)
        page_label_p2 = page.inner_text("#hist-grid-page-label")
        print(f"Page 2 Label: {page_label_p2}")
        assert "10" in page_label_p2

        # Return to page 1
        page.click("#btn-hist-grid-prev")
        page.wait_for_timeout(400)

        # -------------------------------------------------------------
        # TEST 3: TOGGLE BACK TO SIDE-BY-SIDE DUAL VIEW
        # -------------------------------------------------------------
        print("5. Toggling back to Side-by-Side Dual View (#btn-hist-pert-mode-dual)...")
        page.click("#btn-hist-pert-mode-dual")
        page.wait_for_timeout(600)

        assert page.is_visible("#hist-step4-dual-container")
        assert not page.is_visible("#hist-step4-grid-container")

        # -------------------------------------------------------------
        # TEST 4: DUAL VIEW FOR FLAT RUN
        # -------------------------------------------------------------
        print("6. Testing Dual View on Representative Flat Run...")
        page.select_option("#archive-filter-type", "flat")
        page.wait_for_timeout(600)
        first_row_btn = page.query_selector("#tbl-history tbody tr button")
        if first_row_btn:
            first_row_btn.click()
        else:
            page.click("#tbl-history tbody tr")
        page.wait_for_timeout(1000)

        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        shot3 = DOCS_SCREENSHOTS / "39_tab4_step4_dual_side_by_side_flat.png"
        page.screenshot(path=str(shot3))
        shutil.copy(shot3, ARTIFACT_DIR / "39_tab4_step4_dual_side_by_side_flat.png")
        print(f"Saved Screenshot 3: {shot3}")

        # -------------------------------------------------------------
        # TEST 5: DUAL VIEW FOR OSCILLATING RUN
        # -------------------------------------------------------------
        print("7. Testing Dual View on Representative Oscillating Run...")
        page.select_option("#archive-filter-type", "oscillating")
        page.wait_for_timeout(600)
        first_row_btn = page.query_selector("#tbl-history tbody tr button")
        if first_row_btn:
            first_row_btn.click()
        else:
            page.click("#tbl-history tbody tr")
        page.wait_for_timeout(1000)

        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        shot4 = DOCS_SCREENSHOTS / "40_tab4_step4_dual_side_by_side_oscillating.png"
        page.screenshot(path=str(shot4))
        shutil.copy(shot4, ARTIFACT_DIR / "40_tab4_step4_dual_side_by_side_oscillating.png")
        print(f"Saved Screenshot 4: {shot4}")

        browser.close()
        print("All Tab 4 Step 4 Side-by-Side & 3x3 Grid tests passed with flying colors!")


if __name__ == "__main__":
    test_archive_side_by_side_and_3x3_grid()
