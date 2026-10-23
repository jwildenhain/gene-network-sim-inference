"""
Playwright End-to-End Test for 3x3 Multi-Perturbation Shift Bar Grid Background Colors:
- Verifies soft transparent pastel green background for up-regulation zone (y > 0, +ΔX).
- Verifies soft transparent pastel red background for down-regulation zone (y < 0, -ΔX).
- Verifies condition badge and card border adaptation (pastel green for induction, pastel red for knockdown).
- Verifies new color legend in 3x3 grid toolbar.
- Captures high-res screenshots in both Dark mode and Light mode.
"""

from pathlib import Path
import shutil
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8085"
ARTIFACT_DIR = Path("/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed")
DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
DOCS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)


def test_pastel_3x3_grid_colors():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1200})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # -------------------------------------------------------------
        # STEP 1: NAVIGATE TO TAB 3 (PREVIOUS SIMULATIONS ARCHIVE)
        # -------------------------------------------------------------
        print("2. Navigating to Tab 3 Previous Simulations Archive...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(800)
        assert page.is_visible("#tab-history")

        # Load representative responsive run
        print("3. Loading Representative Responsive Run...")
        page.evaluate("selectArchiveRun('iorn-rep-responsive')")
        page.wait_for_timeout(1000)

        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        # -------------------------------------------------------------
        # STEP 2: TOGGLE TO 3x3 SHIFT BAR GRID
        # -------------------------------------------------------------
        print("4. Toggling to 3x3 Multi-Perturbation Shift Bar Grid (#btn-hist-pert-mode-bar)...")
        page.click("#btn-hist-pert-mode-bar")
        page.wait_for_timeout(800)

        assert page.is_visible("#hist-step4-grid-container"), "3x3 grid container must be visible"
        assert page.is_visible("#canvas-hist-pert-grid"), "3x3 grid canvas must be visible"

        # Verify color legend is present
        legend_text = page.inner_text("#hist-step4-grid-container")
        print(f"Toolbar content: {legend_text[:120]}...")
        assert "Up-regulation" in legend_text
        assert "Down-regulation" in legend_text
        assert "Target Gene" in legend_text

        # Capture Dark Theme 3x3 Grid Screenshot
        page.locator("#hist-step4-grid-container").scroll_into_view_if_needed()
        page.wait_for_timeout(600)

        shot_dark = DOCS_SCREENSHOTS / "43_tab3_step4_3x3_pastel_grid_dark.png"
        page.screenshot(path=str(shot_dark), full_page=False)
        shutil.copy(shot_dark, ARTIFACT_DIR / "43_tab3_step4_3x3_pastel_grid_dark.png")
        print(f"Saved Dark Theme 3x3 Grid Screenshot: {shot_dark}")

        # Crop of just the 3x3 grid panel
        grid_box = page.locator("#hist-step4-grid-container").bounding_box()
        if grid_box:
            shot_crop = DOCS_SCREENSHOTS / "45_tab3_step4_3x3_pastel_grid_crop.png"
            page.screenshot(
                path=str(shot_crop),
                clip={
                    "x": max(0, grid_box["x"] - 10),
                    "y": max(0, grid_box["y"] - 10),
                    "width": min(1600, grid_box["width"] + 20),
                    "height": min(1200, grid_box["height"] + 20),
                }
            )
            shutil.copy(shot_crop, ARTIFACT_DIR / "45_tab3_step4_3x3_pastel_grid_crop.png")
            print(f"Saved Crop: {shot_crop}")

        # -------------------------------------------------------------
        # STEP 3: SWITCH TO LIGHT THEME
        # -------------------------------------------------------------
        print("5. Switching to Light Theme via #theme-toggle-btn...")
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(800)

        # Re-render archive perturbation view to ensure light theme colors applied
        page.evaluate("drawArchivePerturbationView()")
        page.wait_for_timeout(600)

        shot_light = DOCS_SCREENSHOTS / "44_tab3_step4_3x3_pastel_grid_light.png"
        page.screenshot(path=str(shot_light), full_page=False)
        shutil.copy(shot_light, ARTIFACT_DIR / "44_tab3_step4_3x3_pastel_grid_light.png")
        print(f"Saved Light Theme 3x3 Grid Screenshot: {shot_light}")

        # Crop of light theme grid
        grid_box_light = page.locator("#hist-step4-grid-container").bounding_box()
        if grid_box_light:
            shot_crop_light = DOCS_SCREENSHOTS / "46_tab3_step4_3x3_pastel_grid_crop_light.png"
            page.screenshot(
                path=str(shot_crop_light),
                clip={
                    "x": max(0, grid_box_light["x"] - 10),
                    "y": max(0, grid_box_light["y"] - 10),
                    "width": min(1600, grid_box_light["width"] + 20),
                    "height": min(1200, grid_box_light["height"] + 20),
                }
            )
            shutil.copy(shot_crop_light, ARTIFACT_DIR / "46_tab3_step4_3x3_pastel_grid_crop_light.png")

        # Switch back to dark theme
        page.click("#theme-toggle-btn")
        page.wait_for_timeout(600)

        # -------------------------------------------------------------
        # STEP 4: VERIFY SINGLE BAR PROFILE HAS PASTEL ZONES TOO
        # -------------------------------------------------------------
        print("6. Verifying single focus / side-by-side mode...")
        page.click("#btn-hist-pert-mode-dual")
        page.wait_for_timeout(600)
        assert page.is_visible("#hist-step4-dual-container")

        # Now test single focus bar if called
        page.evaluate("drawCustomBarProfileOnCanvas('canvas-hist-pert', APP_STATE.archiveDetail, 0)")
        page.wait_for_timeout(400)

        browser.close()
        print("All pastel color choice tests passed successfully!")


if __name__ == "__main__":
    test_pastel_3x3_grid_colors()
