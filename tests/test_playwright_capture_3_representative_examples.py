"""
Playwright script to load and capture screenshots of the 3 Representative Examples in Tab 4:
1. Responsive: Clear Level Changes (iorn-rep-responsive)
2. Flat: Flat Levels / Insensitive (iorn-rep-flat)
3. Oscillating: Persistent Limit Cycle Oscillations (iorn-rep-oscillating)
"""

import sys
import shutil
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8085"
ARTIFACT_DIR = Path("/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed")
DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
DOCS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)


def capture_3_representative_examples():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1150})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        print("2. Navigating to Tab 4 Previous Simulations Archive...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(800)

        # -------------------------------------------------------------
        # EXAMPLE 1: REPRESENTATIVE RESPONSIVE (CLEAR LEVEL CHANGES)
        # -------------------------------------------------------------
        print("3. Loading Representative Responsive run (iorn-rep-responsive)...")
        page.evaluate("selectArchiveRun('iorn-rep-responsive')")
        page.wait_for_timeout(1000)

        # Ensure in timecourse mode
        page.click("#btn-hist-pert-mode-timecourse")
        page.wait_for_timeout(500)

        # Scroll into view of step 4
        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        banner_text_resp = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Responsive Banner:\n{banner_text_resp.strip()}\n")

        # Capture screenshots
        shot1_path = DOCS_SCREENSHOTS / "34_representative_responsive_timecourse.png"
        page.screenshot(path=str(shot1_path))
        shutil.copy(shot1_path, ARTIFACT_DIR / "34_representative_responsive_timecourse.png")
        shutil.copy(shot1_path, ARTIFACT_DIR / "representative_responsive_timecourse.png")
        print(f"Saved: {shot1_path}")

        # -------------------------------------------------------------
        # EXAMPLE 2: REPRESENTATIVE FLAT (DAMPED / INSENSITIVE)
        # -------------------------------------------------------------
        print("4. Loading Representative Flat run (iorn-rep-flat)...")
        page.evaluate("selectArchiveRun('iorn-rep-flat')")
        page.wait_for_timeout(1000)

        page.click("#btn-hist-pert-mode-timecourse")
        page.wait_for_timeout(500)
        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        banner_text_flat = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Flat Banner:\n{banner_text_flat.strip()}\n")

        shot2_path = DOCS_SCREENSHOTS / "35_representative_flat_timecourse.png"
        page.screenshot(path=str(shot2_path))
        shutil.copy(shot2_path, ARTIFACT_DIR / "35_representative_flat_timecourse.png")
        shutil.copy(shot2_path, ARTIFACT_DIR / "representative_flat_timecourse.png")
        print(f"Saved: {shot2_path}")

        # -------------------------------------------------------------
        # EXAMPLE 3: REPRESENTATIVE OSCILLATING (LIMIT CYCLE)
        # -------------------------------------------------------------
        print("5. Loading Representative Oscillating run (iorn-rep-oscillating)...")
        page.evaluate("selectArchiveRun('iorn-rep-oscillating')")
        page.wait_for_timeout(1000)

        page.click("#btn-hist-pert-mode-timecourse")
        page.wait_for_timeout(500)
        page.locator("#panel-hist-step4").scroll_into_view_if_needed()
        page.wait_for_timeout(500)

        banner_text_osc = page.inner_text("#hist-trajectory-diagnosis-banner")
        print(f"Oscillating Banner:\n{banner_text_osc.strip()}\n")

        shot3_path = DOCS_SCREENSHOTS / "36_representative_oscillating_timecourse.png"
        page.screenshot(path=str(shot3_path))
        shutil.copy(shot3_path, ARTIFACT_DIR / "36_representative_oscillating_timecourse.png")
        shutil.copy(shot3_path, ARTIFACT_DIR / "representative_oscillating_timecourse.png")
        print(f"Saved: {shot3_path}")

        browser.close()
        print("All 3 representative screenshots captured successfully!")


if __name__ == "__main__":
    capture_3_representative_examples()
