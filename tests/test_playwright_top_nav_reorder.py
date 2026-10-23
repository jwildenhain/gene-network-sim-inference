"""
Playwright End-to-End Test for Reordered Top Navigation:
- Verifies tab order:
  1. Simulation & Inference Studio (tab-wizard)
  2. Real-World Benchmarks (tab-sos)
  3. Previous Simulations Archive (tab-history)
  4. Virtual Cell Challenge & AtlasShift (tab-vcc)
  5. Thesis & Paper Figures Gallery (tab-figures)
- Verifies navigation switching for all 5 tabs.
- Captures screenshot of reordered navigation header.
"""

from pathlib import Path
import shutil
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8085"
ARTIFACT_DIR = Path("/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed")
DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
DOCS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)


def test_top_navigation_reorder():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1000)

        # -------------------------------------------------------------
        # STEP 1: VERIFY TAB ORDER AND LABELS
        # -------------------------------------------------------------
        expected_tabs = [
            ("tab-wizard", "1. Simulation & Inference Studio"),
            ("tab-sos", "2. Real-World Benchmarks"),
            ("tab-history", "3. Previous Simulations Archive"),
            ("tab-vcc", "4. Virtual Cell Challenge & AtlasShift"),
            ("tab-figures", "5. Overall Statistics"),
        ]

        tabs = page.locator("nav.nav-tabs button.nav-tab")
        count = tabs.count()
        print(f"Total nav tabs found: {count}")
        assert count == 5, f"Expected 5 nav tabs, found {count}"

        for i, (expected_id, expected_prefix) in enumerate(expected_tabs):
            tab = tabs.nth(i)
            actual_data_tab = tab.get_attribute("data-tab")
            actual_text = tab.inner_text().strip()
            print(f"Tab {i}: data-tab='{actual_data_tab}', text='{actual_text}'")
            assert actual_data_tab == expected_id, (
                f"Tab at position {i} should be {expected_id}, found {actual_data_tab}"
            )
            assert expected_prefix in actual_text, (
                f"Tab at position {i} should contain '{expected_prefix}', found '{actual_text}'"
            )

        # -------------------------------------------------------------
        # STEP 2: TEST SWITCHING TO EACH TAB
        # -------------------------------------------------------------
        for expected_id, label in expected_tabs:
            print(f"Testing click on {label} ({expected_id})...")
            page.click(f"button[data-tab='{expected_id}']")
            page.wait_for_timeout(500)
            assert page.is_visible(f"#{expected_id}"), f"Section #{expected_id} must be visible after click"
            assert page.locator(f"button[data-tab='{expected_id}']").evaluate("el => el.classList.contains('active')"), (
                f"Button {expected_id} must have .active class"
            )

        # -------------------------------------------------------------
        # STEP 3: CAPTURE HIGH-RES SCREENSHOT OF REORDERED NAV
        # -------------------------------------------------------------
        # Switch back to tab-wizard or tab-figures to show navigation in context
        page.click("button[data-tab='tab-figures']")
        page.wait_for_timeout(600)

        nav_screenshot_path = DOCS_SCREENSHOTS / "41_top_navigation_reordered.png"
        page.screenshot(path=str(nav_screenshot_path), full_page=False)
        print(f"Saved reordered navigation screenshot: {nav_screenshot_path}")

        # Also copy to artifacts dir
        artifact_path = ARTIFACT_DIR / "41_top_navigation_reordered.png"
        shutil.copy(nav_screenshot_path, artifact_path)
        print(f"Copied to artifact dir: {artifact_path}")

        # Capture a tight crop of the top nav header
        header_el = page.locator("header")
        nav_el = page.locator("nav.nav-tabs")
        if nav_el.count() > 0:
            header_crop_path = DOCS_SCREENSHOTS / "42_top_navigation_bar_crop.png"
            # bounding box around header + nav
            header_box = header_el.bounding_box()
            nav_box = nav_el.bounding_box()
            if header_box and nav_box:
                crop_y = min(header_box["y"], nav_box["y"])
                crop_h = (nav_box["y"] + nav_box["height"]) - crop_y + 20
                page.screenshot(
                    path=str(header_crop_path),
                    clip={"x": 0, "y": crop_y, "width": 1600, "height": crop_h}
                )
                print(f"Saved header crop: {header_crop_path}")
                shutil.copy(header_crop_path, ARTIFACT_DIR / "42_top_navigation_bar_crop.png")

        browser.close()
        print("All reordered navigation tests passed successfully!")


if __name__ == "__main__":
    test_top_navigation_reorder()
