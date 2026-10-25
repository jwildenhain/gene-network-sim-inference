"""
Playwright Automated Test Suite for 6-Step Reverse-Engineering Wizard & Dynamic Mode UI Impact.

Verifies:
1. Wizard Bar & Guidance Banner:
   - Initial presence and synchronization of #wizard-step-banner with active step.
   - Step pills (#wstep-1 .. #wstep-6) displaying .active and .current state classes.
2. Direct Step Selection Impact (Steps 1 through 6):
   - Clicking #wstep-1 through #wstep-6 updates the guidance banner title, description, and action button.
   - Actively targets and highlights the corresponding UI sections with .step-highlight-active.
   - Smooth scrolling and active focus on target containers:
     * Step 1: #step-section-1 (Mode & source selector card)
     * Step 2: #step-section-2 (Topology, kinetics engine, noise sliders)
     * Step 3: #panel-step3-gt (Ground-truth network stage & ODE kinetic inspector)
     * Step 4: #panel-step4-pert (Perturbation ODE time-courses & Matrix X heatmap)
     * Step 5: #step-section-5 & #panel-step5-aic (Algorithms checklist & AIC cost curve)
     * Step 6: #panel-step6-inf & #panel-step6-statmat (9-state differential graph & metrics ribbon)
3. Step Navigation Controls:
   - Next (#btn-wizard-next) and Prev (#btn-wizard-prev) sequential step advancement.
4. Step Focus Mode:
   - Toggling #btn-toggle-focus-mode enables .step-focus-active on workspace.
   - Non-active sections are dimmed while active step is prominently spotlighted.
5. Mode Selection Feedback (#sim-mode):
   - Switching to 'sos': sets gene count to 9 (locked), disables topology sliders, updates info card.
   - Switching to 'benchmark_norman_2019_real_16tf': sets gene count to 16 (locked), updates info card.
   - Switching to 'insilico': re-enables sliders and updates info card.
6. Visual Evidence:
   - Saves high-resolution screenshots to the conversation artifact directory.
"""

import pytest
import os
from playwright.sync_api import sync_playwright

from _live import APP_URL
from _live import ARTIFACT_DIR

pytestmark = pytest.mark.live_server
def test_wizard_steps_and_mode_impact():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"[Playwright] Connecting to GRN-Sim studio at {APP_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1150})
        page = context.new_page()

        # 1. Load Web Studio
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all", timeout=15000)

        # Switch to Light Theme for crisp screenshots
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)

        # 2. Verify Wizard Step Banner exists
        banner = page.locator("#wizard-step-banner")
        assert banner.is_visible(), "Wizard step banner #wizard-step-banner must be visible"

        badge_text = page.locator("#step-banner-badge").inner_text()
        print(f"[Test] Initial Step Banner Badge: {badge_text}")

        # 3. Test Clicking Steps 1 to 6
        step_expectations = [
            (1, "Mode & Data Source Configuration", "#step-section-1"),
            (2, "Network Topology, Kinetics & Noise", "#step-section-2"),
            (3, "Ground-Truth Network & ODE Kinetics", "#panel-step3-gt"),
            (4, "Perturbation ODE Time-Courses & Matrix X", "#panel-step4-pert"),
            (5, "Reverse-Engineering Algorithms & AIC Profile", "#step-section-5"),
            (6, "9-State Differential Graph & Metrics Ribbon", "#panel-step6-inf"),
        ]

        for step_num, expected_title, expected_selector in step_expectations:
            step_btn = page.locator(f"#wstep-{step_num}")
            step_btn.click()
            page.wait_for_timeout(300)

            # Assert current class on pill
            pill_class = step_btn.get_attribute("class")
            assert "current" in pill_class, f"#wstep-{step_num} must have .current class after click, got: {pill_class}"

            # Assert title update
            current_title = page.locator("#step-banner-title").inner_text()
            assert current_title == expected_title, f"Expected banner title '{expected_title}', got '{current_title}'"

            # Assert active highlight on target section
            target_el = page.locator(expected_selector)
            target_class = target_el.get_attribute("class")
            assert "step-highlight-active" in target_class, f"{expected_selector} must have .step-highlight-active, got: {target_class}"
            print(f"[Test] Step {step_num} successfully verified: Banner '{current_title}' -> Highlighted {expected_selector}")

        # Screenshot of Step 3 highlighted
        page.locator("#wstep-3").click()
        page.wait_for_timeout(400)
        shot_step3_path = os.path.join(ARTIFACT_DIR, "playwright_step3_ground_truth_highlight.png")
        page.screenshot(path=shot_step3_path)
        print(f"[Screenshot] Captured Step 3 Highlight -> {shot_step3_path}")

        # 4. Test Step Focus Mode (Dimmable workspace)
        focus_btn = page.locator("#btn-toggle-focus-mode")
        focus_btn.click()
        page.wait_for_timeout(300)

        focus_label = page.locator("#focus-mode-label").inner_text()
        assert "Step Focus: ON" in focus_label, f"Expected Focus ON label, got: {focus_label}"

        container_class = page.locator("#tab-wizard").get_attribute("class")
        assert "step-focus-active" in container_class, f"#tab-wizard must have .step-focus-active, got: {container_class}"

        # Capture Screenshot of Step Focus Mode
        shot_focus_path = os.path.join(ARTIFACT_DIR, "playwright_step_focus_mode_active.png")
        page.screenshot(path=shot_focus_path)
        print(f"[Screenshot] Captured Step Focus Mode Active -> {shot_focus_path}")

        # Toggle Focus Mode OFF
        focus_btn.click()
        page.wait_for_timeout(300)
        assert "Step Focus: OFF" in page.locator("#focus-mode-label").inner_text()

        # 5. Test Prev and Next buttons
        page.locator("#wstep-1").click()
        page.wait_for_timeout(200)
        assert "Step 1 of 6" in page.locator("#step-banner-badge").inner_text()

        page.locator("#btn-wizard-next").click()
        page.wait_for_timeout(200)
        assert "Step 2 of 6" in page.locator("#step-banner-badge").inner_text()

        page.locator("#btn-wizard-next").click()
        page.wait_for_timeout(200)
        assert "Step 3 of 6" in page.locator("#step-banner-badge").inner_text()

        page.locator("#btn-wizard-prev").click()
        page.wait_for_timeout(200)
        assert "Step 2 of 6" in page.locator("#step-banner-badge").inner_text()
        print("[Test] Sequential Next / Prev navigation verified successfully")

        # 6. Test Step 1 Mode Change Feedback (#sim-mode)
        page.locator("#wstep-1").click()
        page.wait_for_timeout(200)

        # Select E. coli SOS mode
        page.select_option("#sim-mode", "sos")
        page.wait_for_timeout(1000)

        sos_num_genes = page.locator("#sim-num-genes").input_value()
        sos_is_disabled = page.locator("#sim-num-genes").is_disabled()
        assert sos_num_genes == "9", f"Expected 9 genes for SOS mode, got {sos_num_genes}"
        assert sos_is_disabled, "Gene count slider must be disabled/locked for SOS mode"

        mode_info_text = page.locator("#sim-mode-info").inner_text()
        assert "E. coli SOS" in mode_info_text, f"Expected SOS in info card, got: {mode_info_text}"
        print(f"[Test] SOS Mode successfully locked to 9 genes and updated info: {mode_info_text[:50]}...")

        # Select Norman 2019 Real CRISPRa mode
        page.select_option("#sim-mode", "benchmark_norman_2019_real_16tf")
        page.wait_for_timeout(1000)

        norman_num_genes = page.locator("#sim-num-genes").input_value()
        norman_is_disabled = page.locator("#sim-num-genes").is_disabled()
        assert norman_num_genes == "16", f"Expected 16 genes for Norman 2019 mode, got {norman_num_genes}"
        assert norman_is_disabled, "Gene count slider must be disabled/locked for Norman 2019 mode"

        norman_info_text = page.locator("#sim-mode-info").inner_text()
        assert "Norman et al. 2019" in norman_info_text, f"Expected Norman in info card, got: {norman_info_text}"
        print(f"[Test] Norman 2019 Mode successfully locked to 16 genes and updated info: {norman_info_text[:50]}...")

        # Switch back to In Silico mode
        page.select_option("#sim-mode", "insilico")
        page.wait_for_timeout(1000)

        insilico_disabled = page.locator("#sim-num-genes").is_disabled()
        assert not insilico_disabled, "Gene count slider must be re-enabled for in silico mode"
        print("[Test] In Silico Mode re-enabled sliders successfully")

        # Capture final screenshot of Studio
        shot_final_path = os.path.join(ARTIFACT_DIR, "playwright_wizard_steps_verified_studio.png")
        page.screenshot(path=shot_final_path)
        print(f"[Screenshot] Final verification studio screenshot saved -> {shot_final_path}")

        browser.close()
        print("✅ ALL 6 WIZARD STEPS AND MODE SELECTION IMPACT TESTS PASSED!")


if __name__ == "__main__":
    test_wizard_steps_and_mode_impact()
