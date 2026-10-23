"""
Playwright End-to-End UX Test for Benchmark Loading Bar, Real Data Verification,
Greyed-out Simulation Configuration, and Dynamic Step 3 / Step 4 Label Adjustments.
"""

from pathlib import Path
import pytest
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8085"
ARTIFACT_DIR = Path("/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed")
DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def test_benchmark_loading_bar_and_ux_distinctions():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1050})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)

        # ---------------------------------------------------------------------
        # TEST 1: TAB 2 BENCHMARK LOADING BAR
        # ---------------------------------------------------------------------
        print("2. Navigating to Tab 2 Real-World Benchmarks Studio...")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(800)
        assert page.is_visible("#tab-sos")

        # Verify loading bar container element exists in DOM
        assert page.query_selector("#benchmark-loading-bar-container") is not None
        assert page.query_selector("#benchmark-loading-progress-bar") is not None

        # Verify switching to Norman 2019 dataset
        print("3. Switching benchmark dataset to Norman 2019...")
        page.wait_for_selector("#pill-ds-norman_2019_real_16tf:not([disabled])", timeout=10000)
        page.click("#pill-ds-norman_2019_real_16tf")
        page.wait_for_function("document.getElementById('prov-card-title').textContent.includes('Norman')", timeout=10000)
        card_title = page.inner_text("#prov-card-title")
        assert "Norman" in card_title

        # Capture Tab 2 state
        page.screenshot(path=str(ARTIFACT_DIR / "tab2_benchmark_ready_norman.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "22_tab2_benchmark_ready_norman.png"))

        # ---------------------------------------------------------------------
        # TEST 2: LOAD DIRECT WRANGLED DATA INTO TAB 1
        # ---------------------------------------------------------------------
        print("4. Loading Direct Wrangled Norman 2019 data into Tab 1...")
        page.click("#btn-load-wrangled")
        page.wait_for_timeout(1200)

        # Should automatically switch to Tab 1
        assert page.is_visible("#tab-wizard"), "Should transition to Tab 1 Studio"

        # ---------------------------------------------------------------------
        # TEST 3: REAL EXPERIMENTAL DATA VERIFICATION BANNER
        # ---------------------------------------------------------------------
        print("5. Verifying Real Experimental Data confirmation banner...")
        assert page.is_visible("#empirical-verification-banner"), "Empirical verification banner must be visible"
        banner_text = page.inner_text("#empirical-verification-banner")
        assert "REAL EXPERIMENTAL DATA VERIFIED" in banner_text
        assert "Norman" in banner_text
        assert "Direct Empirical Wrangled Mode" in banner_text
        assert "Numerical ODE time-course simulation is bypassed" in banner_text

        # ---------------------------------------------------------------------
        # TEST 4: SIMULATION WIZARD CONFIGURATION GREYED OUT
        # ---------------------------------------------------------------------
        print("6. Verifying Step 2 Simulation Wizard controls are locked & greyed out...")
        step2_class = page.get_attribute("#step-section-2", "class")
        assert "is-disabled-empirical" in step2_class, "Step 2 section must have is-disabled-empirical class"

        assert page.is_visible("#step2-empirical-locked-message"), "Lock explanation message must be visible"
        lock_msg = page.inner_text("#step2-empirical-locked-message")
        assert "Simulation Wizard Locked" in lock_msg
        assert "Norman" in lock_msg or "real benchmark" in lock_msg

        # Inputs should be disabled
        assert page.is_disabled("#sim-num-genes"), "Gene count slider must be disabled"
        assert page.is_disabled("#sim-topology"), "Topology selector must be disabled"
        assert page.is_disabled("#sim-kav"), "Connectivity slider must be disabled"
        assert page.is_disabled("#sim-noise"), "Noise slider must be disabled"
        assert page.is_disabled("#btn-simulate-only"), "Simulate Only button must be disabled"

        # Run button label should adapt
        run_btn_text = page.inner_text("#btn-run-all")
        assert "Run Reverse-Engineering on Real Data" in run_btn_text

        # Capture screenshot of locked Step 2 and verification banner
        page.screenshot(path=str(ARTIFACT_DIR / "tab1_simulation_controls_greyed_out.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "24_tab1_simulation_controls_greyed_out.png"))

        # ---------------------------------------------------------------------
        # TEST 5: STEP 3 VALIDATED GROUND TRUTH LABEL ADJUSTMENT
        # ---------------------------------------------------------------------
        print("7. Verifying Step 3 label adjustments (Validated Network vs Generated GT)...")
        gt_title = page.inner_text("#gt-panel-title-text")
        assert "Experimentally Validated Ground-Truth Network" in gt_title
        assert "Generated" not in gt_title

        gt_sub = page.inner_text("#gt-panel-subtitle-text")
        assert "Literature-curated physical regulatory topology" in gt_sub

        wstep3_text = page.inner_text("#wstep-3-label")
        assert "Validated Circuit" in wstep3_text

        # Click Step 3 in wizard bar and check context banner
        page.click("#wstep-3")
        page.wait_for_timeout(400)
        banner_title = page.inner_text("#step-banner-title")
        assert "Experimentally Validated Ground-Truth Network" in banner_title
        banner_badge = page.inner_text("#step-banner-badge")
        assert "Literature Curated" in banner_badge

        page.screenshot(path=str(ARTIFACT_DIR / "tab1_step3_validated_circuit_label.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "25_tab1_step3_validated_circuit_label.png"))

        # ---------------------------------------------------------------------
        # TEST 6: STEP 4 EMPIRICAL MEASUREMENTS (X) LABEL ADJUSTMENT
        # ---------------------------------------------------------------------
        print("8. Verifying Step 4 label adjustments (Measured Observations vs ODE Time-Course)...")
        pert_title = page.inner_text("#pert-panel-title-text")
        assert "Measured Experimental Perturbation Profile" in pert_title
        assert "ODE Time-Course" not in pert_title

        pert_sub = page.inner_text("#pert-panel-subtitle-text")
        assert "No ODE integration performed" in pert_sub

        heat_title = page.inner_text("#heatmap-panel-title-text")
        assert "Measured Experimental Expression Matrix" in heat_title

        wstep4_text = page.inner_text("#wstep-4-label")
        assert "Measured Data (X)" in wstep4_text

        # Click Step 4 in wizard bar and check context banner
        page.click("#wstep-4")
        page.wait_for_timeout(400)
        banner_title_4 = page.inner_text("#step-banner-title")
        assert "Empirical Observational Measurements (X)" in banner_title_4
        banner_badge_4 = page.inner_text("#step-banner-badge")
        assert "Real Data" in banner_badge_4
        banner_desc_4 = page.inner_text("#step-banner-desc")
        assert "Numerical ODE integration is bypassed" in banner_desc_4

        page.screenshot(path=str(ARTIFACT_DIR / "tab1_step4_empirical_measurements_profile.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "26_tab1_step4_empirical_measurements_profile.png"))

        # ---------------------------------------------------------------------
        # TEST 7: SWITCH BACK TO SYNTHETIC SIMULATION MODE
        # ---------------------------------------------------------------------
        print("9. Testing 'Switch to Synthetic Simulation Mode' button...")
        page.click("#btn-switch-to-synthetic")
        page.wait_for_function("document.getElementById('btn-run-all').textContent.includes('Simulate & Re-Engineer Network')", timeout=15000)

        # Verification banner should disappear
        assert not page.is_visible("#empirical-verification-banner"), "Verification banner should be hidden"

        # Step 2 controls should be re-enabled
        step2_class_restored = page.get_attribute("#step-section-2", "class")
        assert "is-disabled-empirical" not in step2_class_restored
        assert not page.is_disabled("#sim-num-genes")
        assert not page.is_disabled("#sim-topology")
        assert not page.is_disabled("#sim-kav")
        assert not page.is_disabled("#btn-simulate-only")
        assert "Simulate & Re-Engineer Network" in page.inner_text("#btn-run-all")

        # Step 3 and 4 titles should restore to synthetic ODE names
        assert "Generated Ground-Truth Network" in page.inner_text("#gt-panel-title-text")
        assert "Network & ODE Kinetics" in page.inner_text("#wstep-3-label")
        assert "Perturbation ODE Time-Course" in page.inner_text("#pert-panel-title-text")
        assert "Perturbation Time-Courses" in page.inner_text("#wstep-4-label")

        page.screenshot(path=str(ARTIFACT_DIR / "tab1_restored_synthetic_ode_simulation.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "27_tab1_restored_synthetic_ode_simulation.png"))

        # ---------------------------------------------------------------------
        # TEST 8: BIOPHYSICAL PARAMETER ESTIMATION MODE DISTINCTION
        # ---------------------------------------------------------------------
        print("10. Testing Biophysical Parameter Estimation mode loading...")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(500)
        page.click("#btn-load-biophysical")
        page.wait_for_timeout(1500)

        assert page.is_visible("#tab-wizard")
        assert page.is_visible("#empirical-verification-banner")
        banner_text_bio = page.inner_text("#empirical-verification-banner")
        assert "Biophysical Parameter Estimation Mode" in banner_text_bio

        # Step 3 and 4 should reflect calibrated biophysical ODE dynamics
        gt_title_bio = page.inner_text("#gt-panel-title-text")
        assert "Calibrated Biophysical Network & ODE Kinetics" in gt_title_bio
        pert_title_bio = page.inner_text("#pert-panel-title-text")
        assert "Simulated Relaxation Dynamics (Calibrated ODEs)" in pert_title_bio

        page.screenshot(path=str(ARTIFACT_DIR / "tab1_biophysical_mode_calibrated_labels.png"))
        page.screenshot(path=str(DOCS_SCREENSHOTS / "28_tab1_biophysical_mode_calibrated_labels.png"))

        browser.close()
        print("All UX tests passed successfully!")


if __name__ == "__main__":
    test_benchmark_loading_bar_and_ux_distinctions()
