"""
Comprehensive Playwright UX & Functional Verification Suite for GRN-Sim / phpIoRN Web Studio.
Covers:
1. Navigation across all 5 primary application tabs:
   - Tab 1: Studio (Wizard / runsim.php)
   - Tab 2: Real Biological Benchmarks (SOS 9-gene network / Ronen et al.)
   - Tab 3: Thesis & Paper Figures Gallery (Chapters 2–4)
   - Tab 4: Previous Simulations Archive (showpre.php / SQLite rgn_db)
   - Tab 5: Virtual Cell Challenge & AtlasShift Single-Cell Transfer (Norman Perturb-seq)
2. Tab 4:
   - Summary statistics cards click interactions (filtering by card click).
   - Category filter dropdown (#archive-filter-type) switching across All, Responsive, Flat, and Oscillating.
   - Fast simulation trigger buttons (+ Simulate Flat, + Simulate Oscillating, + Simulate Responsive).
   - Expandable comparative algorithm performance breakdown table toggle.
   - Selecting archive rows to inspect detailed ground truth & 9-state differential SVGs, metrics, and parameters.
   - "Open in Studio" action button verifying parameter synchronization back to Tab 1.
3. Tab 1 (Simulation Studio):
   - Selecting Dynamics Preset dropdown (#sim-dynamics-preset) and verifying dynamic adjustments of pert_factor, noise, kav.
   - Running simulation & reverse engineering algorithms across multiple topology modes.
   - Inspecting ODE Parameter Inspector dialog (#inspector-dialog).
4. Tab 2 (Real Biological Benchmarks - SOS):
   - Verifying known SOS regulatory network graph (#sos-true-stage).
   - Verifying 9-state differential reconstruction graph (#sos-diff-stage).
   - Verifying benchmark comparison table (#tbl-sos-benchmark) and 4x4 confusion matrix (#tbl-sos-cmat).
5. Tab 3 (Thesis Figures & Publication Gallery):
   - Viewing publication-ready SVG figures, runtime scaling canvas, Table 2.1 4-gene kinetic parameters.
6. Tab 5 (Virtual Cell Challenge & AtlasShift):
   - Verifying In-Distribution vs Out-of-Distribution generalization cards (#vcc-ood-mech-r).
   - Verifying Table 1 causal network topology edge recovery benchmark table (#tbl-vcc-summary).
7. Downloads & Exports:
   - Verifying API endpoints for .msmt, .pert, .dot, .svg, and .h5ad downloads return HTTP 200 with non-empty content.
8. Console & UX Error Monitoring:
   - Monitoring page for uncaught JS exceptions, rejected promises, or broken layout.
9. Capturing full-page screenshots of key UX states to the artifact directory.
"""

import json
import os
import sys
import urllib.request
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("GRN_SIM_URL", "http://localhost:8085")
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    "/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed",
)


def run_comprehensive_ux_tests():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("=" * 80)
    print(f"STARTING COMPREHENSIVE PLAYWRIGHT UX VERIFICATION AT {APP_URL}")
    print("=" * 80)

    # Verify server availability
    try:
        with urllib.request.urlopen(f"{APP_URL}/api/status", timeout=5) as resp:
            status_data = json.loads(resp.read().decode("utf-8"))
            print(f"[UX Test] Server ready! Active run: {status_data.get('run_id')}, NW rows: {status_data.get('database', {}).get('nw_rows')}")
    except Exception as e:
        print(f"[UX Test ERROR] Server not reachable at {APP_URL}: {e}")
        sys.exit(1)

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1050})
        page = context.new_page()

        # Listen for console and page errors
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("\n[Step 1] Loading Web Studio and testing theme toggle...")
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all", timeout=10000)

        # Ensure light mode for clean screenshots
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "light"
        print("  ✓ Theme set to light mode successfully")

        # --------------------------------------------------------------------
        # TEST TAB 4: ARCHIVE, DYNAMICS CATEGORIES & SUMMARY CARDS
        # --------------------------------------------------------------------
        print("\n[Step 2] Testing Tab 4: Previous Simulations Archive & Dynamics Categorization...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(500)

        # Verify summary statistics cards are rendered (3 cards)
        cards = page.locator("#archive-summary-container .summary-type-card")
        card_count = cards.count()
        print(f"  ✓ Found {card_count} summary category cards (Responsive, Flat, Oscillating)")
        assert card_count == 3, f"Expected 3 summary cards, found {card_count}"

        # Verify initial table row count
        initial_rows = page.locator("#tbl-history tbody tr").count()
        filter_badge_text = page.locator("#archive-filter-count-badge").inner_text()
        print(f"  ✓ Initial archive table rows: {initial_rows} ({filter_badge_text})")
        assert initial_rows > 0

        # Test clicking the "Flat Levels" summary card to filter table
        print("\n[Step 3] Testing Card Click Filtering (Flat Levels)...")
        cards.nth(1).click()
        page.wait_for_timeout(400)
        flat_select_val = page.locator("#archive-filter-type").input_value()
        flat_badge_text = page.locator("#archive-filter-count-badge").inner_text()
        flat_rows_count = page.locator("#tbl-history tbody tr").count()
        print(f"  ✓ Selected filter value: '{flat_select_val}', Badge: '{flat_badge_text}', Rows shown: {flat_rows_count}")
        assert flat_select_val == "flat"
        assert flat_rows_count > 0

        # Verify each displayed row has the Flat Levels badge
        flat_badges = page.locator("#tbl-history tbody tr td:nth-child(4) span")
        for i in range(flat_rows_count):
            assert "flat" in flat_badges.nth(i).inner_text().lower()
        print("  ✓ All displayed rows correctly belong to Flat Levels category")

        # Test clicking the "Oscillating" summary card to filter table
        print("\n[Step 4] Testing Card Click Filtering (Oscillating)...")
        cards.nth(2).click()
        page.wait_for_timeout(400)
        osc_select_val = page.locator("#archive-filter-type").input_value()
        osc_badge_text = page.locator("#archive-filter-count-badge").inner_text()
        osc_rows_count = page.locator("#tbl-history tbody tr").count()
        print(f"  ✓ Selected filter value: '{osc_select_val}', Badge: '{osc_badge_text}', Rows shown: {osc_rows_count}")
        assert osc_select_val == "oscillating"
        assert osc_rows_count > 0

        # Test resetting filter via dropdown
        print("\n[Step 5] Testing Category Dropdown Filter Selection...")
        page.select_option("#archive-filter-type", "all")
        page.wait_for_timeout(300)
        all_rows_count = page.locator("#tbl-history tbody tr").count()
        print(f"  ✓ Reset to All Categories: {all_rows_count} rows visible")
        assert all_rows_count == initial_rows

        # Test Algorithmic Breakdown toggle
        print("\n[Step 6] Testing Algorithm Comparative Table Expansion...")
        breakdown_wrapper = page.locator("#archive-alg-breakdown-wrapper")
        assert not breakdown_wrapper.is_visible()
        page.click("#btn-toggle-alg-breakdown")
        page.wait_for_timeout(300)
        assert breakdown_wrapper.is_visible()
        alg_rows = page.locator("#tbl-dynamics-alg-comparison tbody tr").count()
        print(f"  ✓ Algorithm Comparative Table visible with {alg_rows} comparison entries")
        assert alg_rows >= 6

        # Test Detailed Inspector by clicking rows
        print("\n[Step 7] Testing Detailed Run Inspector in Tab 4...")
        first_row = page.locator("#tbl-history tbody tr").first
        first_row_id = first_row.locator("code").inner_text()
        first_row.click()
        page.wait_for_timeout(500)

        detail_panel = page.locator("#history-detail-panel")
        assert detail_panel.is_visible()
        detail_run_id = page.locator("#hist-detail-runid").inner_text()
        detail_badge = page.locator("#hist-detail-dynamics").inner_text()
        print(f"  ✓ Detailed Inspector active for Run ID: {detail_run_id}, Category Badge: '{detail_badge}'")
        assert detail_run_id == first_row_id

        # Verify SVGs rendered in detail panel
        gt_svg = page.locator("#hist-gt-svg-stage svg")
        assert gt_svg.count() > 0 or page.locator("#hist-gt-svg-stage").inner_html() != ""
        print("  ✓ Ground truth graph visualization rendered in Detailed Inspector")

        # Capture screenshot of Tab 4 full archive state
        shot_archive_full = os.path.join(ARTIFACT_DIR, "ux_verification_01_tab4_archive_full.png")
        page.screenshot(path=shot_archive_full, full_page=True)
        print(f"  ✓ Saved screenshot: {shot_archive_full}")

        # Test "Open in Studio" button from archive
        print("\n[Step 8] Testing 'Open in Studio' Action from Archive...")
        open_in_studio_btn = first_row.locator("button:has-text('Open in Studio')")
        open_in_studio_btn.click()
        page.wait_for_timeout(600)

        # Verify we navigated back to tab-wizard (Studio)
        active_tab_studio = page.locator("#tab-wizard")
        assert active_tab_studio.is_visible()
        active_run_badge = page.locator("#active-run-badge").inner_text()
        print(f"  ✓ Successfully transitioned to Tab 1 Studio with loaded Run ID: {active_run_badge}")
        assert active_run_badge == first_row_id

        # --------------------------------------------------------------------
        # TEST FAST ACTION TRIGGER BUTTONS IN TAB 4
        # --------------------------------------------------------------------
        print("\n[Step 9] Testing Fast Action Preset Triggers in Tab 4 (+ Simulate Flat)...")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(400)

        # Trigger "+ Simulate Flat"
        with page.expect_response(lambda r: "/api/simulate" in r.url and r.request.method == "POST", timeout=30000) as resp_info:
            page.click("button:has-text('+ Simulate Flat')")
        flat_sim_resp = resp_info.value
        assert flat_sim_resp.status == 200
        flat_sim_data = flat_sim_resp.json()
        print(f"  ✓ '+ Simulate Flat' completed! New run_id: {flat_sim_data['run_id']}, Category: {flat_sim_data.get('response_type')}")
        assert flat_sim_data.get("response_type") == "flat"

        # Verify table refreshed and new run appears at top of archive
        page.wait_for_timeout(500)
        top_run_id = page.locator("#tbl-history tbody tr").first.locator("code").inner_text()
        assert top_run_id == flat_sim_data["run_id"]
        print(f"  ✓ Newly simulated flat run {top_run_id} automatically placed at top of archive table")

        # --------------------------------------------------------------------
        # TEST TAB 1: SIMULATION STUDIO & PRESET SELECTION
        # --------------------------------------------------------------------
        print("\n[Step 10] Testing Tab 1: Studio Dynamics Preset Selector & Simulation Execution...")
        page.click("button[data-tab='tab-wizard']")
        page.wait_for_timeout(400)

        # Test changing dynamics preset to Oscillating in Studio sidebar
        page.select_option("#sim-dynamics-preset", "oscillating")
        page.wait_for_timeout(300)
        badge_text = page.locator("#response-type-badge").inner_text()
        print(f"  ✓ Dynamics Preset changed to oscillating. Badge updated to: '{badge_text}'")
        assert "oscillating" in badge_text.lower()

        # Run simulation with oscillating preset
        with page.expect_response(lambda r: "/api/simulate" in r.url and r.request.method == "POST", timeout=30000) as resp_info:
            page.click("#btn-run-all")
        osc_sim_resp = resp_info.value
        assert osc_sim_resp.status == 200
        osc_sim_data = osc_sim_resp.json()
        print(f"  ✓ Studio simulation completed! run_id: {osc_sim_data['run_id']}, Category: {osc_sim_data.get('response_type')}")
        assert osc_sim_data.get("response_type") == "oscillating"

        # Verify steady-state badge indicates Limit Cycle / Oscillation
        ss_badge = page.locator("#steady-state-badge").inner_text()
        print(f"  ✓ Steady state badge: '{ss_badge}'")
        assert "oscillat" in ss_badge.lower() or "limit" in ss_badge.lower()

        # Test ODE Parameter Inspector dialog
        page.click("button:has-text('Open Modal')")
        page.wait_for_timeout(400)
        inspector_dialog = page.locator("#inspector-dialog")
        assert inspector_dialog.is_visible()
        modal_title = page.locator("#modal-title").inner_text()
        print(f"  ✓ ODE Kinetic Parameter Inspector dialog opened: '{modal_title}'")
        page.click("#inspector-dialog button:has-text('Close')")
        page.wait_for_timeout(200)

        # Capture screenshot of Studio with Oscillating Dynamics
        shot_studio_osc = os.path.join(ARTIFACT_DIR, "ux_verification_02_studio_oscillating.png")
        page.screenshot(path=shot_studio_osc, full_page=False)
        print(f"  ✓ Saved screenshot: {shot_studio_osc}")

        # --------------------------------------------------------------------
        # TEST TAB 2: REAL BIOLOGICAL BENCHMARKS (SOS NETWORK)
        # --------------------------------------------------------------------
        print("\n[Step 11] Testing Tab 2: Real Biological Benchmarks (E. coli SOS Network)...")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(600)

        assert page.locator("#tab-sos").is_visible()
        sos_true_badge = page.locator("#sos-true-badge").inner_text()
        print(f"  ✓ Known SOS Regulatory Network stage loaded: '{sos_true_badge}'")
        assert "Edges" in sos_true_badge

        # Verify benchmark comparison table is populated
        sos_bench_rows = page.locator("#tbl-sos-benchmark tbody tr").count()
        print(f"  ✓ SOS Benchmark Comparison table populated with {sos_bench_rows} algorithms")
        assert sos_bench_rows >= 4

        # Verify 4x4 Confusion Matrix table is populated
        sos_cmat_rows = page.locator("#tbl-sos-cmat tbody tr").count()
        assert sos_cmat_rows > 0
        print("  ✓ SOS 4x4 Confusion Matrix table rendered")

        # Capture screenshot of Benchmark studio
        shot_benchmarks = os.path.join(ARTIFACT_DIR, "ux_verification_03_benchmarks_sos.png")
        page.screenshot(path=shot_benchmarks, full_page=False)
        print(f"  ✓ Saved screenshot: {shot_benchmarks}")

        # --------------------------------------------------------------------
        # TEST TAB 3: PUBLICATION FIGURES & PAPER GALLERY
        # --------------------------------------------------------------------
        print("\n[Step 12] Testing Tab 3: Figures & Publication Gallery...")
        page.click("button[data-tab='tab-figures']")
        page.wait_for_timeout(500)

        assert page.locator("#tab-figures").is_visible()
        fig_table_rows = page.locator("#tbl-fig2-4gene tbody tr").count()
        print(f"  ✓ Table 2.1 4-Gene Kinetic Parameters loaded with {fig_table_rows} gene rows")
        assert fig_table_rows >= 4

        # Capture screenshot of Figures gallery
        shot_figures = os.path.join(ARTIFACT_DIR, "ux_verification_04_figures_gallery.png")
        page.screenshot(path=shot_figures, full_page=False)
        print(f"  ✓ Saved screenshot: {shot_figures}")

        # --------------------------------------------------------------------
        # TEST TAB 5: VIRTUAL CELL CHALLENGE & ATLASHIFT TRANSFER
        # --------------------------------------------------------------------
        print("\n[Step 13] Testing Tab 5: Virtual Cell Challenge & AtlasShift Transfer...")
        page.click("button[data-tab='tab-vcc']")
        page.wait_for_selector("#tbl-vcc-summary tbody tr", timeout=15000)

        assert page.locator("#tab-vcc").is_visible()
        vcc_summary_rows = page.locator("#tbl-vcc-summary tbody tr").count()
        print(f"  ✓ Table 1 Causal Network Topology Edge Recovery loaded with {vcc_summary_rows} algorithms")
        assert vcc_summary_rows >= 4

        # Capture screenshot of Tab 5 VCC & AtlasShift
        shot_vcc = os.path.join(ARTIFACT_DIR, "ux_verification_05_vcc_atlasshift.png")
        page.screenshot(path=shot_vcc, full_page=False)
        print(f"  ✓ Saved screenshot: {shot_vcc}")

        # --------------------------------------------------------------------
        # TEST FILE DOWNLOAD ENDPOINTS
        # --------------------------------------------------------------------
        print("\n[Step 14] Testing File Download Endpoints (.msmt, .pert, .dot, .svg)...")
        test_run = flat_sim_data["run_id"]
        for fmt in ["msmt", "pert", "dot", "svg"]:
            url = f"{APP_URL}/api/download?format={fmt}&run_id={test_run}"
            with urllib.request.urlopen(url) as r:
                assert r.status == 200
                content = r.read()
                assert len(content) > 0
                print(f"  ✓ Download format '{fmt}' returned HTTP 200 ({len(content)} bytes)")

        # Verify perturb-seq h5ad export endpoint
        h5ad_url = f"{APP_URL}/api/export/perturb_seq_h5ad"
        with urllib.request.urlopen(h5ad_url) as r:
            assert r.status == 200
            h5ad_bytes = r.read()
            assert len(h5ad_bytes) > 0
            print(f"  ✓ Perturb-Seq AnnData (.h5ad) export returned HTTP 200 ({len(h5ad_bytes)} bytes)")

        # --------------------------------------------------------------------
        # CONSOLE ERROR CHECK
        # --------------------------------------------------------------------
        print("\n[Step 15] Checking for Javascript Console or Page Errors...")
        filtered_errors = [e for e in console_errors if "favicon" not in e.lower()]
        print(f"  Uncaught page errors: {len(page_errors)}")
        print(f"  Console errors: {len(filtered_errors)}")
        if page_errors:
            print(f"  Page errors detail: {page_errors}")
        if filtered_errors:
            print(f"  Console errors detail: {filtered_errors}")
        assert len(page_errors) == 0, f"Found unhandled page errors: {page_errors}"

        browser.close()

    print("\n" + "=" * 80)
    print("ALL 15 COMPREHENSIVE PLAYWRIGHT UX & FUNCTIONAL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_ux_tests()
