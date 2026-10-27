"""
Playwright End-to-End Test for:
1. Systematic Benchmark Overall Statistics (Tab 5 renamed to "5. Overall Statistics"):
   - Dynamic Leaderboard Table (7 algorithms evaluated across multi-parameter sweep).
   - Figure 1A & 1B: Network Size vs Power & FDR.
   - Figure 2: Network Topology Impact (Random vs Scale-Free vs Hierarchical).
   - Figure 3: Average Connectivity Sweep (k_av ∈ [1.0, 4.0]).
   - Figure 4: Measurement Noise Sensitivity Sweep (σ_m ∈ [0.0, 0.25]).
   - Instant toggle between New Parameter Sweep Stats and Authentic 2004 MATLAB Historical Gallery.
2. Tab 3 Previous Simulations Archive Restricted View:
   - Restricts display strictly to the best 15 simulations for each set (Responsive, Flat, Oscillating, Topologies).
   - Dynamic rank badges (#1 in Set .. #15 in Set) and F1 Score column.
   - Step 4 Side-by-Side Continuous ODE Relaxation & Expression Matrix (X) + 3×3 Shift Bar Grid toggle.
"""

import shutil
import pytest
from playwright.sync_api import sync_playwright

from _live import APP_URL as BASE_URL
from _live import ARTIFACT_DIR

pytestmark = pytest.mark.live_server
from _live import DOCS_SCREENSHOTS


def test_overall_stats_and_best15_restricted_archive():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()

        print("1. Opening phpIoRN web application...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1000)

        # -----------------------------------------------------------------
        # STEP 1: VERIFY TOP NAVIGATION RENAMED TO "5. Overall Statistics"
        # -----------------------------------------------------------------
        print("2. Verifying top navigation tabs...")
        tab5_btn = page.locator("button[data-tab='tab-figures']")
        assert tab5_btn.is_visible()
        tab5_text = tab5_btn.inner_text().strip()
        print(f"Tab 5 text: '{tab5_text}'")
        assert "5. Overall Statistics" in tab5_text, f"Expected '5. Overall Statistics', got '{tab5_text}'"

        tab3_btn = page.locator("button[data-tab='tab-history']")
        assert tab3_btn.is_visible()
        tab3_text = tab3_btn.inner_text().strip()
        print(f"Tab 3 text: '{tab3_text}'")
        assert "3. Previous Simulations" in tab3_text

        # -----------------------------------------------------------------
        # STEP 2: VERIFY TAB 5 OVERALL STATISTICS (NEW PARAMETER SWEEP)
        # -----------------------------------------------------------------
        print("3. Navigating to Tab 5 (Overall Statistics)...")
        tab5_btn.click()
        page.wait_for_timeout(1000)

        assert page.is_visible("#tab-figures")
        assert page.is_visible("#tab5-view-overall"), "Overall stats view must be visible by default"
        assert not page.is_visible("#tab5-view-historical"), "Historical gallery view must be hidden by default"

        # Verify KPI Cards
        kpi_total = page.locator("#kpi-overall-total-runs").inner_text().strip()
        kpi_alg = page.locator("#kpi-overall-alg-evals").inner_text().strip()
        kpi_top = page.locator("#kpi-overall-top-alg").inner_text().strip()
        kpi_single = page.locator("#kpi-overall-best-single").inner_text().strip()
        print(f"KPIs: Total={kpi_total}, AlgEvals={kpi_alg}, Top={kpi_top}, BestSingle={kpi_single}")
        assert int(kpi_total) >= 40, f"Expected >= 40 runs, found {kpi_total}"
        assert "1,242" in kpi_alg or "1242" in kpi_alg or "1,080" in kpi_alg or "1080" in kpi_alg or "720" in kpi_alg, f"Expected alg evals, got '{kpi_alg}'"
        assert "AtlasShift" in kpi_top, f"Expected 'AtlasShift' in top, got '{kpi_top}'"
        assert "Simple-to-General" in kpi_single or "S2G" in kpi_single, f"Expected S2G in best single, got '{kpi_single}'"

        # Verify Architecture Cross-Comparison Cards (Random, Scale-Free, Hierarchical Modular)
        arch_cards = page.locator("#overall-arch-cards-grid .kpi-card")
        arch_count = arch_cards.count()
        print(f"Architecture cards found: {arch_count}")
        assert arch_count == 3, f"Expected 3 architecture cards (Random, Scale-Free, Hierarchical), found {arch_count}"

        # Verify Leaderboard Table (All 18 Algorithms)
        lb_rows = page.locator("#tbl-overall-leaderboard tbody tr")
        lb_count = lb_rows.count()
        print(f"Leaderboard rows found: {lb_count}")
        assert lb_count == 18, f"Expected 18 algorithms in leaderboard, found {lb_count}"

        # Row 1 must be AtlasShift (Empirical)
        first_row_text = lb_rows.first.inner_text()
        print(f"Leaderboard #1: {first_row_text}")
        assert "AtlasShift" in first_row_text
        assert "#1" in first_row_text

        # Verify Canvases
        for cid in [
            "canvas-overall-size-power",
            "canvas-overall-size-fdr",
            "canvas-overall-topology",
            "canvas-overall-connectivity",
            "canvas-overall-noise",
            "canvas-overall-runtime",
        ]:
            canvas = page.locator(f"#{cid}")
            assert canvas.is_visible(), f"Canvas #{cid} must be visible"

        # -----------------------------------------------------------------
        # STEP 2B: VERIFY GRAPH LINES HOVER SHOWING ALGORITHM
        # -----------------------------------------------------------------
        print("3b. Testing interactive line graph hover showing algorithm...")
        canvas_power = page.locator("#canvas-overall-size-power")
        canvas_power.scroll_into_view_if_needed()
        page.wait_for_timeout(300)

        # Move mouse over S2G point at N=40
        s2g_coords = page.evaluate('''() => {
            const c = document.getElementById('canvas-overall-size-power');
            const s = c._chartConfig.seriesList[1]; // S2G
            const pad = { l: 46, r: 20, t: 42, b: 32 };
            const pw = c.width - pad.l - pad.r;
            const ph = c.height - pad.t - pad.b;
            const ptX = pad.l + (pw * 3) / 5; // N=40
            const ptY = pad.t + ph * (1 - (s.data[3] - 0) / 100);
            const rect = c.getBoundingClientRect();
            return {
                clientX: rect.left + (ptX / c.width) * rect.width,
                clientY: rect.top + (ptY / c.height) * rect.height
            };
        }''')
        page.mouse.move(s2g_coords['clientX'], s2g_coords['clientY'])
        page.wait_for_timeout(300)

        cursor_style = page.evaluate("() => document.getElementById('canvas-overall-size-power').style.cursor")
        assert cursor_style == "pointer", f"Expected pointer cursor on line hover, got {cursor_style}"

        # Capture Screenshot: Overall Statistics View with Hover Active
        shot54 = DOCS_SCREENSHOTS / "54_tab5_hover_algorithm_line.png"
        canvas_power.screenshot(path=str(shot54))
        shutil.copy(shot54, ARTIFACT_DIR / "54_tab5_hover_algorithm_line.png")

        # Move mouse away to reset
        page.mouse.move(10, 10)
        page.wait_for_timeout(200)

        # Capture Screenshot: Overall Statistics View
        shot47 = DOCS_SCREENSHOTS / "47_tab5_overall_statistics_sweep.png"
        page.screenshot(path=str(shot47), full_page=False)
        shutil.copy(shot47, ARTIFACT_DIR / "47_tab5_overall_statistics_sweep.png")
        print(f"Captured: {shot47}")

        # -----------------------------------------------------------------
        # STEP 3: TEST TOGGLE TO HISTORICAL THESIS & PAPER GALLERY (2004)
        # -----------------------------------------------------------------
        print("4. Testing toggle to Historical Figures Gallery...")
        page.click("#btn-tab5-view-historical")
        page.wait_for_timeout(800)

        assert not page.is_visible("#tab5-view-overall"), "Overall stats view must now be hidden"
        assert page.is_visible("#tab5-view-historical"), "Historical gallery view must now be visible"

        # Verify authentic 2004 MATLAB elements
        assert page.is_visible("#btn-fig-src-matlab")
        assert page.is_visible("#fig-overallstats-bar")
        assert page.is_visible("#canvas-fig-topology")
        assert page.is_visible("#canvas-fig-connectivity")
        assert page.is_visible("#canvas-fig-noise")
        assert page.is_visible("#canvas-fig-voting")
        assert page.is_visible("#tbl-fig2-4gene")

        # Capture Screenshot: Historical Gallery View
        shot48 = DOCS_SCREENSHOTS / "48_tab5_historical_gallery_toggle.png"
        page.screenshot(path=str(shot48), full_page=False)
        shutil.copy(shot48, ARTIFACT_DIR / "48_tab5_historical_gallery_toggle.png")
        print(f"Captured: {shot48}")

        # Switch back to Overall Statistics
        page.click("#btn-tab5-view-overall")
        page.wait_for_timeout(500)
        assert page.is_visible("#tab5-view-overall")

        # -----------------------------------------------------------------
        # STEP 4: VERIFY TAB 3 PREVIOUS SIMULATIONS ARCHIVE (BEST 15 PER SET)
        # -----------------------------------------------------------------
        print("5. Navigating to Tab 3 (Previous Simulations Archive)...")
        tab3_btn.click()
        page.wait_for_timeout(1000)

        assert page.is_visible("#tab-history")
        assert page.is_visible("#archive-best15-banner")
        banner_text = page.locator("#archive-best15-banner-text").inner_text()
        print(f"Banner text: {banner_text}")
        assert "Best 15" in banner_text

        # Helper to verify minimum network size N >= 12 for all rows
        def verify_current_table_min_genes(expected_count: int, set_name: str):
            rows = page.locator("#tbl-history tbody tr")
            count = rows.count()
            assert count == expected_count, f"[{set_name}] Expected {expected_count} rows, got {count}"
            for i in range(count):
                row_cells = rows.nth(i).locator("td")
                gene_text = row_cells.nth(4).inner_text().strip()
                n_genes = int(gene_text)
                assert n_genes >= 12, f"[{set_name}] Row {i+1} has num_genes={n_genes} < 12!"

        # 4A. Default 'all' filter: Curated corpus has exactly 45 runs (15 responsive + 15 flat + 15 oscillating, all N >= 12)
        table_rows = page.locator("#tbl-history tbody tr")
        total_all_rows = table_rows.count()
        print(f"Visible rows under 'all' curated filter: {total_all_rows}")
        assert total_all_rows == 45, f"Expected 45 curated runs (15 per dynamics set), got {total_all_rows}"
        verify_current_table_min_genes(45, "all")

        # Verify Rank Badge and F1 Score in the table
        first_row = table_rows.first
        first_row_text = first_row.inner_text()
        print(f"Tab 3 Row 1 text: {first_row_text}")
        assert "#1 in Responsive" in first_row_text
        # F1 score must be displayed (e.g. 1.000)
        assert "1.000" in first_row_text or "0." in first_row_text

        # 4B. Filter: 'responsive' (strictly best 15, N >= 12)
        print("Testing filter 'responsive'...")
        page.select_option("#archive-filter-type", "responsive")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "responsive")
        assert "#1 in Responsive" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4C. Filter: 'flat' (strictly best 15, N >= 12)
        print("Testing filter 'flat'...")
        page.select_option("#archive-filter-type", "flat")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "flat")
        assert "#1 in Flat Levels" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4D. Filter: 'oscillating' (strictly best 15, N >= 12)
        print("Testing filter 'oscillating'...")
        page.select_option("#archive-filter-type", "oscillating")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "oscillating")
        assert "#1 in Oscillating" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4E. Filter: 'power_law' (strictly best 15 scale-free, N >= 12)
        print("Testing filter 'power_law'...")
        page.select_option("#archive-filter-type", "power_law")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "power_law")
        assert "#1 in Scale-Free Topo" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4F. Filter: 'random' (strictly best 15 random, N >= 12)
        print("Testing filter 'random'...")
        page.select_option("#archive-filter-type", "random")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "random")
        assert "#1 in Random Topo" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4G. Filter: 'hierarchical' (strictly best 15 hierarchical, N >= 12)
        print("Testing filter 'hierarchical'...")
        page.select_option("#archive-filter-type", "hierarchical")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "hierarchical")
        assert "#1 in Hierarchical Topo" in page.locator("#tbl-history tbody tr").first.inner_text()

        # 4H. Filter: 'overall_top15' (strictly overall top 15, N >= 12)
        print("Testing filter 'overall_top15'...")
        page.select_option("#archive-filter-type", "overall_top15")
        page.wait_for_timeout(600)
        verify_current_table_min_genes(15, "overall_top15")
        assert "#1 in Overall Best" in page.locator("#tbl-history tbody tr").first.inner_text()

        # Reset filter to 'all' for clean full screenshot
        page.select_option("#archive-filter-type", "all")
        page.wait_for_timeout(600)

        # Capture Screenshot: Tab 3 Archive Best 15 View
        shot49 = DOCS_SCREENSHOTS / "49_tab3_best15_restricted_view.png"
        page.screenshot(path=str(shot49), full_page=False)
        shutil.copy(shot49, ARTIFACT_DIR / "49_tab3_best15_restricted_view.png")
        print(f"Captured: {shot49}")

        # -----------------------------------------------------------------
        # STEP 5: VERIFY STEP 4 SIDE-BY-SIDE & 3x3 SHIFT BAR GRID INSPECTOR
        # -----------------------------------------------------------------
        print("6. Verifying Detailed Inspector for Run #1...")
        # Select first row
        page.locator("#tbl-history tbody tr").first.click()
        page.wait_for_timeout(800)

        # Ensure detailed inspector panel is visible
        assert page.is_visible("#history-detail-panel")
        assert page.is_visible("#hist-step4-dual-container")
        assert page.is_visible("#canvas-hist-pert-timecourse")
        assert page.is_visible("#canvas-hist-pert-matrix")

        # Verify all 18 algorithms are available in the inspector dropdown
        alg_options = page.locator("#hist-detail-alg option")
        print(f"Algorithm options in Tab 3 detail selector: {alg_options.count()}")
        assert alg_options.count() == 18, f"Expected 18 algorithms in detail selector, got {alg_options.count()}"

        # Toggle to 3x3 Shift Bar Grid
        print("Testing 3x3 Shift Bar Grid toggle...")
        page.click("#btn-hist-pert-mode-bar")
        page.wait_for_timeout(600)
        assert page.is_visible("#hist-step4-grid-container")
        assert page.is_visible("#canvas-hist-pert-grid")

        # Capture Screenshot: Detailed Inspector with 3x3 Grid
        shot50 = DOCS_SCREENSHOTS / "50_tab3_step4_dual_and_inspector.png"
        page.locator("#history-detail-panel").scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(path=str(shot50), full_page=False)
        shutil.copy(shot50, ARTIFACT_DIR / "50_tab3_step4_dual_and_inspector.png")
        print(f"Captured: {shot50}")

        browser.close()
        print("All tests passed successfully!")


if __name__ == "__main__":
    test_overall_stats_and_best15_restricted_archive()
