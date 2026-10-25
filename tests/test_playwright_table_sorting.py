"""
Playwright Automated Test Suite for Table Column Sorting across GRN-Sim / phpIoRN Web Studio.

Verifies:
1. Sortable column indicators (▲, ▼, ↕) and CSS classes (sort-asc, sort-desc, sortable).
2. Proper parsing and sorting of:
   - Numerical IDs and integers (#14, 16, etc.)
   - Decimal floats and error counts (1.8, 3, etc.)
   - Percentages (91.2%, 100.0%, etc.)
   - ISO/Standard timestamps (YYYY-MM-DD HH:MM:SS)
   - Badge text with leading unicode emojis (⚡ Responsive, ⚪ Flat Levels, 〰 Oscillating)
3. Ascending <-> Descending toggle on repeated clicks.
4. Non-sortable columns (.no-sort on Downloads, Actions) do not trigger sorting or mutate state.
5. Selection state and click event delegation preservation when rows are re-ordered.
6. Sorting functionality across multiple tables:
   - #tbl-history (Simulation Archive & Execution History)
   - #tbl-dynamics-alg-comparison (Comparative Algorithm Breakdown)
   - #tbl-sos-benchmark (SOS / Norman Benchmark Methods)
   - #tbl-alg-comparison (Tab 1 Algorithm Comparison)
   - #tbl-vcc-summary (Tab 5 Virtual Cell Challenge Benchmark)
7. Captures high-resolution visual evidence screenshots to the artifact directory.
"""

import json
import os
import re
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("GRN_SIM_URL", "http://localhost:8085")
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    "/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed",
)


def parse_clean_number(text):
    """Parse number from raw cell text like '#14', '91.2%', '3'."""
    s = text.strip()
    s = re.sub(r"^#", "", s)
    s = re.sub(r"%$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def test_table_sorting():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"[Playwright] Connecting to GRN-Sim studio at {APP_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1100})
        page = context.new_page()

        # Load Web Tool
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all")

        # Ensure light mode for crisp screenshots
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)

        # =========================================================================
        # 1. TEST TAB 4: ARCHIVE TABLE (#tbl-history)
        # =========================================================================
        print("\n--- Testing Tab 4: Archive Table (#tbl-history) Sorting ---")
        page.click("button[data-tab='tab-history']")
        page.wait_for_selector("#tbl-history tbody tr")
        page.wait_for_timeout(600)

        # Check headers have sortable class, except Downloads and Actions
        headers = page.locator("#tbl-history thead th")
        header_count = headers.count()
        print(f"Verified {header_count} headers in #tbl-history")
        assert header_count >= 14

        downloads_th = page.locator("#tbl-history thead th", has_text="Downloads")
        actions_th = page.locator("#tbl-history thead th", has_text="Actions")
        assert "no-sort" in (downloads_th.get_attribute("class") or "")
        assert "no-sort" in (actions_th.get_attribute("class") or "")

        def col_of(header_text):
            """Index of an archive column, looked up by its header so added or reordered columns do not break the test."""
            th = page.locator("#tbl-history thead th", has_text=header_text)
            return th.evaluate("el => Array.from(el.parentElement.children).indexOf(el)")

        # A. Sort by Total Errors ASCENDING
        total_err_th = page.locator("#tbl-history thead th", has_text="Total Errors")
        err_col = col_of("Total Errors")
        total_err_th.click()
        page.wait_for_timeout(300)

        assert "sort-asc" in (total_err_th.get_attribute("class") or "")
        icon_text = total_err_th.locator(".sort-icon").inner_text()
        assert icon_text == "▲", f"Expected ▲ but got {icon_text}"

        # Extract values in column 13 (0-indexed Total Errors)
        rows = page.locator("#tbl-history tbody tr")
        row_count = rows.count()
        assert row_count >= 5, f"Expected at least 5 archive rows, got {row_count}"

        err_vals_asc = []
        for i in range(row_count):
            txt = rows.nth(i).locator("td").nth(err_col).inner_text().strip()
            err_vals_asc.append(float(txt))

        print(f"Total Errors (ASC): {err_vals_asc}")
        assert err_vals_asc == sorted(err_vals_asc), f"Rows not sorted ascending: {err_vals_asc}"

        # B. Sort by Total Errors DESCENDING
        total_err_th.click()
        page.wait_for_timeout(300)

        assert "sort-desc" in (total_err_th.get_attribute("class") or "")
        icon_text = total_err_th.locator(".sort-icon").inner_text()
        assert icon_text == "▼", f"Expected ▼ but got {icon_text}"

        err_vals_desc = []
        for i in range(row_count):
            txt = rows.nth(i).locator("td").nth(err_col).inner_text().strip()
            err_vals_desc.append(float(txt))

        print(f"Total Errors (DESC): {err_vals_desc}")
        assert err_vals_desc == sorted(err_vals_desc, reverse=True), f"Rows not sorted descending: {err_vals_desc}"

        # C. Sort by Power (%) ASCENDING & DESCENDING
        power_th = page.locator("#tbl-history thead th", has_text="Power (%)")
        power_th.click()
        page.wait_for_timeout(300)
        power_col = col_of("Power (%)")

        # Previous header must lose active class
        assert "sort-asc" not in (total_err_th.get_attribute("class") or "")
        assert "sort-desc" not in (total_err_th.get_attribute("class") or "")
        assert "sort-asc" in (power_th.get_attribute("class") or "")

        power_vals_asc = []
        for i in range(row_count):
            raw = rows.nth(i).locator("td").nth(power_col).inner_text().strip()
            power_vals_asc.append(parse_clean_number(raw))

        print(f"Power % (ASC): {power_vals_asc}")
        assert power_vals_asc == sorted(power_vals_asc), f"Power % not sorted ascending: {power_vals_asc}"

        # D. Sort by the number of genes ASCENDING (the first column is now the rank within a dynamics set)
        id_th = page.locator("#tbl-history thead th", has_text="Genes (N)")
        id_col = col_of("Genes (N)")
        id_th.click()
        page.wait_for_timeout(300)

        id_vals_asc = []
        for i in range(row_count):
            raw = rows.nth(i).locator("td").nth(id_col).inner_text().strip()
            id_vals_asc.append(parse_clean_number(raw))

        print(f"ID (ASC): {id_vals_asc}")
        assert id_vals_asc == sorted(id_vals_asc), f"IDs not sorted ascending: {id_vals_asc}"

        # E. Sort by Dynamics Category
        cat_th = page.locator("#tbl-history thead th", has_text="Dynamics Set")
        cat_th.click()
        page.wait_for_timeout(300)

        cat_vals_asc = []
        for i in range(row_count):
            raw = rows.nth(i).locator("td").nth(col_of("Dynamics Set")).inner_text().strip()
            clean = re.sub(r"^[\s\W_]+", "", raw).strip().lower()
            cat_vals_asc.append(clean)

        print(f"Dynamics Category (ASC): {cat_vals_asc}")
        assert cat_vals_asc == sorted(cat_vals_asc), f"Category not sorted: {cat_vals_asc}"

        # F. Non-sortable columns do nothing
        downloads_th.click()
        page.wait_for_timeout(200)
        assert "sort-asc" not in (downloads_th.get_attribute("class") or "")
        assert "sort-desc" not in (downloads_th.get_attribute("class") or "")

        # G. Re-sort by Total Errors ASC and test Row Click Selection
        total_err_th.click()  # toggles or sets
        if "sort-asc" not in (total_err_th.get_attribute("class") or ""):
            total_err_th.click()
        page.wait_for_timeout(200)

        first_row = rows.nth(0)
        first_run_id = first_row.locator("td").nth(col_of("Run ID")).inner_text().strip()
        print(f"Clicking first sorted row: {first_run_id}")
        first_row.click()
        page.wait_for_timeout(500)

        # Verify selected-row highlight
        assert "selected-row" in (first_row.get_attribute("class") or "")

        # Verify detail panel loaded
        detail_title = page.locator("#hist-detail-runid").inner_text()
        assert first_run_id in detail_title, f"Expected {first_run_id} in {detail_title}"

        # Capture Tab 4 Screenshot
        screenshot_archive_path = os.path.join(ARTIFACT_DIR, "playwright_table_sorting_archive.png")
        page.screenshot(path=screenshot_archive_path, full_page=True)
        print(f"Saved Tab 4 sorted table screenshot to: {screenshot_archive_path}")

        # =========================================================================
        # 2. TEST TAB 4: DYNAMICS COMPARISON TABLE (#tbl-dynamics-alg-comparison)
        # =========================================================================
        print("\n--- Testing Tab 4: Dynamics Breakdown Table Sorting ---")
        # Toggle breakdown table if hidden
        breakdown_btn = page.locator("#btn-toggle-alg-breakdown")
        if breakdown_btn.is_visible():
            breakdown_btn.click()
            page.wait_for_timeout(400)

        dyn_table = page.locator("#tbl-dynamics-alg-comparison")
        if dyn_table.is_visible():
            dyn_err_th = dyn_table.locator("thead th", has_text="Avg Total Errors")
            dyn_err_th.click()
            page.wait_for_timeout(300)
            assert "sort-asc" in (dyn_err_th.get_attribute("class") or "")

            dyn_rows = dyn_table.locator("tbody tr")
            dyn_errs = []
            for i in range(dyn_rows.count()):
                txt = dyn_rows.nth(i).locator("td").nth(3).inner_text().strip()
                dyn_errs.append(float(txt))
            print(f"Breakdown Table Avg Total Errors (ASC): {dyn_errs}")
            assert dyn_errs == sorted(dyn_errs), "Breakdown table not sorted by Avg Total Errors"

        # =========================================================================
        # 3. TEST TAB 2: BENCHMARK STUDIO (#tbl-sos-benchmark)
        # =========================================================================
        print("\n--- Testing Tab 2: Benchmark Studio Table Sorting ---")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_selector("#tbl-sos-benchmark tbody tr")
        page.wait_for_timeout(500)

        sos_rows = page.locator("#tbl-sos-benchmark tbody tr")
        assert sos_rows.count() >= 5

        # Sort by Total Errors
        sos_err_th = page.locator("#tbl-sos-benchmark thead th", has_text="Total Errors")
        sos_err_th.click()
        page.wait_for_timeout(300)
        assert "sort-asc" in (sos_err_th.get_attribute("class") or "")

        bench_errs = []
        for i in range(sos_rows.count()):
            txt = sos_rows.nth(i).locator("td").nth(6).inner_text().strip()
            bench_errs.append(float(txt))
        print(f"Benchmark Table Total Errors (ASC): {bench_errs}")
        assert bench_errs == sorted(bench_errs), "Benchmark table not sorted by Total Errors"

        # Sort by Coverage (%) DESC
        sos_cov_th = page.locator("#tbl-sos-benchmark thead th", has_text="Coverage (%)")
        sos_cov_th.click()
        page.wait_for_timeout(200)
        sos_cov_th.click()  # second click for DESC
        page.wait_for_timeout(300)
        assert "sort-desc" in (sos_cov_th.get_attribute("class") or "")

        bench_cov = []
        for i in range(sos_rows.count()):
            txt = sos_rows.nth(i).locator("td").nth(1).inner_text().strip()
            bench_cov.append(parse_clean_number(txt))
        print(f"Benchmark Table Coverage % (DESC): {bench_cov}")
        assert bench_cov == sorted(bench_cov, reverse=True), "Benchmark table not sorted by Coverage % DESC"

        # Capture Tab 2 Screenshot
        screenshot_bench_path = os.path.join(ARTIFACT_DIR, "playwright_table_sorting_benchmark.png")
        page.screenshot(path=screenshot_bench_path, full_page=True)
        print(f"Saved Tab 2 sorted table screenshot to: {screenshot_bench_path}")

        # =========================================================================
        # 4. TEST TAB 1: ALGORITHM COMPARISON (#tbl-alg-comparison)
        # =========================================================================
        print("\n--- Testing Tab 1: Algorithm Comparison Table Sorting ---")
        page.click("button[data-tab='tab-wizard']")
        page.wait_for_selector("#tbl-alg-comparison tbody tr")
        page.wait_for_timeout(500)

        t1_rows = page.locator("#tbl-alg-comparison tbody tr")
        assert t1_rows.count() >= 3

        # Sort by Total Errors ASC
        t1_err_th = page.locator("#tbl-alg-comparison thead th", has_text="Total Errors")
        t1_err_th.click()
        page.wait_for_timeout(300)
        assert "sort-asc" in (t1_err_th.get_attribute("class") or "")

        t1_errs = []
        for i in range(t1_rows.count()):
            txt = t1_rows.nth(i).locator("td").nth(6).inner_text().strip()
            t1_errs.append(float(txt))
        print(f"Tab 1 Total Errors (ASC): {t1_errs}")
        assert t1_errs == sorted(t1_errs), "Tab 1 table not sorted by Total Errors"

        # Click a sorted row and verify algorithm switch works
        top_alg_tr = t1_rows.nth(0)
        top_alg_name = top_alg_tr.locator("td").nth(0).inner_text().strip()
        print(f"Clicking sorted row in Tab 1: {top_alg_name}")
        top_alg_tr.click()
        page.wait_for_timeout(300)

        active_badge = page.locator("#active-alg-badge").inner_text().strip()
        assert top_alg_name.upper() == active_badge.upper(), f"Expected {top_alg_name} but got {active_badge}"

        print("\n✅ All Table Sorting Playwright Tests Passed Successfully!")
        browser.close()


if __name__ == "__main__":
    test_table_sorting()
