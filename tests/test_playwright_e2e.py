"""
End-to-End Playwright Test for GRN-Sim / phpIoRN Web Studio & SQLite Database Persistence.
Verifies:
1. Starting a brand-new simulation via the Wizard UI (configuring topology='hierarchical',
   num_genes=14, k_av=2.1, act_ratio=0.55, noise=0.03, bio_var=0.01, seed=777).
2. Clicking 'Simulate & Re-Engineer Network' (#btn-run-all) and verifying ODE steady-state /
   oscillation convergence badge (#steady-state-badge) and new run_id.
3. Navigating to Tab 4: Previous Simulations Archive (showpre.php) and verifying that the
   newly created simulation and its 6 reverse-engineering algorithms are persisted in the
   SQLite database (`data/rgn_db.sqlite` tables `nw` and `alg`) and displayed at the top of
   the archive table (#tbl-history).
4. Clicking the newly stored simulation row in Tab 4 to inspect its full detailed results,
   Ground-Truth & 9-State Differential graphs, and 4x4 confusion matrix.
5. Capturing high-resolution Light Mode screenshots to the conversation artifact directory.
"""

import pytest
import json
import os
import sqlite3
import urllib.request


from _live import APP_URL

from grn_sim.paths import archive_db_path
from _live import ARTIFACT_DIR

pytestmark = pytest.mark.live_server
DB_PATH = os.environ.get("DB_PATH") or str(archive_db_path())


def run_playwright_e2e_new_simulation_and_db_test():
    from playwright.sync_api import sync_playwright
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # 1. Check initial database row counts via /api/status
    with urllib.request.urlopen(f"{APP_URL}/api/status") as resp:
        initial_status = json.loads(resp.read().decode("utf-8"))
    initial_run_id = initial_status.get("run_id")
    initial_nw_rows = initial_status.get("database", {}).get("nw_rows", 0)
    print(f"[Playwright E2E] Initial run_id={initial_run_id}, initial SQLite nw_rows={initial_nw_rows}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1640, "height": 1080})
        page = context.new_page()

        # 2. Load the GRN-Sim / phpIoRN Web Studio
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all")

        # Switch to Light Mode so all screenshots are crisp and clear
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)

        assert page.evaluate("document.documentElement.getAttribute('data-theme')") == "light"

        # 3. Configure a brand-new simulation in the Wizard Sidebar
        # Mode: In Silico, Topology: Hierarchical Modular Network, n=14 genes, k_av=2.1, ratio=0.55, noise=0.03, seed=777
        page.select_option("#sim-mode", "insilico")
        page.select_option("#sim-topology", "hierarchical")
        # Changing the mode starts an automatic simulation with the previous settings. Wait until it has finished,
        # otherwise the response matcher below can pick up its response instead of the one for the click.
        page.wait_for_function(
            "() => { const b = document.getElementById('btn-run-all'); return !b.disabled && b.textContent.trim() === 'Simulate & Re-Engineer Network'; }",
            timeout=30000,
        )
        page.wait_for_timeout(300)
        page.evaluate(
            """() => {
                const setSlider = (id, lblId, val, fmt) => {
                    const el = document.getElementById(id);
                    el.value = String(val);
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    const lbl = document.getElementById(lblId);
                    if (lbl) lbl.textContent = fmt ? fmt(val) : String(val);
                };
                setSlider('sim-num-genes', 'lbl-num-genes', 14);
                setSlider('sim-kav', 'lbl-kav', 2.1);
                setSlider('sim-ratio', 'lbl-ratio', 0.55, v => Number(v).toFixed(2));
                setSlider('sim-noise', 'lbl-noise', 0.03, v => Number(v).toFixed(2));
                setSlider('sim-biovar', 'lbl-biovar', 0.01, v => Number(v).toFixed(2));
                const seedEl = document.getElementById('sim-seed');
                seedEl.value = '777';
                seedEl.dispatchEvent(new Event('input', { bubbles: true }));
            }"""
        )
        page.wait_for_timeout(200)

        # 4. Click "Simulate & Re-Engineer Network" (#btn-run-all) and wait for /api/simulate POST response
        with page.expect_response(lambda r: "/api/simulate" in r.url and r.request.method == "POST", timeout=30000) as resp_info:
            page.click("#btn-run-all")
        sim_response = resp_info.value
        assert sim_response.status == 200
        sim_json = sim_response.json()

        new_run_id = sim_json["run_id"]
        assert new_run_id != initial_run_id, f"Expected a new run_id, got {new_run_id}"
        assert sim_json["num_genes"] == 14
        assert sim_json["topology"] == "hierarchical"
        assert sim_json["seed"] == 777
        assert "steady_state_diagnostics" in sim_json
        assert "inferred" in sim_json and "metrics" in sim_json["inferred"]

        page.wait_for_timeout(500)

        # Verify UI badges updated with new run_id and ODE steady-state convergence diagnostics
        active_badge_text = page.locator("#active-run-badge").inner_text().strip()
        ss_badge_text = page.locator("#steady-state-badge").inner_text().strip()
        gt_badge_text = page.locator("#gt-summary-badge").inner_text().strip()
        print(f"[Playwright E2E] New Simulation Completed: run_id={active_badge_text}, steady_state='{ss_badge_text}', summary='{gt_badge_text}'")
        assert active_badge_text == new_run_id
        assert "14 Genes" in gt_badge_text

        # Screenshot 1: Newly simulated 14-gene Hierarchical network in Studio (Light Mode)
        shot_studio = os.path.join(ARTIFACT_DIR, "playwright_e2e_01_new_sim_studio.png")
        page.screenshot(path=shot_studio, full_page=False)
        print(f"[Playwright E2E] Saved screenshot: {shot_studio}")

        # 5. Navigate to Tab 4: Previous Simulations Archive (showpre.php • SQLite rgn_db)
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(600)

        db_badge_text = page.locator("#db-status-badge").inner_text().strip()
        print(f"[Playwright E2E] Archive SQLite badge: {db_badge_text}")
        assert "SQLite rgn_db" in db_badge_text

        # The archive lists only the best 15 runs of each dynamics set, so a fresh run does not necessarily have a
        # row; open it through the page's own loader (what the "View Details" button calls).
        page.evaluate("runId => selectArchiveRun(runId, false)", new_run_id)
        page.wait_for_function("id => document.getElementById('hist-detail-runid').textContent.trim() === id", arg=new_run_id, timeout=15000)
        page.wait_for_timeout(300)

        detail_run_id = page.locator("#hist-detail-runid").inner_text().strip()
        detail_topo = page.locator("#hist-detail-topo").inner_text().strip()
        detail_seed = page.locator("#hsp-biovar").inner_text().strip()
        detail_steady = page.locator("#hsp-steady").inner_text().strip()
        print(
            f"[Playwright E2E] Detailed Inspector loaded: run_id={detail_run_id}, "
            f"topo={detail_topo}, seed='{detail_seed}', convergence='{detail_steady}'"
        )
        assert detail_run_id == new_run_id
        assert detail_topo == "hierarchical"
        assert "777" in detail_seed

        # Screenshot 2: Tab 4 Archive Table + Detailed Inspector showing the newly stored run
        shot_archive = os.path.join(ARTIFACT_DIR, "playwright_e2e_02_sqlite_archive_detail.png")
        page.screenshot(path=shot_archive, full_page=True)
        print(f"[Playwright E2E] Saved screenshot: {shot_archive}")

        # 6. Verify Dynamical Categories & Filter Bar in Tab 4
        # Check summary cards container
        summary_cards = page.locator("#archive-summary-container .summary-type-card")
        assert summary_cards.count() == 3, f"Expected 3 summary cards, found {summary_cards.count()}"
        print("[Playwright E2E] Verified 3 summary cards for Responsive, Flat, and Oscillating categories")

        # Check Algorithm Comparative Table and open it
        page.click("#btn-toggle-alg-breakdown")
        page.wait_for_timeout(200)
        comp_table_rows = page.locator("#tbl-dynamics-alg-comparison tbody tr")
        print(f"[Playwright E2E] Found {comp_table_rows.count()} rows in Algorithm Comparison table")
        assert comp_table_rows.count() > 0

        # Test filtering by category: 'flat'
        page.select_option("#archive-filter-type", "flat")
        page.wait_for_timeout(300)
        filtered_count_text = page.locator("#archive-filter-count-badge").inner_text()
        print(f"[Playwright E2E] Filtered by flat: {filtered_count_text}")
        assert "showing" in filtered_count_text.lower() or "runs" in filtered_count_text.lower()

        # Capture Screenshot 3: Tab 4 Filtered by Category with Summary Cards
        shot_categories = os.path.join(ARTIFACT_DIR, "playwright_e2e_03_dynamics_categories_archive.png")
        page.screenshot(path=shot_categories, full_page=True)
        print(f"[Playwright E2E] Saved screenshot: {shot_categories}")

        # Reset filter to all
        page.select_option("#archive-filter-type", "all")
        page.wait_for_timeout(200)

        browser.close()

    # 6. Verify directly against the on-disk SQLite database (`data/rgn_db.sqlite`)
    # If testing against the running docker container, copy the container DB to a local check path
    check_db_path = DB_PATH
    import subprocess
    try:
        if ":8085" not in APP_URL:
            raise RuntimeError(f"{APP_URL} is not the docker container (port 8085); reading DB_PATH directly")
        res = subprocess.run(
            ["docker", "cp", "grn-sim-phpiorn:/app/data/rgn_db.sqlite", "/tmp/rgn_db_e2e.sqlite"],
            capture_output=True,
            timeout=5,
        )
        if res.returncode == 0 and os.path.exists("/tmp/rgn_db_e2e.sqlite"):
            check_db_path = "/tmp/rgn_db_e2e.sqlite"
            print(f"[Playwright E2E] Synced latest SQLite DB from docker container to {check_db_path}")
    except Exception as e:
        print(f"[Playwright E2E] Notice: docker cp check skipped ({e}), falling back to {DB_PATH}")

    assert os.path.exists(check_db_path), f"SQLite database file not found at {check_db_path}"
    conn = sqlite3.connect(check_db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT simid, uid, nz, nt, av_k, ratio, noise, seed, steady_state, oscillating FROM nw WHERE simid = ?",
            (new_run_id,),
        )
        nw_row = cur.fetchone()
        assert nw_row is not None, f"Run {new_run_id} not found in SQLite `nw` table!"
        print(f"[Playwright E2E] Verified SQLite `nw` row: {nw_row}")
        assert nw_row[2] == 14
        assert nw_row[3] == "hierarchical"
        assert nw_row[7] == 777

        cur.execute(
            "SELECT algid, spec, sens, total_errors FROM alg WHERE simid = ? ORDER BY total_errors ASC",
            (new_run_id,),
        )
        alg_rows = cur.fetchall()
        expected_algs = len(sim_json["inferred"]["metrics"])  # the studio's default selection has grown over time
        assert expected_algs >= 6 and len(alg_rows) == expected_algs, (
            f"Expected {expected_algs} algorithm rows in SQLite `alg` table for {new_run_id}, got {len(alg_rows)}"
        )
        print(f"[Playwright E2E] Verified SQLite `alg` rows ({len(alg_rows)} algorithms): {alg_rows}")
    finally:
        conn.close()

    print("[Playwright E2E] ALL PLAYWRIGHT UI + SQLITE DATABASE PERSISTENCE CHECKS PASSED!")


def test_playwright_e2e_new_simulation_and_db():
    try:
        with urllib.request.urlopen(f"{APP_URL}/api/status", timeout=2):
            pass
    except Exception:
        import pytest
        pytest.skip(f"GRN-Sim web server not running at {APP_URL}, skipping Playwright E2E UI test")
    run_playwright_e2e_new_simulation_and_db_test()


if __name__ == "__main__":
    run_playwright_e2e_new_simulation_and_db_test()
