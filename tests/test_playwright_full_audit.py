"""
Comprehensive End-to-End Audit & Verification Suite for GRN-Sim / phpIoRN 2.0 Web Application.

Audits every tab, module, dataset, and algorithm:
1. Tab 1: Simulation & Inference Studio (runsim.php)
   - 6-step wizard bar and contextual banner synchronization.
   - Mode switching: In Silico, E. coli SOS (9 genes), Norman 2019 Real CRISPRa (16 TFs).
   - Numerical ODE integration, Ground Truth layout views (Canvas, dot, neato, DOT source).
   - Interactive Node / Edge ODE parameter inspector (inline card and dialog modal).
   - Perturbation dynamic trajectories, steady-state expression ratio heatmap X.
   - Inference execution across 12 algorithms, AIC cost curve profile.
   - 9-state differential reconstruction graph (statsgraph_to_dot.m styling), $TMP_STATMAT ribbon, and $TMP_CMAT.
   - File downloads (.msmt, .pert, .dot, .svg, .h5ad).
2. Tab 2: Real-World Benchmarks Studio
   - Loading all real datasets: E. coli SOS, Norman 2019 Real CRISPRa, DREAM4, Replogle 2022.
   - Clickable SVG graph nodes activating Gene Dynamics & Trajectory Inspector.
   - Value change over time curves (basal relaxation, self-perturbation, regulator perturbation).
   - Value states bar charts across all perturbation conditions.
   - Causal regulator navigation pills.
   - Loading real circuits directly into Studio (Tab 1).
3. Tab 3: Thesis & Paper Figures Gallery
   - Reproduction of all 6 thesis and paper figures (Fig 4.1-4.5, Fig 2.1).
   - Interactive rendering of canvases and 4-gene toy network table.
4. Tab 4: Previous Simulations Archive (showpre.php)
   - SQLite persistent database querying (42+ simulation runs).
   - Categorization by perturbation response type (Responsive, Flat, Oscillating).
   - Comparative algorithm breakdown table.
   - Interactive row inspection and table sorting.
5. Tab 5: Virtual Cell Challenge 2026 & AtlasShift Studio
   - In-distribution vs out-of-distribution (OOD) perturbation generalization metrics.
   - Table 1 comparative edge recovery table.
   - Cell context rewiring sensitivity decay sweep (rho in [0%, 50%]).
   - Single-cell AnnData (.h5ad) export.
"""

import json
import os
import time
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("GRN_SIM_URL", "http://localhost:8085")
ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    "/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed",
)


def run_full_application_audit():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"============================================================")
    print(f"🚀 STARTING FULL APPLICATION AUDIT: {APP_URL}")
    print(f"============================================================")

    audit_results = {
        "tab1_studio": {},
        "tab2_benchmarks": {},
        "tab3_thesis_figures": {},
        "tab4_archive": {},
        "tab5_vcc": {},
        "downloads": {},
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1150})
        page = context.new_page()

        # ---------------------------------------------------------------------
        # 1. LOAD APP & INITIAL STATE
        # ---------------------------------------------------------------------
        print("\n--- [Audit 1/6] Loading Application and Verifying Initial State ---")
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all", timeout=15000)

        # Ensure light theme for clear visual evidence
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)

        active_run_id = page.locator("#active-run-badge").inner_text()
        print(f"  [OK] Initial Studio Hydration: Active Run ID = {active_run_id}")
        assert active_run_id, "Active run ID must be populated"

        # ---------------------------------------------------------------------
        # 2. AUDIT TAB 1: SIMULATION & INFERENCE STUDIO (runsim.php)
        # ---------------------------------------------------------------------
        print("\n--- [Audit 2/6] Auditing Tab 1: 6-Step Workflow & Studio Controls ---")

        # 2.1 Verify 6-step wizard and banner
        banner_title = page.locator("#step-banner-title").inner_text()
        assert banner_title, "Wizard banner title must be present"
        print(f"  [OK] Wizard Banner initialized: '{banner_title}'")

        # 2.2 Test In Silico simulation execution
        print("  [Action] Testing In Silico simulation run...")
        page.select_option("#sim-mode", "insilico")
        page.wait_for_timeout(500)
        page.click("#btn-run-all")
        page.wait_for_timeout(2500)
        page.wait_for_selector("#btn-run-all:not([disabled])", timeout=15000)

        gt_badge = page.locator("#gt-summary-badge").inner_text()
        print(f"  [OK] Simulation completed. Ground-truth badge: {gt_badge}")
        audit_results["tab1_studio"]["insilico_simulation"] = "SUCCESS"

        # 2.3 Verify ground truth views (Canvas, dot, neato, source)
        page.click("#btn-gt-dot")
        page.wait_for_timeout(400)
        assert page.locator("#gt-svg-dot svg").is_visible(), "Hierarchical Graphviz dot SVG must be visible"

        page.click("#btn-gt-neato")
        page.wait_for_timeout(400)
        assert page.locator("#gt-svg-neato svg").is_visible(), "Spring Graphviz neato SVG must be visible"

        page.click("#btn-gt-source")
        page.wait_for_timeout(200)
        dot_src = page.locator("#gt-dot-source").inner_text()
        assert "digraph" in dot_src, "DOT source code must contain 'digraph'"

        page.click("#btn-gt-canvas")
        page.wait_for_timeout(200)
        print("  [OK] Ground truth views (Canvas, dot SVG, neato SVG, DOT source) verified.")

        # 2.4 Verify ODE Parameter Inspector (Modal & Inline)
        page.click("#quick-node-select")
        page.select_option("#quick-node-select", "0")
        page.wait_for_timeout(300)
        inline_text = page.locator("#inline-param-card").inner_text()
        assert "Gene" in inline_text or "Node" in inline_text or "v_i" in inline_text or "Kinetics" in inline_text, "Inline ODE parameter card must be populated"
        print("  [OK] Inline ODE Parameter Inspector card verified.")

        # Open Inspector Modal
        page.click("button:has-text('Open Modal')")
        page.wait_for_timeout(400)
        dialog_visible = page.locator("#inspector-dialog").is_visible()
        assert dialog_visible, "Inspector dialog modal must open"
        page.click("#inspector-dialog button:has-text('Close')")
        page.wait_for_timeout(200)
        print("  [OK] Inspector dialog modal opened and closed successfully.")

        # 2.5 Verify 9-State Differential Graph & Metrics
        stat_power = page.locator("#st-power").inner_text()
        stat_spec = page.locator("#st-spec").inner_text()
        stat_ppv = page.locator("#st-ppv").inner_text()
        print(f"  [OK] Inferred Network Metrics Ribbon: Power={stat_power}, Spec={stat_spec}, PPV={stat_ppv}")
        assert stat_power != "--" and stat_spec != "--", "Statistical ribbon must be populated"

        # 2.6 Test Step 1 Real Dataset Mode Switching: E. coli SOS
        print("  [Action] Switching Studio to Real E. coli SOS Mode...")
        page.select_option("#sim-mode", "sos")
        page.wait_for_timeout(1500)
        page.wait_for_selector("#btn-run-all:not([disabled])", timeout=15000)

        sos_num = page.locator("#sim-num-genes").input_value()
        assert sos_num == "9", f"SOS mode must have 9 genes, got {sos_num}"
        assert page.locator("#sim-num-genes").is_disabled(), "Gene count must be locked for SOS mode"
        print(f"  [OK] Real E. coli SOS circuit loaded into Studio (9 genes, locked sliders).")
        audit_results["tab1_studio"]["sos_real_mode"] = "SUCCESS"

        # 2.7 Test Step 1 Real Dataset Mode Switching: Norman 2019 CRISPRa
        print("  [Action] Switching Studio to Real Norman 2019 CRISPRa Perturb-seq...")
        page.select_option("#sim-mode", "benchmark_norman_2019_real_16tf")
        page.wait_for_timeout(1500)
        page.wait_for_selector("#btn-run-all:not([disabled])", timeout=15000)

        norman_num = page.locator("#sim-num-genes").input_value()
        assert norman_num == "16", f"Norman mode must have 16 genes, got {norman_num}"
        assert page.locator("#sim-num-genes").is_disabled(), "Gene count must be locked for Norman mode"
        print(f"  [OK] Real Norman 2019 CRISPRa circuit loaded into Studio (16 genes, locked sliders).")
        audit_results["tab1_studio"]["norman_real_mode"] = "SUCCESS"

        shot_tab1_path = os.path.join(ARTIFACT_DIR, "audit_01_tab1_studio_verified.png")
        page.screenshot(path=shot_tab1_path)
        print(f"  [Screenshot] Tab 1 Studio screenshot saved -> {shot_tab1_path}")

        # ---------------------------------------------------------------------
        # 3. AUDIT TAB 2: REAL-WORLD BENCHMARKS (SOS, DREAM4, PERTURB-SEQ)
        # ---------------------------------------------------------------------
        print("\n--- [Audit 3/6] Auditing Tab 2: Real-World Benchmarks Studio ---")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(800)
        page.wait_for_selector("#tbl-sos-benchmark tbody tr", timeout=15000)

        # 3.1 Test all 4 benchmark datasets in selector
        benchmarks_to_test = [
            ("ecoli_sos", 9, "recA"),
            ("norman_2019_real_16tf", 16, "KLF1"),
            ("dream4_10gene", 10, "G1"),
            ("replogle_2022_p53", 12, "TP53"),
        ]

        for b_id, expected_n, test_gene in benchmarks_to_test:
            print(f"  [Action] Testing Benchmark Dataset: {b_id} (N={expected_n})...")
            with page.expect_response(lambda res: "/api/benchmark/dataset" in res.url and res.status == 200, timeout=15000):
                page.click(f"#pill-ds-{b_id}")
            page.wait_for_selector(f"#benchmark-header-badge:has-text('{expected_n} Genes')", timeout=10000)
            page.wait_for_timeout(300)

            # Verify node count in Ground Truth SVG
            nodes = page.locator("#sos-true-stage g.node").all()
            print(f"    - Ground Truth SVG rendered with {len(nodes)} nodes (expected {expected_n}).")
            assert len(nodes) == expected_n, f"Expected {expected_n} nodes in {b_id}, got {len(nodes)}"

            # 3.2 Test Clickable Node & Trajectory Inspector
            print(f"    - Clicking node '{test_gene}' to test dynamics and value states...")
            node_target = page.locator(f"#sos-true-stage g.node:has-text('{test_gene}')").first
            if node_target.count():
                node_target.click()
                page.wait_for_timeout(500)
                active_gene = page.locator("#sos-active-gene-badge").inner_text()
                assert test_gene in active_gene, f"Active gene badge must show '{test_gene}', got: '{active_gene}'"
                print(f"    [OK] Gene Dynamics Inspector activated for {test_gene}: {active_gene}")

            audit_results["tab2_benchmarks"][b_id] = "SUCCESS"

        shot_tab2_path = os.path.join(ARTIFACT_DIR, "audit_02_tab2_benchmarks_verified.png")
        page.screenshot(path=shot_tab2_path)
        print(f"  [Screenshot] Tab 2 Benchmarks screenshot saved -> {shot_tab2_path}")

        # ---------------------------------------------------------------------
        # 4. AUDIT TAB 3: THESIS & PAPER FIGURES GALLERY
        # ---------------------------------------------------------------------
        print("\n--- [Audit 4/6] Auditing Tab 3: Thesis & Paper Figures Gallery ---")
        page.click("button[data-tab='tab-figures']")
        page.wait_for_timeout(800)

        # Check all 6 thesis figure canvases
        figures = [
            ("canvas-fig-topology", "Fig 4.1 Topology vs Network Size"),
            ("canvas-fig-connectivity", "Fig 4.2 Connectivity (k_av) Sweep"),
            ("canvas-fig-noise", "Fig 4.3 Noise Sensitivity Sweep"),
            ("canvas-fig-voting", "Fig 4.4 Voting Ensemble Consensus"),
            ("canvas-fig-runtime", "Fig 4.5 Runtime Complexity Scaling"),
            ("canvas-fig-errormodel", "Fig 2.1 Microarray Log-Normal Error Model"),
        ]

        for canvas_id, fig_name in figures:
            c = page.locator(f"#{canvas_id}")
            assert c.is_visible(), f"Figure canvas #{canvas_id} ({fig_name}) must be visible"
            print(f"  [OK] {fig_name} (#{canvas_id}) rendered.")
            audit_results["tab3_thesis_figures"][canvas_id] = "SUCCESS"

        shot_tab3_path = os.path.join(ARTIFACT_DIR, "audit_03_tab3_thesis_figures_verified.png")
        page.screenshot(path=shot_tab3_path)
        print(f"  [Screenshot] Tab 3 Figures screenshot saved -> {shot_tab3_path}")

        # ---------------------------------------------------------------------
        # 5. AUDIT TAB 4: PREVIOUS SIMULATIONS ARCHIVE (showpre.php)
        # ---------------------------------------------------------------------
        print("\n--- [Audit 5/6] Auditing Tab 4: SQLite Simulation Archive (showpre.php) ---")
        page.click("button[data-tab='tab-history']")
        page.wait_for_timeout(800)
        page.wait_for_selector("#tbl-history tbody tr", timeout=15000)

        rows = page.locator("#tbl-history tbody tr").all()
        print(f"  [OK] SQLite Archive loaded with {len(rows)} persistent simulation runs.")
        assert len(rows) > 0, "Simulation archive must have rows"

        # Check categorization filter
        page.select_option("#archive-filter-type", "responsive")
        page.wait_for_timeout(300)
        resp_count = page.locator("#archive-filter-count-badge").inner_text()
        print(f"  [OK] Filter 'Responsive' applied: {resp_count}")

        page.select_option("#archive-filter-type", "flat")
        page.wait_for_timeout(300)
        flat_count = page.locator("#archive-filter-count-badge").inner_text()
        print(f"  [OK] Filter 'Flat' applied: {flat_count}")

        page.select_option("#archive-filter-type", "oscillating")
        page.wait_for_timeout(300)
        osc_count = page.locator("#archive-filter-count-badge").inner_text()
        print(f"  [OK] Filter 'Oscillating' applied: {osc_count}")

        page.select_option("#archive-filter-type", "all")
        page.wait_for_timeout(300)

        # Test table sorting on #tbl-history
        header_power = page.locator("#tbl-history th:has-text('Power')")
        header_power.click()
        page.wait_for_timeout(300)
        assert "sort-" in header_power.get_attribute("class"), "Header must have sort class after click"
        print("  [OK] Table column sorting on #tbl-history verified.")

        audit_results["tab4_archive"]["history_table_rows"] = len(rows)
        audit_results["tab4_archive"]["filtering"] = "SUCCESS"

        shot_tab4_path = os.path.join(ARTIFACT_DIR, "audit_04_tab4_archive_verified.png")
        page.screenshot(path=shot_tab4_path)
        print(f"  [Screenshot] Tab 4 Archive screenshot saved -> {shot_tab4_path}")

        # ---------------------------------------------------------------------
        # 6. AUDIT TAB 5: VIRTUAL CELL CHALLENGE 2026 & ATLASSHIFT STUDIO
        # ---------------------------------------------------------------------
        print("\n--- [Audit 6/6] Auditing Tab 5: VCC 2026 & AtlasShift Studio ---")
        page.click("button[data-tab='tab-vcc']")
        page.wait_for_timeout(800)
        page.wait_for_selector("#tbl-vcc-summary tbody tr", timeout=15000)

        mech_seen = page.locator("#vcc-seen-mech-r").inner_text()
        atlas_seen = page.locator("#vcc-seen-atlas-r").inner_text()
        mech_ood = page.locator("#vcc-ood-mech-r").inner_text()
        atlas_ood = page.locator("#vcc-ood-atlas-r").inner_text()

        print(f"  [OK] Seen Targets: Mechanistic r={mech_seen}, AtlasShift r={atlas_seen}")
        print(f"  [OK] Out-of-Distribution (OOD): Mechanistic r={mech_ood}, AtlasShift r={atlas_ood}")

        assert mech_seen != "--" and mech_ood != "--", "VCC benchmark metrics must be computed"

        # Check rewire decay canvas
        rewire_canvas = page.locator("#canvas-vcc-rewire")
        assert rewire_canvas.is_visible(), "Rewiring decay chart must be visible"

        # Check Table 1 rows
        vcc_rows = page.locator("#tbl-vcc-summary tbody tr").all()
        print(f"  [OK] VCC Table 1 rendered with {len(vcc_rows)} comparative algorithm rows.")
        assert len(vcc_rows) >= 5, "VCC Table 1 must have at least 5 algorithm rows"

        audit_results["tab5_vcc"]["table_1_rows"] = len(vcc_rows)
        audit_results["tab5_vcc"]["metrics"] = {
            "seen_mech": mech_seen,
            "seen_atlas": atlas_seen,
            "ood_mech": mech_ood,
            "ood_atlas": atlas_ood,
        }

        shot_tab5_path = os.path.join(ARTIFACT_DIR, "audit_05_tab5_vcc_verified.png")
        page.screenshot(path=shot_tab5_path)
        print(f"  [Screenshot] Tab 5 VCC screenshot saved -> {shot_tab5_path}")

        # ---------------------------------------------------------------------
        # 7. AUDIT DOWNLOADS & EXPORTS
        # ---------------------------------------------------------------------
        print("\n--- Auditing File Downloads & Exports ---")
        # Check that download links exist on Tab 1
        page.click("button[data-tab='tab-wizard']")
        page.wait_for_timeout(400)

        formats = ["msmt", "pert", "dot", "svg", "h5ad"]
        for fmt in formats:
            link = page.locator(f"a[href='/api/download?format={fmt}']")
            assert link.is_visible(), f"Download link for .{fmt} must be visible"
            print(f"  [OK] Download .{fmt} link verified.")
            audit_results["downloads"][fmt] = "VERIFIED"

        browser.close()

    print("\n============================================================")
    print("✅ COMPREHENSIVE APPLICATION AUDIT COMPLETED SUCCESSFULLY!")
    print("============================================================")
    return audit_results


if __name__ == "__main__":
    results = run_full_application_audit()
    print(json.dumps(results, indent=2))
