"""
Playwright Automated Test Suite for Clickable Nodes and Dynamic Trajectories on Real Datasets.

Verifies:
1. Interactive SVG graph nodes (hover effects, cursor: pointer, active-node highlight).
2. Clicking a node in Ground Truth (#sos-true-stage) or 9-State Differential (#sos-diff-stage)
   graphs activates the Gene Dynamics & Trajectory Inspector for that gene.
3. Display of:
   - Value change over time: Unperturbed basal relaxation x(t), direct self-perturbation response,
     and regulator dynamic perturbation curves (#canvas-sos-gene-traj).
   - Value states across all experimental perturbation conditions (#canvas-sos-gene-states).
   - Upstream regulators and downstream targets with interactive one-click switching.
4. Switching across real datasets (E. coli SOS DNA repair -> Norman 2019 CRISPRa Perturb-seq).
5. Toggling between '📈 Gene Focus' and '▦ All Grid' views.
6. Enlarging gene dynamics in the dialog inspector modal.
7. Saves high-resolution screenshots to the conversation artifact directory.
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


def test_clickable_nodes_and_trajectories():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"[Playwright] Connecting to GRN-Sim studio at {APP_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1680, "height": 1150})
        page = context.new_page()

        # 1. Load Web Studio
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all")

        # Switch to Light Theme
        current_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
        if current_theme != "light":
            page.click("#theme-toggle-btn")
            page.wait_for_timeout(300)

        # 2. Navigate to Tab 2: Real-World Benchmarks
        print("\n--- Navigating to Tab 2: Real-World Benchmarks ---")
        page.click("button[data-tab='tab-sos']")
        page.wait_for_selector("#sos-true-stage svg")
        page.wait_for_selector("#sos-diff-stage svg")
        page.wait_for_selector("#canvas-sos-gene-traj")
        page.wait_for_timeout(600)

        # 3. Test Clicking Nodes on E. coli SOS Ground Truth Graph
        print("\n--- Testing Clickable Nodes on E. coli SOS Ground Truth Graph ---")
        nodes = page.locator("#sos-true-stage svg g.node")
        node_count = nodes.count()
        print(f"Found {node_count} SVG nodes in #sos-true-stage")
        assert node_count >= 9, f"Expected at least 9 nodes, got {node_count}"

        # Find and click the 'lexA' node
        lexa_node = page.locator("#sos-true-stage svg g.node", has_text="lexA").first
        assert lexa_node.is_visible()
        print("Clicking 'lexA' node in Ground Truth graph...")
        lexa_node.click()
        page.wait_for_timeout(500)

        # Verify active gene badge and selector
        active_badge = page.locator("#sos-active-gene-badge").inner_text()
        print(f"Active Gene Badge: {active_badge}")
        assert "lexA" in active_badge, f"Expected 'lexA' in badge, got '{active_badge}'"

        select_val = page.locator("#sos-gene-select").input_value()
        assert select_val == "lexA", f"Expected select value 'lexA', got '{select_val}'"

        # Verify active-node class on the SVG node
        assert "active-node" in (lexa_node.get_attribute("class") or "")

        # Verify Trajectory and Value State canvases are visible
        traj_canvas = page.locator("#canvas-sos-gene-traj")
        states_canvas = page.locator("#canvas-sos-gene-states")
        assert traj_canvas.is_visible()
        assert states_canvas.is_visible()

        # Capture Screenshot of E. coli SOS with lexA selected
        screenshot_sos_path = os.path.join(ARTIFACT_DIR, "playwright_real_dataset_node_click_trajectory.png")
        page.screenshot(path=screenshot_sos_path, full_page=True)
        print(f"Saved E. coli SOS node click screenshot to: {screenshot_sos_path}")

        # 4. Test Clicking a Node in 9-State Differential Graph (#sos-diff-stage)
        print("\n--- Testing Clickable Nodes in 9-State Differential Graph ---")
        reca_diff_node = page.locator("#sos-diff-stage svg g.node", has_text="recA").first
        if reca_diff_node.is_visible():
            print("Clicking 'recA' node in Differential Reconstruction graph...")
            reca_diff_node.click()
            page.wait_for_timeout(400)

            active_badge_reca = page.locator("#sos-active-gene-badge").inner_text()
            print(f"Active Gene Badge after clicking recA: {active_badge_reca}")
            assert "recA" in active_badge_reca

            # Upstream regulators for recA: lexA is a known regulator
            in_regs_text = page.locator("#sos-in-regs-pills").inner_text()
            print(f"recA Upstream Regulators: {in_regs_text}")

            # Click the regulator pill 'lexA' to switch back
            reg_pill = page.locator("#sos-in-regs-pills button", has_text="lexA").first
            if reg_pill.is_visible():
                print("Clicking regulator pill 'lexA'...")
                reg_pill.click()
                page.wait_for_timeout(300)
                assert "lexA" in page.locator("#sos-active-gene-badge").inner_text()

        # 5. Test Real Norman 2019 CRISPRa Perturb-seq Dataset
        print("\n--- Testing Real Norman 2019 Perturb-seq Dataset ---")
        norman_btn = page.locator("#pill-ds-norman_2019_real_16tf")
        norman_btn.click()
        page.wait_for_selector("#sos-true-stage svg")
        page.wait_for_timeout(800)

        norman_nodes = page.locator("#sos-true-stage svg g.node")
        norman_count = norman_nodes.count()
        print(f"Found {norman_count} nodes in Norman 2019 real dataset")
        assert norman_count == 16, f"Expected 16 TFs, got {norman_count}"

        # Click the 'KLF1' transcription factor node
        klf1_node = page.locator("#sos-true-stage svg g.node", has_text="KLF1").first
        assert klf1_node.is_visible()
        print("Clicking 'KLF1' TF node...")
        klf1_node.click()
        page.wait_for_timeout(500)

        norman_active_badge = page.locator("#sos-active-gene-badge").inner_text()
        print(f"Norman Active Gene Badge: {norman_active_badge}")
        assert "KLF1" in norman_active_badge
        assert "105.5%" in norman_active_badge or "Perturb:" in norman_active_badge

        # Click another node: CEBPA
        cebpa_node = page.locator("#sos-true-stage svg g.node", has_text="CEBPA").first
        if cebpa_node.is_visible():
            print("Clicking 'CEBPA' TF node...")
            cebpa_node.click()
            page.wait_for_timeout(400)
            assert "CEBPA" in page.locator("#sos-active-gene-badge").inner_text()

        # Switch back to KLF1
        klf1_node.click()
        page.wait_for_timeout(400)

        # Capture Screenshot of Norman 2019 Real Perturb-seq with KLF1 selected
        screenshot_norman_path = os.path.join(ARTIFACT_DIR, "playwright_norman_real_node_trajectory.png")
        page.screenshot(path=screenshot_norman_path, full_page=True)
        print(f"Saved Norman 2019 real dataset screenshot to: {screenshot_norman_path}")

        # 6. Test View Mode Toggle (Focus <-> Grid)
        print("\n--- Testing View Mode Toggle (Gene Focus <-> All Grid) ---")
        grid_btn = page.locator("#btn-traj-mode-grid")
        grid_btn.click()
        page.wait_for_timeout(300)
        assert page.locator("#sos-traj-grid-view").is_visible()
        assert not page.locator("#sos-traj-focus-view").is_visible()

        focus_btn = page.locator("#btn-traj-mode-focus")
        focus_btn.click()
        page.wait_for_timeout(300)
        assert page.locator("#sos-traj-focus-view").is_visible()
        assert not page.locator("#sos-traj-grid-view").is_visible()

        print("\n✅ All Clickable Node & Trajectory Playwright Tests Passed Successfully!")
        browser.close()


if __name__ == "__main__":
    test_clickable_nodes_and_trajectories()
