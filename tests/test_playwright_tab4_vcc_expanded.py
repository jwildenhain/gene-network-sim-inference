"""Playwright end-to-end verification for the expanded Tab 4 Virtual Cell Simulation Engine."""

import os
import sys
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = "/home/jw/.gemini/antigravity/brain/2034be68-5fcf-4315-bc81-67eeb7bf53ed"
SCREENSHOT_PATH = os.path.join(ARTIFACT_DIR, "vcc_expanded_simulation_engine.png")
DETAIL_SCREENSHOT_PATH = os.path.join(ARTIFACT_DIR, "vcc_perturbation_and_epistasis_detail.png")


def test_tab4_vcc_expanded():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 2000})
        page = context.new_page()

        # Collect console errors
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        print("Navigating to http://localhost:8085 ...")
        page.goto("http://localhost:8085", wait_until="networkidle")

        # Click on Tab 4 (Virtual Cell Challenge & Simulation Engine)
        tab4_btn = page.locator('button[data-tab="tab-vcc"]')
        assert tab4_btn.is_visible(), "Tab 4 button button[data-tab='tab-vcc'] not found"
        print("Clicking Tab 4...")
        tab4_btn.click()
        page.wait_for_timeout(1500)

        # Ensure Tab 4 container is active
        tab4_pane = page.locator("#tab-vcc")
        assert tab4_pane.is_visible(), "Tab 4 pane #tab-vcc is not active/visible"

        # Check if benchmark table has rows, if not, click #btn-run-vcc
        rows = page.locator("#tbl-vcc-summary tbody tr")
        row_count = rows.count()
        print(f"Initial VCC table row count: {row_count}")

        if row_count < 18:
            print("Triggering #btn-run-vcc ...")
            page.locator("#btn-run-vcc").click()
            # Wait for benchmark to complete (up to 35s)
            page.wait_for_selector("#tbl-vcc-summary tbody tr:nth-child(18)", timeout=40000)
            page.wait_for_timeout(1500)
            row_count = page.locator("#tbl-vcc-summary tbody tr").count()

        print(f"Verified VCC table row count: {row_count}")
        assert row_count == 18, f"Expected 18 algorithm rows in VCC table, got {row_count}"

        # Verify algorithm names in table
        table_text = page.locator("#tbl-vcc-summary").inner_text()
        core_algos = ["VOTE", "CELLBOX", "S2G", "G2S", 
                      "RIDGE_CV", "CPA_LITE", "NETDECONV", "NIR", "FAST", "ARACHNE", 
                      "CAUSALGRN", "DMDGRN", "DEEPRIG", "SFINN", "NOTEARS", "ATLASSHIFT", "BY_CHANCE"]
        for algo in core_algos:
            assert algo in table_text, f"Algorithm '{algo}' missing from VCC summary table!"

        print("All 18 algorithms verified in summary table.")

        # Verify Section 3: Interactive In Silico Perturbation Simulator
        target_select = page.locator("#vcc-target-select")
        assert target_select.is_visible(), "#vcc-target-select not visible"
        target_options = target_select.locator("option").all_inner_texts()
        print(f"Target gene options ({len(target_options)}): {target_options[:5]}...")
        assert len(target_options) >= 12, f"Expected at least 12 target gene options, got {len(target_options)}"

        model_select = page.locator("#vcc-model-select")
        assert model_select.is_visible(), "#vcc-model-select not visible"
        model_options = model_select.locator("option").all_inner_texts()
        print(f"Model options: {model_options}")
        assert len(model_options) >= 5, f"Expected multiple response models, got {len(model_options)}"

        # Check canvas and metrics
        canvas_profile = page.locator("#canvas-vcc-target-profile")
        assert canvas_profile.is_visible(), "Target profile canvas not visible"

        model_r = page.locator("#vcc-target-model-r").inner_text()
        atlas_r = page.locator("#vcc-target-atlas-r").inner_text()
        ood_badge = page.locator("#vcc-target-ood-badge").inner_text()
        print(f"Default Profile Metrics: Model Pearson r={model_r}, AtlasShift Pearson r={atlas_r}, Status={ood_badge}")
        assert len(model_r) > 0 and model_r != "--", "Model Pearson r metric is empty"

        # Change target to another gene and model
        print("Changing target gene to Gene_03 and model to DMD-GRN...")
        target_select.select_option(index=2)
        model_select.select_option(value="dmdgrn")
        page.wait_for_timeout(500)

        model_r_new = page.locator("#vcc-target-model-r").inner_text()
        print(f"Updated Profile Metrics for Gene_03 / DMD-GRN: Model Pearson r={model_r_new}")
        assert len(model_r_new) > 0 and model_r_new != "--", "Updated metric is empty"

        # Verify Section 4: Combinatorial Epistasis Screen
        canvas_epistasis = page.locator("#canvas-vcc-epistasis")
        assert canvas_epistasis.is_visible(), "Epistasis canvas not visible"

        synergy_list = page.locator("#vcc-top-synergy-list")
        assert synergy_list.is_visible(), "Synergy list container not visible"
        synergy_items = page.locator("#vcc-top-synergy-list > div")
        synergy_count = synergy_items.count()
        print(f"Synergy list item count: {synergy_count}")
        assert synergy_count >= 3, f"Expected top synergy items, got {synergy_count}"
        top_synergy_text = synergy_items.first.inner_text()
        print(f"Top synergy entry: {top_synergy_text.replace(chr(10), ' | ')}")

        # Verify Section 4: Virtual Single-Cell Sequencer & PCA
        canvas_pca = page.locator("#canvas-vcc-pca")
        assert canvas_pca.is_visible(), "Single-cell PCA canvas not visible"

        pca_cells = page.locator("#vcc-pca-cells-badge").inner_text()
        pca_var = page.locator("#vcc-pca-var-exp").inner_text()
        print(f"Virtual Single-Cell Info: Cells={pca_cells}, Variance Explained={pca_var}")
        assert "cells" in pca_cells.lower(), f"Expected cell count badge, got: {pca_cells}"
        assert "PC1" in pca_var, f"Expected PC1 variance explained, got: {pca_var}"

        # Verify Section 5: Rewiring Decay Curves
        canvas_rewire = page.locator("#canvas-vcc-rewire")
        assert canvas_rewire.is_visible(), "Rewiring decay canvas not visible"

        # Check export button
        btn_export = page.locator("#btn-vcc-dl-h5ad")
        assert btn_export.is_visible(), "Export h5ad button not visible"

        # Check for any unexpected console errors
        filtered_errors = [e for e in console_errors if "favicon" not in e and "404" not in e]
        print(f"Console errors: {filtered_errors}")
        assert len(filtered_errors) == 0, f"Unexpected JS console errors: {filtered_errors}"

        # Capture full-page screenshot
        print(f"Saving full screenshot to {SCREENSHOT_PATH} ...")
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)

        # Capture zoomed detail of simulator and epistasis
        print(f"Saving detail screenshot to {DETAIL_SCREENSHOT_PATH} ...")
        page.locator("#vcc-target-select").scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=DETAIL_SCREENSHOT_PATH)

        browser.close()
        print("Playwright test completed successfully!")


if __name__ == "__main__":
    test_tab4_vcc_expanded()
