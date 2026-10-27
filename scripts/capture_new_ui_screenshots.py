import os
import tempfile
import time
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("GRN_SIM_URL", "http://localhost:8085")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR") or os.path.join(tempfile.gettempdir(), "grn-sim-artifacts")
DOCS_IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

def capture_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1640, "height": 1050})
        page = context.new_page()

        print(f"Connecting to {APP_URL}...")
        page.goto(APP_URL, wait_until="networkidle")
        page.wait_for_selector("#btn-run-all")
        time.sleep(1)

        # 1. Dark Mode: Tab 1 Studio with Dynamics Modeling Engine & 12 Algorithms
        page.evaluate("document.documentElement.setAttribute('data-theme', 'dark')")
        page.wait_for_timeout(400)
        shot1 = os.path.join(ARTIFACT_DIR, "12_tab1_multiparadigm_studio_dark.png")
        page.screenshot(path=shot1, full_page=False)
        print(f"Captured: {shot1}")

        # 2. Light Mode: Tab 1 Studio
        page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
        page.wait_for_timeout(400)
        shot2 = os.path.join(ARTIFACT_DIR, "13_tab1_multiparadigm_studio_light.png")
        page.screenshot(path=shot2, full_page=False)
        print(f"Captured: {shot2}")

        # 3. Tab 2: Real-World Benchmarks Studio (E. coli SOS 9-gene default)
        page.click("button[data-tab='tab-sos']")
        page.wait_for_timeout(800)
        shot3 = os.path.join(ARTIFACT_DIR, "14_tab2_benchmarks_studio_sos.png")
        page.screenshot(path=shot3, full_page=False)
        print(f"Captured: {shot3}")

        # 4. Tab 2: Switch to Norman 2019 Perturb-seq 16-TF Circuit
        page.click("#pill-ds-norman_2019_tf16")
        page.wait_for_timeout(1000)
        shot4 = os.path.join(ARTIFACT_DIR, "15_tab2_benchmarks_norman_tf16.png")
        page.screenshot(path=shot4, full_page=False)
        print(f"Captured: {shot4}")

        # 5. Click "Load Dataset into Tab 1 Studio"
        page.click("#btn-load-bench-studio")
        page.wait_for_timeout(1000)
        shot5 = os.path.join(ARTIFACT_DIR, "16_tab1_loaded_norman_tf16_studio.png")
        page.screenshot(path=shot5, full_page=False)
        print(f"Captured: {shot5}")

        # Copy key shots to docs/screenshots
        for src, dst_name in [
            (shot1, "12_tab1_multiparadigm_studio_dark.png"),
            (shot2, "13_tab1_multiparadigm_studio_light.png"),
            (shot3, "14_tab2_benchmarks_studio_sos.png"),
            (shot4, "15_tab2_benchmarks_norman_tf16.png"),
            (shot5, "16_tab1_loaded_norman_tf16_studio.png"),
        ]:
            dst_path = os.path.join(DOCS_IMG_DIR, dst_name)
            with open(src, "rb") as fsrc, open(dst_path, "wb") as fdst:
                fdst.write(fsrc.read())

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture_screenshots()
