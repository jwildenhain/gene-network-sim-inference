"""
Browser test of the top navigation of the web studio (in-process server, see conftest.py):
the five tabs in order with short titles, switching between them, and a screenshot of the header.
"""

import pytest

pytest.importorskip("playwright.sync_api")

EXPECTED_TABS = [
    ("tab-wizard", "1. Simulation & Inference Studio"),
    ("tab-sos", "2. Real-World Benchmarks"),
    ("tab-history", "3. Previous Simulations Archive"),
    ("tab-vcc", "4. Virtual Cell Challenge"),
    ("tab-figures", "5. Overall Statistics"),
]


@pytest.fixture
def page(browser, app_url):
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    page.goto(app_url, wait_until="networkidle", timeout=30000)
    yield page
    context.close()


def test_tabs_are_in_order_with_short_titles(page):
    tabs = page.locator("nav.nav-tabs button.nav-tab")
    assert tabs.count() == len(EXPECTED_TABS)
    for i, (tab_id, title) in enumerate(EXPECTED_TABS):
        tab = tabs.nth(i)
        text = tab.inner_text().strip()
        assert tab.get_attribute("data-tab") == tab_id
        assert title in text
        assert not any(word in text for word in ("AtlasShift", "DREAM4", "SOS")), f"title should be short: {text!r}"


def test_every_tab_opens_its_section(page):
    for tab_id, title in EXPECTED_TABS:
        page.click(f"button[data-tab='{tab_id}']")
        page.wait_for_selector(f"#{tab_id}", state="visible")
        assert page.locator(f"button[data-tab='{tab_id}']").evaluate("el => el.classList.contains('active')"), title


def test_header_screenshot_can_be_captured(page, tmp_path):
    page.click("button[data-tab='tab-figures']")
    shot = tmp_path / "top_navigation.png"
    page.locator("header").screenshot(path=str(shot))
    assert shot.stat().st_size > 1000
