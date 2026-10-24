"""
Browser test for the page header: the title and the light/dark button share the first row, the five tab
buttons sit on the row beneath, and the page content starts below both.

All /api/** calls are stubbed (503) so only the static page is exercised.
"""

import pytest

pytest.importorskip("playwright.sync_api")

TABS = ["tab-wizard", "tab-sos", "tab-history", "tab-vcc", "tab-figures"]


@pytest.fixture
def open_page(browser, app_url):
    contexts = []

    def opener(width, height=900, color_scheme="light"):
        ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=color_scheme)
        contexts.append(ctx)
        page = ctx.new_page()
        page.route("**/api/**", lambda route: route.fulfill(status=503, content_type="application/json", body="{}"))
        page.goto(app_url, wait_until="domcontentloaded")
        page.wait_for_selector("nav.nav-tabs")
        return page

    yield opener
    for ctx in contexts:
        ctx.close()


def boxes(page):
    return {
        name: page.locator(selector).first.bounding_box()
        for name, selector in {
            "header": "header.topbar",
            "brand": ".brand",
            "title": ".brand-title h1",
            "button": "#theme-toggle-btn",
            "nav": "nav.nav-tabs",
            "main": "main.main-container",
        }.items()
    }


@pytest.mark.parametrize("width", [1500, 1100, 800, 600])
def test_title_and_theme_button_share_the_first_row_with_the_tabs_below(open_page, width):
    b = boxes(open_page(width))

    # first row: title block on the left, button at the right end, vertically overlapping
    assert b["button"]["x"] > b["brand"]["x"] + b["brand"]["width"] - 1, "button must sit to the right of the title block"
    assert b["button"]["x"] + b["button"]["width"] >= b["header"]["x"] + b["header"]["width"] - 40
    assert b["button"]["y"] < b["brand"]["y"] + b["brand"]["height"]
    assert b["button"]["y"] + b["button"]["height"] > b["brand"]["y"]

    # second row: the tabs start under both, and the button is not pushed down next to/under them
    assert b["nav"]["y"] >= b["brand"]["y"] + b["brand"]["height"] - 1
    assert b["nav"]["y"] >= b["button"]["y"] + b["button"]["height"] - 1

    # content starts below the whole header
    assert b["main"]["y"] >= b["header"]["y"] + b["header"]["height"] - 1


@pytest.mark.parametrize("width", [1500, 1100])
def test_tabs_fill_the_row_evenly_on_one_line(open_page, width):
    page = open_page(width)
    nav = page.locator("nav.nav-tabs").bounding_box()
    tabs = [page.locator(f"button[data-tab='{t}']").bounding_box() for t in TABS]

    assert len({round(t["y"]) for t in tabs}) == 1, "all five tabs must be on one line"
    assert [t["x"] for t in tabs] == sorted(t["x"] for t in tabs), "tab order is unchanged"
    widths = [t["width"] for t in tabs]
    assert max(widths) - min(widths) < 2, f"tabs should have equal widths, got {widths}"
    assert tabs[0]["x"] - nav["x"] < 12 and (nav["x"] + nav["width"]) - (tabs[-1]["x"] + tabs[-1]["width"]) < 12


def test_narrow_screens_keep_the_header_inside_the_viewport_and_scroll_the_tabs(open_page):
    page = open_page(420)
    b = boxes(page)
    for name in ("header", "button", "nav"):
        assert b[name]["x"] >= 0 and b[name]["x"] + b[name]["width"] <= 420 + 1, f"{name} sticks out of the viewport"
    assert page.evaluate("(() => { const n = document.querySelector('nav.nav-tabs'); return n.scrollWidth > n.clientWidth; })()")
    assert b["nav"]["y"] >= b["button"]["y"] + b["button"]["height"] - 1


@pytest.mark.parametrize("width, sticky", [(1500, True), (900, True), (420, False)])
def test_header_is_sticky_except_on_phones(open_page, width, sticky):
    page = open_page(width)
    position = page.evaluate("getComputedStyle(document.querySelector('header.topbar')).position")
    assert position == ("sticky" if sticky else "static")


def test_all_five_tabs_still_switch_panes(open_page):
    page = open_page(1500)
    labels = [page.locator(f"button[data-tab='{t}']").inner_text().strip()[:2] for t in TABS]
    assert labels == ["1.", "2.", "3.", "4.", "5."]
    for tab in TABS:
        page.click(f"button[data-tab='{tab}']")
        assert page.is_visible(f"#{tab}")
        assert page.evaluate(f"document.querySelector(\"button[data-tab='{tab}']\").classList.contains('active')")


def test_theme_button_still_toggles_the_theme(open_page):
    page = open_page(1500, color_scheme="light")
    before = page.evaluate("getComputedStyle(document.body).backgroundColor")
    label_before = page.inner_text("#theme-toggle-label")
    page.click("#theme-toggle-btn")
    page.wait_for_timeout(300)
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") != before
    assert page.inner_text("#theme-toggle-label") != label_before
