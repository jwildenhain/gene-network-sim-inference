"""Accessibility: what axe-core cannot see (text alternatives, keyboard operation, names) and the colour tokens."""
import pytest

pytest.importorskip("playwright.sync_api")

LIGHT, DARK = "light", "dark"


def _open(browser, app_url, scheme=LIGHT, width=1280):
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
    page.goto(app_url, wait_until="networkidle")
    page.evaluate(f"document.documentElement.setAttribute('data-theme', '{scheme}')")
    page.wait_for_timeout(500)
    return page


def _show_all_tabs(page):
    """Visit every tab once so lazily built content (tables, charts) exists."""
    page.evaluate("switchMainTab('tab-sos')")
    page.wait_for_selector("#tbl-sos-benchmark tbody tr", state="attached", timeout=60000)
    page.evaluate("switchMainTab('tab-history')")
    page.wait_for_selector("#tbl-history tbody tr", state="attached", timeout=60000)
    page.evaluate("switchMainTab('tab-figures'); switchTab5View('overall')")
    page.wait_for_selector("canvas#canvas-overall-size-f1", state="attached", timeout=60000)
    page.wait_for_timeout(900)                                                 # the accessibility pass is debounced


def test_tabs_are_a_real_tablist_that_arrow_keys_navigate(app_url, browser):
    page = _open(browser, app_url)
    assert page.locator("nav[role=tablist] > [role=tab]").count() == 6                          # from the first paint
    assert page.locator('[role=tab][tabindex="0"]').count() == 1                                # one tab stop, roving
    first = page.locator("#tabbtn-tab-wizard")
    first.focus()
    page.keyboard.press("ArrowRight")
    assert page.evaluate("document.querySelector('.nav-tab.active').dataset.tab") == "tab-sos"
    assert page.evaluate("document.activeElement.dataset.tab") == "tab-sos"
    assert page.locator("#tabbtn-tab-sos").get_attribute("aria-selected") == "true"
    page.keyboard.press("End")
    assert page.evaluate("document.querySelector('.nav-tab.active').dataset.tab") == "tab-jobs"
    page.keyboard.press("ArrowRight")                                                           # wraps around
    assert page.evaluate("document.querySelector('.nav-tab.active').dataset.tab") == "tab-wizard"
    page.keyboard.press("ArrowLeft")
    assert page.evaluate("document.querySelector('.nav-tab.active').dataset.tab") == "tab-jobs"
    assert page.locator('[role=tab][tabindex="0"]').count() == 1
    for pane in page.locator("section.tab-pane").all():
        assert pane.get_attribute("role") == "tabpanel" and pane.get_attribute("aria-labelledby")
    page.close()


def test_every_chart_has_a_text_alternative(app_url, browser):
    page = _open(browser, app_url)
    _show_all_tabs(page)
    bad = page.evaluate("""() => [...document.querySelectorAll('canvas')]
        .filter(c => c.getAttribute('role') !== 'img' || !(c.getAttribute('aria-label') || '').trim()).map(c => c.id || c.className)""")
    assert bad == []
    # a line chart also carries its numbers for a screen reader
    desc = page.evaluate("""() => { const c = document.getElementById('canvas-overall-size-f1');
        const d = document.getElementById(c.getAttribute('aria-describedby')); return [c.getAttribute('aria-label'), d && d.textContent, d && d.className]; }""")
    assert "Line chart" in desc[0] and "N=12" in desc[0]
    assert "S2G" in desc[1] and "sr-only" in desc[2]
    page.close()


def test_clickable_non_buttons_work_from_the_keyboard(app_url, browser):
    page = _open(browser, app_url)
    _show_all_tabs(page)
    unreachable = page.evaluate("""() => [...document.querySelectorAll('[onclick], tbody tr')]
        .filter(el => !['BUTTON','A','INPUT','SELECT','TEXTAREA','SUMMARY','OPTION','LABEL'].includes(el.tagName) && !el.closest('svg'))
        .filter(el => (el.hasAttribute('onclick') || typeof el.onclick === 'function') && el.getAttribute('tabindex') === null)
        .map(el => el.tagName + '#' + el.id + '.' + String(el.className).slice(0, 25))""")
    assert unreachable == []
    # an archive row opens with Enter, like a click
    page.evaluate("switchMainTab('tab-history')")
    row = page.locator("#tbl-history tbody tr").first
    row.focus()
    page.keyboard.press("Enter")
    page.wait_for_function("document.querySelector('#tbl-history tbody tr').classList.contains('selected-row')", timeout=10000)
    page.close()


def test_tables_have_names_and_header_scopes(app_url, browser):
    page = _open(browser, app_url)
    _show_all_tabs(page)
    res = page.evaluate("""() => ({
        unnamed: [...document.querySelectorAll('table')].filter(t => !t.querySelector('caption') && !t.getAttribute('aria-label') && !t.getAttribute('aria-labelledby')).map(t => t.id),
        noScope: [...document.querySelectorAll('thead th')].filter(th => !th.getAttribute('scope')).length,
        tables: document.querySelectorAll('table').length })""")
    assert res["tables"] >= 10 and res["unnamed"] == [] and res["noScope"] == 0
    page.close()


def test_scrollable_regions_can_be_focused(app_url, browser):
    page = _open(browser, app_url, width=375)                                                   # narrow: tables scroll sideways
    check = """() => [...document.querySelectorAll('div, section, svg')].filter(el => {
        if (el === document.body || el.id === 'main-content') return false;
        if (el.scrollWidth <= el.clientWidth + 1 && el.scrollHeight <= el.clientHeight + 1) return false;
        const cs = getComputedStyle(el);
        if (!/(auto|scroll)/.test(cs.overflowX + cs.overflowY)) return false;
        if (el.querySelector('a[href], button, input, select, textarea, [tabindex]')) return false;
        return el.getAttribute('tabindex') === null || !(el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')); })
        .map(el => (el.id || el.tagName) + '.' + String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className).slice(0, 20))"""
    seen = {}
    page.evaluate("switchMainTab('tab-wizard')")
    page.wait_for_selector("#tbl-alg-comparison tbody tr", state="attached", timeout=60000)
    page.wait_for_timeout(800)
    seen["wizard"] = page.evaluate(check)
    page.evaluate("switchMainTab('tab-sos')")
    page.wait_for_selector("#tbl-sos-benchmark tbody tr", state="attached", timeout=60000)
    page.wait_for_timeout(800)
    seen["benchmarks"] = page.evaluate(check)
    page.evaluate("switchMainTab('tab-history')")
    page.wait_for_selector("#tbl-history tbody tr", state="attached", timeout=60000)
    page.locator("#tbl-history tbody tr").first.click()                                          # opens a run's detail view
    page.wait_for_selector("#tbl-hist-alg-comparison tbody tr", state="attached", timeout=60000)
    page.wait_for_timeout(1000)
    seen["archive detail"] = page.evaluate(check)
    assert {k: v for k, v in seen.items() if v} == {}
    page.close()


def test_keyboard_focus_is_visible_and_the_skip_link_works(app_url, browser):
    page = _open(browser, app_url)
    page.keyboard.press("Tab")                                                                  # the first stop is the skip link
    assert page.evaluate("document.activeElement.className") == "skip-link"
    assert page.evaluate("document.activeElement.getBoundingClientRect().top") >= 0            # visible, not off screen
    page.keyboard.press("Enter")
    assert page.evaluate("document.activeElement.id") == "main-content"
    page.locator("#btn-run-all").focus()
    page.keyboard.press("Shift+Tab")
    ring = page.evaluate("(() => { const cs = getComputedStyle(document.activeElement); return [cs.outlineStyle, parseFloat(cs.outlineWidth)]; })()")
    assert ring[0] != "none" and ring[1] >= 2                                                   # a drawn focus ring, 2 px or more
    page.close()


def test_the_theme_toggle_names_its_action(app_url, browser):
    page = _open(browser, app_url, scheme=LIGHT)
    page.evaluate("updateThemeToggleButtonUI()")
    assert page.get_attribute("#theme-toggle-btn", "aria-label") == "Switch to dark mode"
    page.click("#theme-toggle-btn")
    page.wait_for_timeout(300)
    assert page.get_attribute("#theme-toggle-btn", "aria-label") == "Switch to light mode"
    page.close()


@pytest.mark.parametrize("scheme", [LIGHT, DARK])
def test_text_colours_meet_4_5_to_1_on_their_backgrounds(app_url, browser, scheme):
    """The accent colours are only used as text through the ink tokens, which hold 4.5:1 on solid backgrounds."""
    page = _open(browser, app_url, scheme=scheme)
    _show_all_tabs(page)
    failures = page.evaluate("""() => {
      const lum = c => { const [r, g, b] = c.match(/[\\d.]+/g).slice(0, 3).map(Number).map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
      const ratio = (a, b) => { const x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
      const bgOf = el => { for (let p = el; p; p = p.parentElement) { const cs = getComputedStyle(p); if (cs.backgroundImage !== 'none') return null;
          const m = cs.backgroundColor.match(/[\\d.]+/g); if (m && (m.length < 4 || Number(m[3]) > 0.9)) return cs.backgroundColor; } return 'rgb(255, 255, 255)'; };
      const out = [];
      document.querySelectorAll('.tab-pane.active *, .tab-pane.active').forEach(el => {
        if (!el.childNodes.length || ![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 2)) return;
        const r = el.getBoundingClientRect(); if (!r.width || !r.height || getComputedStyle(el).visibility === 'hidden') return;
        const bg = bgOf(el); if (!bg) return;                       // gradient / image behind: axe handles those
        const cs = getComputedStyle(el), big = parseFloat(cs.fontSize) >= 24 || (parseFloat(cs.fontSize) >= 18.66 && Number(cs.fontWeight) >= 700);
        const need = big ? 3 : 4.5, got = ratio(cs.color, bg);
        if (got < need - 0.01) out.push(el.tagName + '.' + String(el.className).slice(0, 20) + ' ' + got.toFixed(2));
      });
      return out.slice(0, 10); }""")
    assert failures == []
    page.close()
