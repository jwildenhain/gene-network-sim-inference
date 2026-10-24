"""
UI / UX tests for the shared "Choose your view" bar of every network graph in the web studio:

    1. Step 3  ground-truth network            4. Real-World Benchmarks: known regulatory network
    2. Step 6  9-state differential graph      5. Real-World Benchmarks: 9-state differential reconstruction
                                               6. Previous Simulations: archived ground truth + archived 9-state graph

Each bar offers the same labels (Force Canvas / Hierarchical / Spring; "Graph file" where a DOT file exists),
the Hierarchical view has a Layered | Grid layout toggle, and the choice is remembered.

A simulation + inference is created in-process (temporary SQLite file), the benchmark is served from the real
server, and every other API call is stubbed, so nothing touches the project's database.
"""

import pytest

pytest.importorskip("playwright.sync_api")

from grn_sim.web import server  # noqa: E402

GRID_BOX = "0.0 0.0 560.0 420.0"  # viewBox of the grid version

# panel -> (tab to open, view bar id, stage id, labels in the bar, svg host(s) of the hierarchical view)
PANELS = {
    "gt": ("tab-wizard", "gt-view-bar", "gt-stage", ["Force Canvas", "Hierarchical", "Spring", "Graph file"]),
    "inf": ("tab-wizard", "inf-view-bar", "inf-stage", ["Force Canvas", "Hierarchical", "Spring"]),
    "sos_true": ("tab-sos", "sos-true-views", "sos-true-stage", ["Force Canvas", "Hierarchical", "Spring"]),
    "sos_diff": ("tab-sos", "sos-diff-views", "sos-diff-stage", ["Force Canvas", "Hierarchical", "Spring"]),
    "hist_gt": ("tab-history", "hist-gt-views", "hist-gt-svg-stage", ["Force Canvas", "Hierarchical", "Spring"]),
    "hist_diff": ("tab-history", "hist-diff-views", "hist-diff-svg-stage", ["Force Canvas", "Hierarchical", "Spring"]),
}
ALLOWED_API = ("/api/state", "/api/benchmark/dataset", "/api/history")


@pytest.fixture(scope="module")
def prepared_server(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("RGN_DB_PATH", str(tmp_path_factory.mktemp("db") / "views.sqlite"))
    saved = dict(server.CURRENT_STATE)
    # 12 genes: the Previous Simulations list only offers networks with N >= 12
    server._execute_simulation({"num_genes": 12, "topology": "power_law", "k_av": 1.8, "seed": 11})
    server._execute_inference(["s2g", "vote"])
    server._build_benchmark_studio("ecoli_sos")  # cached, so the page does not recompute it
    yield
    server.CURRENT_STATE.clear()
    server.CURRENT_STATE.update(saved)
    mp.undo()


@pytest.fixture
def page(browser, app_url, prepared_server):
    ctx = browser.new_context(viewport={"width": 1500, "height": 1100}, color_scheme="light")
    pg = ctx.new_page()
    pg.js_errors = []
    pg.on("pageerror", lambda e: pg.js_errors.append(str(e)))

    def route(r):
        path = r.request.url.split("?")[0].replace(app_url, "")
        if path in ALLOWED_API or path.startswith("/api/history"):
            r.continue_()
        else:
            r.fulfill(status=503, content_type="application/json", body="{}")

    pg.route("**/api/**", route)
    pg.goto(app_url, wait_until="domcontentloaded")
    pg.wait_for_function(
        "typeof APP_STATE !== 'undefined' && !!APP_STATE.sim && !!APP_STATE.inferred && !!APP_STATE.inferred.svgs_dot"
        " && !!APP_STATE.sosData && !!APP_STATE.archiveDetail",
        timeout=60000,
    )
    yield pg
    ctx.close()


def open_panel(page, panel):
    tab, bar, stage, labels = PANELS[panel]
    page.click(f"button[data-tab='{tab}']")
    page.wait_for_selector(f"#{bar}", state="visible")
    return bar, stage, labels


def view_button(page, bar, view):
    return page.locator(f"#{bar} button[data-view='{view}']")


def stage_svgs(page, stage):
    return page.locator(f"#{stage} svg")


def stage_view_box(page, stage):
    return page.evaluate(f"document.querySelector('#{stage} svg').getAttribute('viewBox')")


def layout_buttons_active(page):
    """The active layout of every toggle on the page (they all follow one shared choice)."""
    return page.evaluate(
        "Array.from(document.querySelectorAll('.hier-layout-toggle')).map(t => "
        "Array.from(t.querySelectorAll('button.active')).map(b => b.dataset.layout).join())"
    )


def canvas_has_drawing(page, canvas_id):
    return page.evaluate(
        """(id) => { const c = document.getElementById(id); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                     const bg = [d[0], d[1], d[2]]; let diff = 0;
                     for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i]-bg[0]) + Math.abs(d[i+1]-bg[1]) + Math.abs(d[i+2]-bg[2]) > 40) diff++;
                     return diff; }""",
        canvas_id,
    )


# ---------------------------------------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("panel", PANELS)
def test_every_panel_offers_the_same_labelled_choices(page, panel):
    bar, stage, expected = open_panel(page, panel)
    texts = page.eval_on_selector_all(f"#{bar} button[data-view]", "els => els.map(e => e.textContent.trim())")
    assert texts == expected
    assert page.inner_text(f"#{bar} .gvb-label").strip() == "Choose your view:"
    for label in texts:
        assert "(dot)" not in label and "neato" not in label.lower() and label != "DOT"


# ---------------------------------------------------------------------------------------------------------
# view switching + layered/grid
# ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("panel", PANELS)
def test_hierarchical_view_switches_between_layered_and_grid(page, panel):
    bar, stage, _ = open_panel(page, panel)
    view_button(page, bar, "hier").click()

    toggle = page.locator(f"#{bar} .hier-layout-toggle")
    assert toggle.is_visible()
    assert stage_svgs(page, stage).count() == 1
    assert layout_buttons_active(page) == ["layered"] * len(PANELS)
    layered_box = stage_view_box(page, stage)
    assert layered_box != GRID_BOX

    toggle.locator("button[data-layout='grid']").click()
    assert stage_svgs(page, stage).count() == 1, "switching layout replaces the graph instead of stacking another"
    assert stage_view_box(page, stage) == GRID_BOX
    assert layout_buttons_active(page) == ["grid"] * len(PANELS)

    toggle.locator("button[data-layout='layered']").click()
    assert stage_view_box(page, stage) == layered_box
    assert layout_buttons_active(page) == ["layered"] * len(PANELS)


@pytest.mark.parametrize("panel", PANELS)
def test_spring_and_force_canvas_views(page, panel):
    bar, stage, _ = open_panel(page, panel)

    view_button(page, bar, "hier").click()
    hier_svg = page.evaluate(f"document.querySelector('#{stage} svg').outerHTML")

    view_button(page, bar, "spring").click()
    assert not page.locator(f"#{bar} .hier-layout-toggle").is_visible(), "layout choice only applies to Hierarchical"
    assert stage_svgs(page, stage).count() == 1
    assert page.evaluate(f"document.querySelector('#{stage} svg').outerHTML") != hier_svg
    assert page.locator(f"#{bar} button[data-view='spring']").evaluate("e => e.classList.contains('active')")

    view_button(page, bar, "canvas").click()
    assert stage_svgs(page, stage).count() == 0
    canvas_id = page.evaluate(f"GRAPH_PANELS['{panel}'].hosts.canvas")
    assert page.locator(f"#{canvas_id}").is_visible()
    assert canvas_has_drawing(page, canvas_id) > 200, "the canvas view paints the network"

    view_button(page, bar, "hier").click()
    assert page.locator(f"#{bar} .hier-layout-toggle").is_visible()
    assert not page.locator(f"#{canvas_id}").is_visible()
    assert page.js_errors == []


def test_only_one_view_button_is_active_per_bar(page):
    for panel in PANELS:
        bar, _, _ = open_panel(page, panel)
        for view in ("hier", "spring", "canvas"):
            view_button(page, bar, view).click()
            active = page.eval_on_selector_all(f"#{bar} button[data-view].active", "els => els.map(e => e.dataset.view)")
            assert active == [view]


def test_graph_file_view_shows_the_dot_text(page):
    bar, stage, _ = open_panel(page, "gt")
    view_button(page, bar, "source").click()
    text = page.inner_text("#gt-dot-source")
    assert text.lstrip().startswith("digraph") and "->" in text
    assert stage_svgs(page, stage).count() == 0
    assert not page.locator(f"#{bar} .hier-layout-toggle").is_visible()
    assert "Graphviz" in view_button(page, bar, "source").get_attribute("title")


def test_legacy_button_ids_still_work(page):
    open_panel(page, "gt")
    page.click("#btn-gt-dot")
    assert page.locator("#gt-svg-dot svg").is_visible()
    page.click("#btn-gt-neato")
    assert page.locator("#gt-svg-neato svg").is_visible()
    page.click("#btn-gt-source")
    assert page.locator("#gt-dot-source").is_visible()
    page.click("#btn-gt-canvas")
    assert page.locator("#canvas-gt-network").is_visible()
    page.click("#btn-inf-dot")
    assert page.locator("#inf-svg-dot svg").is_visible()


# ---------------------------------------------------------------------------------------------------------
# shared preference
# ---------------------------------------------------------------------------------------------------------

def test_layout_choice_is_shared_by_all_panels_and_remembered(page):
    bar, stage, _ = open_panel(page, "sos_true")
    view_button(page, bar, "hier").click()
    page.locator(f"#{bar} .hier-layout-toggle button[data-layout='grid']").click()

    for panel in ("gt", "inf", "sos_diff", "hist_gt", "hist_diff"):
        b, s, _ = open_panel(page, panel)
        view_button(page, b, "hier").click()
        assert stage_view_box(page, s) == GRID_BOX, f"{panel} should follow the shared Grid choice"

    page.reload(wait_until="domcontentloaded")
    page.wait_for_function("typeof APP_STATE !== 'undefined' && !!APP_STATE.sosData && !!APP_STATE.sim", timeout=60000)
    b, s, _ = open_panel(page, "sos_true")
    assert stage_view_box(page, s) == GRID_BOX, "the choice survives a reload"
    assert layout_buttons_active(page) == ["grid"] * len(PANELS)


def test_theme_change_repaints_canvas_views(page):
    bar, stage, _ = open_panel(page, "sos_true")
    view_button(page, bar, "canvas").click()
    corner = "id => Array.from(document.getElementById(id).getContext('2d').getImageData(2, 2, 1, 1).data)"
    before = page.evaluate(corner, "sos-true-canvas")
    page.click("#theme-toggle-btn")
    page.wait_for_timeout(300)
    assert page.evaluate(corner, "sos-true-canvas") != before


# ---------------------------------------------------------------------------------------------------------
# interaction
# ---------------------------------------------------------------------------------------------------------

def click_svg_node(page, stage, gene):
    page.locator(f"#{stage} svg g.node", has_text=gene).first.click()


@pytest.mark.parametrize("layout", ["layered", "grid"])
def test_clicking_a_node_in_the_benchmark_graphs_selects_the_gene(page, layout):
    page.evaluate(f"setHierLayout('{layout}')")
    bar, stage, _ = open_panel(page, "sos_true")
    view_button(page, bar, "hier").click()
    genes = page.evaluate("APP_STATE.sosData.gene_names")
    click_svg_node(page, stage, genes[1])
    assert page.evaluate("APP_STATE.activeSosGene") == genes[1]
    assert page.locator(f"#{stage} svg g.node.active-node").count() == 1

    # the same selection is shown in the other graph and survives a view switch
    diff_bar, diff_stage, _ = open_panel(page, "sos_diff")
    view_button(page, diff_bar, "hier").click()
    click_svg_node(page, diff_stage, genes[2])
    assert page.evaluate("APP_STATE.activeSosGene") == genes[2]
    open_panel(page, "sos_true")
    assert page.locator(f"#{stage} svg g.node.active-node").count() == 1
    assert page.js_errors == []


def test_canvas_click_selects_a_gene_and_highlights_it(page):
    bar, stage, _ = open_panel(page, "sos_true")
    view_button(page, bar, "canvas").click()
    genes = page.evaluate("APP_STATE.sosData.gene_names")
    # position of node 3 inside the (CSS-scaled) canvas, so Playwright scrolls to it and clicks there
    x, y = page.evaluate(
        "(() => { const p = GRAPH_PANELS.sos_true.positions[3]; const c = document.getElementById('sos-true-canvas');"
        " const r = c.getBoundingClientRect(); return [p.x * r.width / c.width, p.y * r.height / c.height]; })()"
    )
    assert page.evaluate("APP_STATE.activeSosGene") != genes[3], "precondition: another gene is selected first"
    fingerprint = (
        "(() => { const c = document.getElementById('sos-true-canvas');"
        " const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;"
        " let h = 0; for (let i = 0; i < d.length; i += 37) h = (h * 31 + d[i]) | 0; return h; })()"
    )
    before = page.evaluate(fingerprint)
    page.locator("#sos-true-canvas").click(position={"x": x, "y": y})
    assert page.evaluate(fingerprint) != before, "the selected node is drawn highlighted"
    assert page.evaluate("APP_STATE.activeSosGene") == genes[3]
    assert page.evaluate("GRAPH_PANELS.sos_true.selectedIdx()") == 3
    assert page.js_errors == []


def test_clicking_a_node_in_step_3_and_step_6_inspects_it(page):
    for panel, selector in (("gt", "#quick-node-select"), ("inf", "#quick-node-select")):
        bar, stage, _ = open_panel(page, panel)
        view_button(page, bar, "hier").click()
        names = page.evaluate("APP_STATE.sim.gene_names")
        click_svg_node(page, stage, names[4])
        assert page.input_value(selector) == "4"
        assert page.locator("#inspector-dialog").is_visible(), "a node click opens the inspector"
        page.keyboard.press("Escape")
        assert not page.locator("#inspector-dialog").is_visible()
        page.evaluate("inspectNode(0, false)")


def test_archive_views_reflect_the_selected_run_and_algorithm(page):
    bar, stage, _ = open_panel(page, "hist_diff")
    view_button(page, bar, "hier").click()
    before = page.evaluate(f"document.querySelector('#{stage} svg').outerHTML")
    other = page.evaluate("Object.keys(APP_STATE.archiveDetail.inferred.svgs_dot).find(a => a !== APP_STATE.archiveActiveAlg)")
    page.evaluate(f"renderArchiveDetailAlg('{other}')")
    assert page.inner_text("#hist-diff-alg-badge").strip() == other.upper()
    assert page.evaluate(f"document.querySelector('#{stage} svg').outerHTML") != before
    assert stage_svgs(page, stage).count() == 1
    assert page.js_errors == []


# ---------------------------------------------------------------------------------------------------------
# layout quality
# ---------------------------------------------------------------------------------------------------------

def test_hierarchical_layered_view_puts_regulators_above_their_targets(page):
    bar, stage, _ = open_panel(page, "sos_true")
    view_button(page, bar, "hier").click()
    page.evaluate("setHierLayout('layered')")
    result = page.evaluate(
        """() => {
          const g = APP_STATE.sosData.true_graph, names = APP_STATE.sosData.gene_names;
          const y = {};
          document.querySelectorAll('#sos-true-stage svg g.node').forEach(n => {
            const name = n.querySelector('title')?.textContent.trim();
            const e = n.querySelector('ellipse');
            if (name && e) y[name] = parseFloat(e.getAttribute('cy'));
          });
          const links = g.links.filter(l => l.source_idx !== l.target_idx);
          const key = l => l.source_idx + '>' + l.target_idx;
          const present = new Set(links.map(key));
          const mutualPairs = links.filter(l => present.has(l.target_idx + '>' + l.source_idx)).length / 2;
          const upward = links.filter(l => y[names[l.source_idx]] >= y[names[l.target_idx]]).length;
          return { total: links.length, upward, mutualPairs, rows: new Set(Object.values(y)).size };
        }"""
    )
    # every mutual pair forces one edge upwards; beyond that, regulators belong above their targets
    assert result["upward"] <= result["mutualPairs"] + 2, result
    assert result["rows"] >= 3, result
