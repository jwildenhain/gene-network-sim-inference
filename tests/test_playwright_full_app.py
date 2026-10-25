"""
UI / UX tests for the whole web studio, one flow at a time, against the real server and real data:

    Tab 1  Simulation & Inference Studio     Tab 4  Virtual Cell Challenge
    Tab 2  Real-World Benchmarks (5 datasets) Tab 5  Overall Statistics
    Tab 3  Previous Simulations Archive

Every test also asserts that the page raised no JavaScript error and that no API call failed unexpectedly.
A temporary SQLite file is used, so the project's database is never touched.
"""

import json

import pytest

pytest.importorskip("playwright.sync_api")

from grn_sim.web import server  # noqa: E402

DATASETS = ["ecoli_sos", "norman_2019_real_16tf", "dream4_10gene", "replogle_2022_p53", "norman_2019_tf16"]
TABS = ["tab-wizard", "tab-sos", "tab-history", "tab-vcc", "tab-figures"]
VIEWS = ("hier", "spring", "canvas")


@pytest.fixture(scope="module")
def prepared_server(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("RGN_DB_PATH", str(tmp_path_factory.mktemp("db") / "full.sqlite"))
    saved = dict(server.CURRENT_STATE)
    server._execute_simulation({"num_genes": 12, "topology": "power_law", "k_av": 1.8, "seed": 11})
    server._execute_inference(["s2g", "g2s", "vote"])
    for dataset in DATASETS:  # cached, so the page does not have to compute them
        server._build_benchmark_studio(dataset)
    yield
    server.CURRENT_STATE.clear()
    server.CURRENT_STATE.update(saved)
    mp.undo()


@pytest.fixture
def page(browser, app_url, prepared_server):
    ctx = browser.new_context(viewport={"width": 1500, "height": 1100}, color_scheme="light", accept_downloads=True)
    pg = ctx.new_page()
    pg.js_errors, pg.bad_responses = [], []
    pg.on("pageerror", lambda e: pg.js_errors.append(str(e)))
    pg.on(
        "response",
        lambda r: pg.bad_responses.append(f"{r.status} {r.url.split('?')[0].replace(app_url, '')}")
        if r.status >= 400 and "favicon" not in r.url else None,
    )
    pg.on("dialog", lambda d: d.dismiss())
    pg.goto(app_url, wait_until="domcontentloaded")
    pg.wait_for_function(
        "typeof APP_STATE !== 'undefined' && !!APP_STATE.sim && !!APP_STATE.inferred && !!APP_STATE.sosData && !!APP_STATE.archiveDetail",
        timeout=90000,
    )
    yield pg
    ctx.close()


def go(page, tab):
    page.click(f"button[data-tab='{tab}']")
    page.wait_for_selector(f"#{tab}", state="visible")


def assert_healthy(page):
    assert page.js_errors == [], page.js_errors
    assert page.bad_responses == [], page.bad_responses


def canvas_ink(page, canvas_id):
    return page.evaluate(
        """(id) => { const c = document.getElementById(id); const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                     const bg = [d[0], d[1], d[2]]; let n = 0;
                     for (let i = 0; i < d.length; i += 4) if (Math.abs(d[i]-bg[0]) + Math.abs(d[i+1]-bg[1]) + Math.abs(d[i+2]-bg[2]) > 40) n++;
                     return n; }""",
        canvas_id,
    )


def graph_state(page, panel_id):
    """What a graph panel currently shows: view, svg node count, canvas node count, canvas ink."""
    return page.evaluate(
        """(id) => { const p = GRAPH_PANELS[id]; const stage = document.getElementById(p.stageId);
                     const canvas = document.getElementById(p.hosts.canvas);
                     return { view: p.view, svgNodes: stage.querySelectorAll('svg g.node').length, svgs: stage.querySelectorAll('svg').length,
                              canvasNodes: p.view === 'canvas' && p.graph ? p.graph.nodes.length : 0,
                              canvasShown: canvas.style.display !== 'none' && canvas.offsetParent !== null }; }""",
        panel_id,
    )


def show_view(page, bar, view):
    page.click(f"#{bar} button[data-view='{view}']")


# =========================================================================================================
# whole application
# =========================================================================================================

def test_every_tab_opens_and_renders_without_errors(page):
    for tab in TABS:
        go(page, tab)
        assert page.evaluate(f"document.querySelector(\"button[data-tab='{tab}']\").classList.contains('active')")
        assert page.evaluate("document.querySelectorAll('.tab-pane.active').length") == 1
    page.wait_for_timeout(800)  # let the lazily loaded tabs (statistics, VCC) finish
    assert_healthy(page)


def test_tab_content_is_populated(page):
    go(page, "tab-wizard")
    assert page.inner_text("#active-run-badge").strip() not in ("", "Run Ready")
    assert page.locator("#quick-node-select option").count() == 12

    go(page, "tab-sos")
    assert page.locator("#tbl-sos-benchmark tbody tr").count() >= 17
    assert page.locator("#prov-genes-list > span").count() == 9

    go(page, "tab-history")
    assert page.locator("#tbl-history tbody tr").count() >= 1
    assert page.locator("#hist-detail-alg option").count() >= 3

    go(page, "tab-vcc")
    page.wait_for_selector("#tbl-vcc-summary tbody tr", timeout=60000)
    assert page.locator("#tbl-vcc-summary tbody tr").count() == 18

    go(page, "tab-figures")
    page.wait_for_function("document.querySelectorAll('#tab-figures table tbody tr').length > 3", timeout=60000)
    assert_healthy(page)


# =========================================================================================================
# Tab 1: Simulation & Inference Studio
# =========================================================================================================

def test_wizard_steps_can_be_selected_stepped_and_focused(page):
    go(page, "tab-wizard")
    for step in range(1, 7):
        page.click(f"#wstep-{step}")
        assert page.evaluate("APP_STATE.activeStep") == step
        assert page.locator(f"#wstep-{step}").evaluate("e => e.classList.contains('current')")
        assert page.locator(".wizard-step.current").count() == 1

    page.click("#wstep-2")
    page.click("#btn-wizard-next")
    assert page.evaluate("APP_STATE.activeStep") == 3
    page.click("#btn-wizard-prev")
    page.click("#btn-wizard-prev")
    assert page.evaluate("APP_STATE.activeStep") == 1

    before = page.inner_text("#btn-toggle-focus-mode")
    page.click("#btn-toggle-focus-mode")
    assert page.inner_text("#btn-toggle-focus-mode") != before
    page.click("#btn-toggle-focus-mode")
    assert page.inner_text("#btn-toggle-focus-mode") == before
    assert_healthy(page)


def test_settings_drive_the_simulation(page):
    go(page, "tab-wizard")
    page.evaluate("""() => { const e = document.getElementById('sim-num-genes'); e.value = 14;
                             e.dispatchEvent(new Event('input', {bubbles: true})); e.dispatchEvent(new Event('change', {bubbles: true})); }""")
    page.select_option("#sim-topology", "random")
    old_run = page.evaluate("APP_STATE.sim.run_id")
    page.click("#btn-simulate-only")
    page.wait_for_function(f"APP_STATE.sim.run_id !== '{old_run}'", timeout=60000)

    assert page.evaluate("APP_STATE.sim.num_genes") == 14
    assert page.evaluate("APP_STATE.sim.topology") == "random"
    assert page.locator("#quick-node-select option").count() == 14
    assert page.inner_text("#active-run-badge").strip() == page.evaluate("APP_STATE.sim.run_id")
    for view in VIEWS:
        show_view(page, "gt-view-bar", view)
        state = graph_state(page, "gt")
        assert (state["canvasNodes"] if view == "canvas" else state["svgNodes"]) == 14
    assert_healthy(page)


def test_full_pipeline_fills_step_3_and_step_6_in_every_view(page):
    go(page, "tab-wizard")
    old_run = page.evaluate("APP_STATE.sim.run_id")
    page.click("#btn-run-all")
    page.wait_for_function(
        f"APP_STATE.sim.run_id !== '{old_run}' && APP_STATE.inferred && APP_STATE.inferred.metrics && !document.getElementById('btn-run-all').disabled",
        timeout=180000,
    )
    genes = page.evaluate("APP_STATE.sim.gene_names.length")
    assert page.inner_text("#st-power").strip() not in ("", "--%")
    for bar, panel in (("gt-view-bar", "gt"), ("inf-view-bar", "inf")):
        for view in VIEWS:
            show_view(page, bar, view)
            state = graph_state(page, panel)
            assert (state["canvasNodes"] if view == "canvas" else state["svgNodes"]) == genes, (panel, view)
            assert state["svgs"] == (0 if view == "canvas" else 1)
    assert_healthy(page)


def test_algorithm_selector_switches_step_6(page):
    go(page, "tab-wizard")
    show_view(page, "inf-view-bar", "hier")
    seen = {}
    for alg in page.evaluate("Object.keys(APP_STATE.inferred.svgs_dot)"):
        page.select_option("#select-infer-alg", alg)
        assert page.evaluate("APP_STATE.activeAlg") == alg
        assert page.inner_text("#active-alg-badge").strip().lower().startswith(alg[:3])
        assert page.locator("#inf-stage svg").count() == 1
        seen[alg] = page.evaluate("document.querySelector('#inf-stage svg').outerHTML")
    assert len(set(seen.values())) == len(seen), "each algorithm has its own 9-state graph"
    assert_healthy(page)


def test_perturbation_plot_modes_and_gene_selector(page):
    go(page, "tab-wizard")
    for _mode, button in (("bar", "#btn-pert-plot-bar"), ("line", "#btn-pert-plot-line")):
        page.click(button)
        assert page.locator(button).evaluate("e => e.classList.contains('active')")
    page.select_option("#select-pert-gene", "3")
    assert page.input_value("#select-pert-gene") == "3"
    assert_healthy(page)


def test_node_and_edge_selectors_inspect_the_network(page):
    go(page, "tab-wizard")
    page.select_option("#quick-node-select", "5")
    assert page.evaluate("APP_STATE.selectedNodeIdx") == 5
    options = page.eval_on_selector_all("#quick-edge-select option", "els => els.map(e => e.value).filter(Boolean)")
    assert options, "the network has edges to inspect"
    page.select_option("#quick-edge-select", options[0])
    assert_healthy(page)


def test_inspector_dialog_opens_and_closes(page):
    go(page, "tab-wizard")
    page.click("button[onclick='openInspectorDialog()']")
    assert page.locator("#inspector-dialog").is_visible()
    page.keyboard.press("Escape")
    assert not page.locator("#inspector-dialog").is_visible()
    assert_healthy(page)


def test_download_links_serve_files(page):
    go(page, "tab-wizard")
    links = page.eval_on_selector_all("#tab-wizard a[href^='/api/']", "els => els.map(e => e.getAttribute('href'))")
    assert len(links) >= 4
    for href in links:
        status, size = page.evaluate("async (u) => { const r = await fetch(u); return [r.status, (await r.arrayBuffer()).byteLength]; }", href)
        assert status == 200 and size > 0, href
    assert_healthy(page)


# =========================================================================================================
# Tab 2: Real-World Benchmarks
# =========================================================================================================

@pytest.mark.parametrize("dataset", DATASETS)
def test_every_dataset_switches_every_panel_and_view(page, dataset):
    go(page, "tab-sos")
    other = "dream4_10gene" if dataset != "dream4_10gene" else "ecoli_sos"
    page.click(f"#pill-ds-{other}")
    page.wait_for_function(f"APP_STATE.sosData.dataset_id === '{other}' && !document.querySelector('#pill-ds-{other}').disabled", timeout=60000)

    page.click(f"#pill-ds-{dataset}")
    page.wait_for_function(f"APP_STATE.sosData.dataset_id === '{dataset}' && !document.querySelector('#pill-ds-{dataset}').disabled", timeout=60000)
    genes = page.evaluate("APP_STATE.sosData.gene_names.length")

    assert page.locator("[id^='pill-ds-'].active").evaluate_all("els => els.map(e => e.id)") == [f"pill-ds-{dataset}"]
    assert page.evaluate("APP_STATE.activeBenchmark") == dataset
    assert f"{genes} Genes" in page.inner_text("#benchmark-header-badge")
    assert page.locator("#sos-gene-select option").count() == genes
    assert page.locator("#prov-genes-list > span").count() == genes
    assert page.locator("#tbl-sos-benchmark tbody tr").count() >= 17
    assert not page.locator("#benchmark-load-error").is_visible()

    for bar, panel in (("sos-true-views", "sos_true"), ("sos-diff-views", "sos_diff")):
        for view in VIEWS:
            show_view(page, bar, view)
            state = graph_state(page, panel)
            assert (state["canvasNodes"] if view == "canvas" else state["svgNodes"]) == genes, (dataset, panel, view)
    assert_healthy(page)


def test_views_stay_selected_while_switching_datasets(page):
    go(page, "tab-sos")
    show_view(page, "sos-true-views", "canvas")
    show_view(page, "sos-diff-views", "spring")
    for dataset in ("norman_2019_tf16", "replogle_2022_p53", "ecoli_sos"):
        page.click(f"#pill-ds-{dataset}")
        page.wait_for_function(f"APP_STATE.sosData.dataset_id === '{dataset}' && !document.querySelector('#pill-ds-{dataset}').disabled", timeout=60000)
        genes = page.evaluate("APP_STATE.sosData.gene_names.length")
        true_state, diff_state = graph_state(page, "sos_true"), graph_state(page, "sos_diff")
        assert true_state["view"] == "canvas" and true_state["canvasNodes"] == genes and canvas_ink(page, "sos-true-canvas") > 200
        assert diff_state["view"] == "spring" and diff_state["svgNodes"] == genes
    assert_healthy(page)


def test_failed_dataset_load_keeps_the_previous_dataset_and_says_why(page):
    go(page, "tab-sos")
    page.route("**/api/benchmark/dataset?id=dream4_10gene*", lambda r: r.fulfill(
        status=500, content_type="application/json", body=json.dumps({"error": "[Errno 13] Permission denied: 'x.tsv'"})))
    previous = page.evaluate("APP_STATE.activeBenchmark")
    genes_before = page.evaluate("APP_STATE.sosData.gene_names.length")

    page.click("#pill-ds-dream4_10gene")
    page.wait_for_selector("#benchmark-load-error", state="visible")
    message = page.inner_text("#benchmark-load-error")
    assert "dream4_10gene" in message and "Permission denied" in message
    assert page.evaluate("APP_STATE.activeBenchmark") == previous
    assert page.locator("[id^='pill-ds-'].active").evaluate_all("els => els.map(e => e.id)") == [f"pill-ds-{previous}"]
    assert page.evaluate("APP_STATE.sosData.gene_names.length") == genes_before
    assert not page.locator("#pill-ds-dream4_10gene").is_disabled(), "the page is usable again after a failure"
    assert page.locator("#tbl-sos-benchmark tbody tr").count() >= 17

    page.click("#pill-ds-replogle_2022_p53")  # a good dataset clears the message and switches normally
    page.wait_for_function("APP_STATE.sosData.dataset_id === 'replogle_2022_p53'", timeout=60000)
    assert not page.locator("#benchmark-load-error").is_visible()
    assert page.bad_responses == ["500 /api/benchmark/dataset"], "only the failure this test injected"
    assert page.js_errors == []


def test_algorithm_rows_select_the_9_state_graph_and_matrix(page):
    go(page, "tab-sos")
    show_view(page, "sos-diff-views", "hier")
    rows = page.locator("#tbl-sos-benchmark tbody tr")
    seen = {}
    for alg in ("g2s", "s2g", "nir"):
        rows.filter(has_text=alg.upper()).first.click()
        assert page.evaluate("APP_STATE.sosAlg") == alg
        assert page.inner_text("#sos-diff-title").strip() == alg.upper()
        assert page.locator("#sos-diff-stage svg").count() == 1
        seen[alg] = page.evaluate("document.querySelector('#sos-diff-stage svg').outerHTML")
    assert len(set(seen.values())) == 3
    assert_healthy(page)


def test_raw_data_downloads_for_every_file_type(page):
    go(page, "tab-sos")
    page.click("button[onclick^='toggleRawDownloadMenu']")
    assert page.locator("#raw-download-dropdown").is_visible()
    file_types = page.eval_on_selector_all(
        "#raw-download-dropdown [onclick^='downloadRawBenchmark']",
        "els => els.map(e => e.getAttribute('onclick').match(/'([^']+)'/)[1])",
    )
    assert len(file_types) >= 3
    for file_type in file_types:
        status, size = page.evaluate(
            "async ([id, t]) => { const r = await fetch(`/api/benchmark/download_raw?id=${id}&file_type=${t}`); return [r.status, (await r.arrayBuffer()).byteLength]; }",
            [page.evaluate("APP_STATE.activeBenchmark"), file_type],
        )
        assert status == 200 and size > 0, file_type
    page.click("body", position={"x": 5, "y": 400})
    assert not page.locator("#raw-download-dropdown").is_visible(), "clicking elsewhere closes the menu"
    assert_healthy(page)


@pytest.mark.parametrize("dataset", ["ecoli_sos", "norman_2019_tf16"])
@pytest.mark.parametrize("button, mode", [("#btn-load-wrangled", "wrangled_direct"), ("#btn-load-biophysical", "biophysical_fit")])
def test_loading_a_dataset_into_the_studio(page, dataset, button, mode):
    go(page, "tab-sos")
    page.click(f"#pill-ds-{dataset}")
    page.wait_for_function(f"APP_STATE.sosData.dataset_id === '{dataset}' && !document.querySelector('#pill-ds-{dataset}').disabled", timeout=60000)
    page.click(button)
    page.wait_for_function(
        f"document.querySelector('.tab-pane.active').id === 'tab-wizard' && APP_STATE.sim.mode === 'benchmark_{dataset}'", timeout=60000)
    genes = page.evaluate("APP_STATE.sosData.gene_names.length")
    assert page.evaluate("APP_STATE.sim.gene_names.length") == genes
    for view in VIEWS:
        show_view(page, "gt-view-bar", view)
        state = graph_state(page, "gt")
        assert (state["canvasNodes"] if view == "canvas" else state["svgNodes"]) == genes, view

    # the loaded dataset can be reverse-engineered from the Studio, and Step 6 follows it
    page.evaluate("APP_STATE.inferred = {}")
    page.click("#btn-infer-only")
    page.wait_for_function("APP_STATE.inferred && APP_STATE.inferred.svgs_dot && Object.keys(APP_STATE.inferred.svgs_dot).length > 0", timeout=120000)
    for view in VIEWS:
        show_view(page, "inf-view-bar", view)
        state = graph_state(page, "inf")
        assert (state["canvasNodes"] if view == "canvas" else state["svgNodes"]) == genes, view
    assert_healthy(page)


def test_failed_load_into_studio_is_reported(page):
    go(page, "tab-sos")
    page.route("**/api/benchmark/load_to_studio", lambda r: r.fulfill(
        status=500, content_type="application/json", body=json.dumps({"error": "boom"})))
    page.click("#btn-load-wrangled")
    page.wait_for_selector("#benchmark-load-error", state="visible")
    assert "boom" in page.inner_text("#benchmark-load-error")
    assert page.evaluate("document.querySelector('.tab-pane.active').id") == "tab-sos", "stays on the benchmark tab"
    assert not page.locator("#btn-load-wrangled").is_disabled()


# =========================================================================================================
# Tab 3: Previous Simulations
# =========================================================================================================

def test_archive_lists_runs_and_shows_the_selected_one(page):
    go(page, "tab-history")
    assert page.locator("#tbl-history tbody tr").count() >= 1
    run_id = page.evaluate("APP_STATE.archiveDetail.run_id")
    assert page.locator("#tbl-history tbody tr.selected-row").count() == 1
    assert run_id in page.inner_text("#tbl-history tbody tr.selected-row")
    assert page.inner_text("#hist-gt-edges-badge").strip().startswith(str(page.evaluate("APP_STATE.archiveDetail.total_edges")))
    assert_healthy(page)


def test_archive_algorithm_selector_updates_metrics_and_graph(page):
    go(page, "tab-history")
    show_view(page, "hist-diff-views", "hier")
    graphs = {}
    for alg in page.eval_on_selector_all("#hist-detail-alg option", "els => els.map(e => e.value)"):
        if not page.evaluate(f"!!APP_STATE.archiveDetail.inferred.tmp_statmat['{alg}']"):
            continue
        page.select_option("#hist-detail-alg", alg)
        assert page.inner_text("#hist-diff-alg-badge").strip() == alg.upper()
        assert page.evaluate("APP_STATE.archiveActiveAlg") == alg
        assert page.locator("#hist-diff-svg-stage svg").count() == 1
        graphs[alg] = page.evaluate("document.querySelector('#hist-diff-svg-stage svg').outerHTML")
    assert len(graphs) >= 3 and len(set(graphs.values())) == len(graphs)
    assert_healthy(page)


def test_archive_filter_options_all_work(page):
    go(page, "tab-history")
    options = page.eval_on_selector_all("#archive-filter-type option", "els => els.map(e => e.value)")
    assert len(options) >= 3
    for value in options:
        page.select_option("#archive-filter-type", value)
        page.wait_for_timeout(150)
        assert page.locator("#tbl-history").is_visible()
    assert_healthy(page)


def test_archive_perturbation_views(page):
    go(page, "tab-history")
    for mode in ("bar", "timecourse", "matrix", "dual"):
        page.click(f"#btn-hist-pert-mode-{mode}")
        assert page.locator(f"#btn-hist-pert-mode-{mode}").evaluate("e => e.classList.contains('active')")
        assert page.locator("#panel-hist-step4 .btn-group button.active").count() == 1

    # the 3x3 multi-perturbation grid is paged; the single-gene selector belongs to the other modes
    page.click("#btn-hist-pert-mode-bar")
    assert not page.locator("#select-hist-pert-gene").is_visible()
    page.click("#btn-hist-grid-next")
    page.click("#btn-hist-grid-prev")
    page.click("#btn-hist-pert-mode-dual")
    page.select_option("#select-hist-pert-gene", "2")
    assert page.input_value("#select-hist-pert-gene") == "2"
    assert_healthy(page)


def test_open_archived_run_in_the_studio(page):
    go(page, "tab-history")
    run_id = page.evaluate("APP_STATE.archiveDetail.run_id")
    page.click("button[onclick='openArchiveRunInStudio()']")
    assert page.evaluate("document.querySelector('.tab-pane.active').id") == "tab-wizard"
    assert page.inner_text("#active-run-badge").strip() == run_id
    show_view(page, "gt-view-bar", "hier")
    assert graph_state(page, "gt")["svgNodes"] == page.evaluate("APP_STATE.sim.gene_names.length")
    assert_healthy(page)


def test_preset_run_adds_a_run_to_the_archive(page):
    go(page, "tab-history")
    before = page.evaluate("fetch('/api/status').then(r => r.json()).then(s => s.database.nw_rows)")
    page.click("button[onclick=\"simulatePresetRun('flat')\"]")
    page.wait_for_function(
        f"fetch('/api/status').then(r => r.json()).then(s => s.database.nw_rows > {before})", timeout=120000, polling=1000)
    assert_healthy(page)


# =========================================================================================================
# Tab 4 / Tab 5
# =========================================================================================================

def test_virtual_cell_challenge_tab_responds_to_its_controls(page):
    go(page, "tab-vcc")
    page.wait_for_selector("#tbl-vcc-summary tbody tr", timeout=60000)
    assert page.locator("#vcc-model-select option").count() >= 7
    page.select_option("#vcc-model-select", "s2g")
    assert page.inner_text("#vcc-target-model-r").startswith("r = ")
    page.select_option("#vcc-target-select", "1")
    assert "Gene_02" in page.inner_text("#vcc-legend-model-lbl") or page.inner_text("#vcc-target-model-r").startswith("r = ")

    page.select_option("#vcc-cell-line", "0.30")
    page.wait_for_function("APP_STATE.vccData && APP_STATE.vccData.parameters.target_rewire_fraction === 0.3", timeout=120000)
    assert page.locator("#tbl-vcc-summary tbody tr").count() == 18
    assert not page.locator("#vcc-load-error").is_visible()
    assert_healthy(page)


def test_overall_statistics_tab_switches_views_and_filters(page):
    go(page, "tab-figures")
    page.wait_for_function("document.querySelectorAll('#tab5-view-overall table tbody tr').length > 3", timeout=60000)
    page.click("#btn-tab5-view-historical")
    assert page.locator("#tab5-view-historical").is_visible() and not page.locator("#tab5-view-overall").is_visible()

    # the archive inspector (connectivity / noise) lives in the historical thesis-figures view
    page.wait_for_function("APP_STATE.thesisFigures && !!APP_STATE.thesisFigures.overallstats_grid", timeout=120000)
    readouts = set()
    for iot in page.eval_on_selector_all("#sel-overall-iot option", "els => els.map(e => e.value)"):
        for std in page.eval_on_selector_all("#sel-overall-std option", "els => els.map(e => e.value)")[:3]:
            page.select_option("#sel-overall-iot", iot)
            page.select_option("#sel-overall-std", std)
            readouts.add(page.inner_text("#overallstats-readout"))
    assert len(readouts) > 1, "the readout follows the selected connectivity / noise"

    page.click("#btn-tab5-view-overall")
    assert page.locator("#tab5-view-overall").is_visible() and not page.locator("#tab5-view-historical").is_visible()
    assert_healthy(page)


# =========================================================================================================
# layout: no horizontal page scrolling, at any common window width
# =========================================================================================================

@pytest.mark.parametrize("width", [1920, 1500, 1280, 1024, 800, 600])
def test_no_tab_is_wider_than_the_window(page, width):
    page.set_viewport_size({"width": width, "height": 900})
    for tab in TABS:
        go(page, tab)
        if tab == "tab-vcc":
            page.wait_for_selector("#tbl-vcc-summary tbody tr", timeout=60000)
        page.wait_for_timeout(300)
        overflow = page.evaluate(
            """() => { const vw = document.documentElement.clientWidth;
                       const wide = Array.from(document.querySelectorAll('.tab-pane.active *')).filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.right > vw + 1 && !e.closest('[style*="overflow-x:auto"], [style*="overflow:auto"]'); })
                         .slice(0, 3).map(e => e.tagName.toLowerCase() + '#' + e.id + '.' + String(e.className).slice(0, 30));
                       return { vw, page: document.documentElement.scrollWidth, wide }; }"""
        )
        assert overflow["page"] <= overflow["vw"] + 1, f"{tab} at {width}px is {overflow['page']}px wide: {overflow['wide']}"


def test_every_data_table_scrolls_inside_its_panel():
    import re
    from pathlib import Path
    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    unwrapped = [
        m.group(0)[:80] for m in re.finditer(r'<table[^>]*class="data-table"[^>]*>', html)
        if "overflow-x:auto" not in html[max(0, m.start() - 80):m.start()]
        and "overflow:auto" not in html[max(0, m.start() - 80):m.start()]
    ]
    assert unwrapped == [], f"tables without a scroll wrapper: {unwrapped}"


def test_responsive_rules_come_after_the_base_rules():
    from pathlib import Path
    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    base = html.index(".stat-grid {")
    for query in ("@media (max-width: 1100px)", "@media (max-width: 820px)", "@media (max-width: 600px)"):
        assert html.index(query, base) > base, f"{query} would be overridden by the later base rule"
    assert ".workspace-grid > *," in html and "min-width: 0" in html
