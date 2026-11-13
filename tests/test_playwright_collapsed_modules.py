"""The collapsed (module) view of a large network: per-module accuracy tallies, the accuracy ring and the module matrix."""
import json
import random

import pytest

pytest.importorskip("playwright.sync_api")

N = 90
STATUSES = ("TP_CORRECT", "TP_SIGN_ERR", "FP", "FN")


def _graphs(classified=True):
    """A 90-gene truth with three dense blocks and a comparison graph that classifies links of a made-up prediction."""
    rng = random.Random(5)
    block = lambda i: i // 30  # noqa: E731
    pairs = set()
    for i in range(N):
        for j in range(N):
            if i != j and rng.random() < (0.12 if block(i) == block(j) else 0.004):
                pairs.add((i, j))
    nodes = [{"id": f"G{i + 1}", "index": i, "label": f"G{i + 1}"} for i in range(N)]
    truth = {"nodes": nodes, "links": [{"source": f"G{i + 1}", "target": f"G{j + 1}", "source_idx": i, "target_idx": j, "sign": 1}
                                       for i, j in sorted(pairs)]}
    links = []
    for k, (i, j) in enumerate(sorted(pairs)):
        link = {"source": f"G{i + 1}", "target": f"G{j + 1}", "source_idx": i, "target_idx": j, "sign": 1,
                "color": "#10b981", "arrowhead": "normal", "style": "solid"}
        if classified:
            link["status"] = STATUSES[k % 4]
        links.append(link)
    return truth, {"nodes": nodes, "links": links}


@pytest.fixture
def page(browser, app_url):
    pg = browser.new_page(viewport={"width": 1400, "height": 1000})
    pg.goto(app_url, wait_until="networkidle")
    yield pg
    pg.close()


def _mount(page, truth, comparison):
    page.evaluate("""([truth, comparison]) => {
        const host = document.createElement('div'); host.id = 'cg-test-host'; document.body.appendChild(host);
        window.__p = { id: 'cg-test', view: 'hier', clusterBasis: () => truth, selectedIdx: () => -1 };
        window.__data = { graph: comparison };
        window.renderGraphPanel = () => renderCollapsedPanel(window.__p, host, window.__data);   // the page's re-render hook
        renderCollapsedPanel(window.__p, host, window.__data);
    }""", [truth, comparison])


def test_tallies_account_for_every_classified_link_and_scores_follow_the_definition(page):
    truth, comparison = _graphs()
    out = page.evaluate("""([truth, comparison]) => {
        const cl = clusterGraph(truth);
        const t = moduleTallies(comparison, cl);
        const sum = (a, b) => ({ tp: a.tp + b.tp, sign: a.sign + b.sign, fp: a.fp + b.fp, fn: a.fn + b.fn });
        let cells = { tp: 0, sign: 0, fp: 0, fn: 0 };
        t.cell.forEach(r => r.forEach(c => { cells = sum(cells, c); }));
        return { k: t.k, total: t.total, cells, within: t.within, between: t.between,
                 scores: tallyScores({ tp: 3, sign: 1, fp: 4, fn: 2 }), empty: tallyScores({ tp: 0, sign: 0, fp: 0, fn: 0 }),
                 wrong: tallyWrongShare({ tp: 6, sign: 1, fp: 2, fn: 1 }) };
    }""", [truth, comparison])
    n = len(comparison["links"])
    assert sum(out["total"].values()) == n
    assert out["cells"] == out["total"]                                                    # the matrix holds every link once
    assert {k: out["within"][k] + out["between"][k] for k in out["total"]} == out["total"]
    assert 2 <= out["k"] <= 25
    assert out["scores"] == {"precision": 0.5, "recall": pytest.approx(4 / 6), "links": 10}
    assert out["empty"]["precision"] is None and out["empty"]["recall"] is None
    assert out["wrong"] == pytest.approx(0.4)


def test_overview_shows_accuracy_rings_scores_and_the_matrix_toggle(page):
    truth, comparison = _graphs()
    _mount(page, truth, comparison)
    assert page.locator("#cg-test-host .cg-node").count() >= 2
    assert page.locator("#cg-test-host .cg-ring").count() >= page.locator("#cg-test-host .cg-node").count()
    info = page.inner_text("#cg-test-host .cg-info")
    assert "modules" in info and "links inside modules: precision" in info and "links between modules: precision" in info
    assert page.locator("#cg-test-host [data-cg-mode='matrix']").count() == 1

    page.click("#cg-test-host [data-cg-mode='matrix']")
    k = page.evaluate("clusterGraph(window.__p.clusterBasis()).clusters.length")
    assert page.locator("#cg-test-host .cg-cell").count() == k * k
    assert page.locator("#cg-test-host .cg-matrix").count() == 1
    title = page.locator("#cg-test-host .cg-cell title").first.text_content()
    assert "correct" in title and "extra" in title

    page.locator("#cg-test-host .cg-cell").first.click()                                     # a cell opens the regulator module
    assert "All modules" in page.inner_text("#cg-test-host .cg-info")
    page.click("#cg-test-host [data-cg-back]")
    assert page.locator("#cg-test-host .cg-cell").count() == k * k                           # back in the matrix, the mode is kept


def test_a_graph_without_classified_links_has_no_rings_and_no_matrix(page):
    truth, comparison = _graphs(classified=False)
    _mount(page, truth, comparison)
    assert page.locator("#cg-test-host .cg-node").count() >= 2
    assert page.locator("#cg-test-host .cg-ring").count() == 0
    assert page.locator("#cg-test-host [data-cg-mode]").count() == 0
