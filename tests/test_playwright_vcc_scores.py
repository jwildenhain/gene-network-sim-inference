"""Browser test of the Virtual Cell Challenge 2026 score tables of the Virtual Cell tab (served by the real handler, real
computations on small networks)."""

import os

import pytest

pytest.importorskip("playwright.sync_api")

from grn_sim.evaluation import vcc2026 as v  # noqa: E402

HEADERS = ["method", "model", "overall", "pds sc", "mse sc", "jac sc", "nmae sc", "fid sc", "reach sc"]   # case-folded


@pytest.fixture(scope="module")
def page(browser, app_url, tmp_path_factory):
    old = os.environ.get("GRN_SIM_HOME")
    os.environ["GRN_SIM_HOME"] = str(tmp_path_factory.mktemp("vcc-home"))     # no stored results, nothing written to the user's
    ctx = browser.new_context(viewport={"width": 1500, "height": 1000})
    pg = ctx.new_page()
    pg.set_default_timeout(240_000)
    pg.goto(app_url, wait_until="domcontentloaded")
    pg.evaluate("switchMainTab('tab-vcc')")
    yield pg
    ctx.close()
    if old is None:
        os.environ.pop("GRN_SIM_HOME", None)
    else:
        os.environ["GRN_SIM_HOME"] = old


def _table(page, table_id):
    return page.evaluate(f"""() => {{
        const t = document.querySelector('#{table_id}');
        return {{headers: [...t.querySelectorAll('thead th')].map(h => h.innerText.replace(/[\\u2195\\u2191\\u2193]/g, '').trim().replace(/\\s+/g, ' ').toLowerCase()),
                 rows: [...t.querySelectorAll('tbody tr')].map(r => ({{model: r.dataset.model, kind: r.dataset.kind, text: r.innerText}}))}};
    }}""")


def test_cross_atlas_table_2b_shows_the_leaderboard_columns_and_both_anchors(page):
    page.wait_for_selector("#tbl-vcc-score tbody tr")
    t = _table(page, "tbl-vcc-score")
    assert t["headers"] == HEADERS
    kinds = {r["model"]: r["kind"] for r in t["rows"]}
    assert kinds["baseline"] == "anchor" and kinds["replicate"] == "anchor" and kinds["truth_means"] == "oracle"
    anchor_rows = [r for r in t["rows"] if r["kind"] == "anchor"]
    assert "0.000" in anchor_rows[0]["text"] or "0.000" in anchor_rows[1]["text"]
    assert "1.000" in anchor_rows[0]["text"] or "1.000" in anchor_rows[1]["text"]
    note = page.inner_text("#vcc-score-note")
    assert "held-out perturbation" in note and "replicate" in note and "VCC2026_SCORING.md" in note
    assert len(t["rows"]) >= len(v.METRICS) + 2


def test_simulate_once_renders_the_challenge_scores_per_generator(page):
    page.fill("#vcs-new-genes", "20")
    page.fill("#vcs-new-cells", "30")
    page.select_option("#vcs-new-generator", "linear_sem")
    page.select_option("#vcs-new-pathway", "cascade")
    page.click("#btn-vcs-new")
    page.wait_for_selector("#tbl-vcs-new-vcc-0 tbody tr")
    t = _table(page, "tbl-vcs-new-vcc-0")
    assert t["headers"] == HEADERS
    models = [r["model"] for r in t["rows"]]
    assert {"baseline", "replicate", "random", "self_prior", "truth_means"} <= set(models)
    assert {"baseline", "replicate"} == set(models[:2])                  # the anchors lead the table
    status = page.inner_text("#vcs-new-status")
    assert "non-standard cell count" in status
    note = page.inner_text("#vcs-new-vcc-note-0")
    assert "quarter of the perturbed genes" in note and "structure + linear refit" in note
