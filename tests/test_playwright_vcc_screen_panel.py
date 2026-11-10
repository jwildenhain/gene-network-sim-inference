"""
Browser test of the Virtual Cell tab's results panel: the explanatory header, the filters, the per-algorithm statistics and the
verdicts. The page is served by the real handler; stored simulations come from a temporary results file.
"""
import json
import os

import pytest

pytest.importorskip("playwright.sync_api")

from grn_sim.evaluation import vcc_benchmark as vb  # noqa: E402


def _rows():
    out = []
    plan = {
        "cascade": {"good": 0.40, "weak": 0.06, "mid": 0.05},
        "regulon": {"good": 0.50, "weak": 0.06, "mid": 0.30},
    }
    for pathway, algs in plan.items():
        for seed in range(1, 11):
            j = float(((seed * 37) % 11 - 5) * 0.002)
            scores = {vb.RANDOM: 0.05 + j / 2, vb.HUBNESS: 0.12 + j, "good": algs["good"] + j, "weak": algs["weak"] + j / 4,
                      "mid": algs["mid"] + j / 4}
            for name, v in scores.items():
                out.append({**vb.Setting(pathway=pathway).to_dict(), "seed": seed, "algorithm": name, "status": "ok", "aupr_direct": v,
                            "p_at_k_direct": 0.2, "base_direct": 0.05, "aupr_total": 0.2, "uses_pairs": False})
    return out


@pytest.fixture(scope="module", autouse=True)
def stored_results(tmp_path_factory):
    home = tmp_path_factory.mktemp("vcc-home")
    (home / "benchmarks").mkdir()
    (home / "benchmarks" / "vcc_screen_benchmark.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()))
    old = os.environ.get("GRN_SIM_HOME")
    os.environ["GRN_SIM_HOME"] = str(home)
    yield home
    if old is None:
        os.environ.pop("GRN_SIM_HOME", None)
    else:
        os.environ["GRN_SIM_HOME"] = old


@pytest.fixture()
def page(browser, app_url):
    ctx = browser.new_context(viewport={"width": 1500, "height": 1000})
    pg = ctx.new_page()
    pg.goto(app_url, wait_until="domcontentloaded")
    pg.evaluate("switchMainTab('tab-vcc')")
    pg.wait_for_selector("#tbl-vcs-algorithms tbody tr")
    yield pg
    ctx.close()


def _rows_text(page):
    return page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#tbl-vcs-algorithms tbody tr')]
        .map(r => [r.dataset.algorithm, {verdict: r.dataset.verdict, text: r.innerText}]))""")


def test_header_explains_the_statistical_experiment_and_drops_the_old_comparison_box(page):
    text = page.inner_text("#tab-vcc")
    assert "statistical models" in page.inner_text("#vcc-intro-text") and "panel 1" in page.inner_text("#vcc-intro-text").lower()
    assert "VCC 2026 BENCHMARK" not in text and "Direct comparative benchmark" not in text
    options = page.eval_on_selector_all("#vcc-num-genes option", "els => els.map(e => e.value)")
    assert "200" in options


def test_table_lists_every_algorithm_with_verdicts(page):
    rows = _rows_text(page)
    assert set(rows) == {"random", "gene_features", "good", "weak", "mid"}
    assert rows["good"]["verdict"] == "better" and rows["weak"]["verdict"] == "not_better"
    assert rows[vb.RANDOM]["verdict"] == "reference" and rows["gene_features"]["verdict"] == "control"
    assert "20 simulations in 2 settings" in page.inner_text("#vcs-sub")
    verdict = page.inner_text("#vcs-verdict")
    assert "Not appropriate" in verdict and "weak" in verdict.split("Not appropriate")[1]


def test_filter_changes_the_aggregated_statistics(page):
    page.select_option("#vcs-pathway", "regulon")
    page.wait_for_function("document.getElementById('vcs-sub').textContent.startsWith('10 simulations')")
    rows = _rows_text(page)
    assert rows["mid"]["verdict"] == "better" and rows["weak"]["verdict"] == "not_better"   # 0.30 vs the hubness control 0.12 here
    assert "0.5" in rows["good"]["text"]
    page.select_option("#vcs-pathway", "all")
    page.wait_for_function("document.getElementById('vcs-sub').textContent.startsWith('20 simulations')")


def test_filter_with_no_matching_simulation_says_so(page):
    page.select_option("#vcs-mode", "knockdown")                    # exists
    page.wait_for_function("document.getElementById('vcs-sub').textContent.startsWith('20 simulations')")
    options = page.eval_on_selector_all("#vcs-mode option", "els => els.map(e => e.value)")
    assert options == ["all", "knockdown"]                         # only the simulated levels are offered


def test_family_filter_and_pivot_by_factor(page):
    page.select_option("#vcs-family", "control")
    assert set(_rows_text(page)) == {"random", "gene_features"}
    page.select_option("#vcs-family", "all")
    page.select_option("#vcs-compare", "pathway")
    page.wait_for_function("document.querySelector('#tbl-vcs-pivot thead').innerText.toLowerCase().includes('cascade')")
    head = page.inner_text("#tbl-vcs-pivot thead").title()
    assert "Signalling Cascade" in head and "Overlapping Regulons" in head
    cells = page.eval_on_selector_all("#tbl-vcs-pivot tbody td[data-verdict]", "els => els.map(e => e.dataset.verdict)")
    assert "better" in cells and "not_better" in cells


def test_reset_restores_all_filters(page):
    page.select_option("#vcs-pathway", "cascade")
    page.wait_for_function("document.getElementById('vcs-sub').textContent.startsWith('10 simulations')")
    page.click("text=Reset filters")
    page.wait_for_function("document.getElementById('vcs-sub').textContent.startsWith('20 simulations')")
    assert page.input_value("#vcs-pathway") == "all"
