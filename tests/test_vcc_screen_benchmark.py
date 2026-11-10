"""Virtual Cell tab: pathway topologies, new gene-selection rules, double-perturbation columns, the benchmark statistics and API."""
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from grn_sim.core.topology import PATHWAY_TOPOLOGIES, TopologyType, generate_topology
from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.perturbseq.design import ScreenDesign, select_perturbed
from grn_sim.perturbseq.pipeline import build_network, run_screen
from grn_sim.web import vcc_screen_view as view


# -- pathway topologies --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("topology", [t.value for t in PATHWAY_TOPOLOGIES])
@pytest.mark.parametrize("n", [30, 200])
def test_pathway_topologies_have_the_requested_size_and_density(topology, n):
    topo = generate_topology(n, k_av=2.0, net_type=TopologyType(topology), ratio=0.7, subdiv=3, rng=np.random.default_rng(4))
    A = topo.adjacency
    assert A.shape == (n, n) and not np.any(np.diag(A))
    assert set(np.unique(A)) <= {-1, 0, 1}
    assert 0.7 * 2.0 * n <= np.count_nonzero(A) <= 1.4 * 2.0 * n
    assert np.any(A > 0) and np.any(A < 0)


@pytest.mark.parametrize("topology", [t.value for t in PATHWAY_TOPOLOGIES])
def test_pathway_topologies_are_deterministic_per_seed(topology):
    a = generate_topology(60, 2.0, TopologyType(topology), rng=np.random.default_rng(9)).adjacency
    b = generate_topology(60, 2.0, TopologyType(topology), rng=np.random.default_rng(9)).adjacency
    c = generate_topology(60, 2.0, TopologyType(topology), rng=np.random.default_rng(10)).adjacency
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_feedforward_network_is_acyclic_and_rich_in_loops():
    A = generate_topology(80, 2.2, TopologyType.FEEDFORWARD, ratio=0.7, rng=np.random.default_rng(2)).adjacency != 0
    reach = A.copy()
    for _ in range(8):
        reach = reach | (reach.astype(int) @ A.astype(int) > 0)
    assert not np.any(np.diag(reach)), "a feed-forward network must have no directed cycle"
    loops = int(np.sum((A.astype(int) @ A.astype(int)) * A))          # x -> y -> z together with x -> z
    assert loops >= 20


def test_regulon_network_is_two_layers_deep_with_hub_tfs():
    A = generate_topology(100, 2.0, TopologyType.REGULON, rng=np.random.default_rng(3)).adjacency != 0
    out = A.sum(axis=1)
    assert out.max() >= 10 and (out == 0).sum() > 60            # a few TFs regulate many; most genes regulate nothing


@pytest.mark.parametrize("preset", ["cascade", "feedforward", "lineage", "regulon"])
def test_pathway_presets_reach_a_steady_state_and_can_be_screened(preset):
    net = build_network(preset, 40, seed=5)
    run = run_screen(net, mode="knockdown", cells_per_condition=15, seed=5)
    assert run.screen.steady.all()
    assert "effective" in run.screen.labels


# -- gene selection ------------------------------------------------------------------------------------------------------

def test_hub_selection_takes_the_best_connected_regulators():
    net = build_network("replogle_like", 60, seed=1)
    genes, info = select_perturbed(net, ScreenDesign.partial(10, "hubs"), seed=1)
    out = (net.topology.adjacency != 0).sum(axis=1)
    assert info["rule"] == "hubs" and len(genes) == 10
    assert out[genes].min() >= np.sort(out)[-12]                  # the top ten, allowing for ties


def test_expressed_selection_takes_the_most_expressed_genes():
    net = build_network("replogle_like", 60, seed=1)
    genes, info = select_perturbed(net, ScreenDesign.partial(10, "expressed"), seed=1)
    level = net.steady_state_unperturbed
    assert info["rule"] == "expressed" and level[genes].min() >= np.sort(level)[-10] - 1e-12


def test_selection_by_effect_is_not_scored_against_genome_targets():
    run = run_screen(build_network("replogle_like", 50, seed=2), cells_per_condition=15, n_perturbed=20,
                     perturbed_selection="hubs", seed=2)
    rows = run.realism()["report"]
    assert any(r.get("comparable") is False for r in rows)


# -- double perturbations as extra experiments ---------------------------------------------------------------------------

def test_pair_columns_carry_both_perturbed_genes():
    from grn_sim.evaluation import screen_benchmark as sb

    run = run_screen(build_network("replogle_like", 30, seed=3), mode="knockdown", cells_per_condition=25, pairs=12, seed=3)
    ace = sb.screen_inputs(run, "cells", 3)
    X, U = vb.pair_columns(run.adata, ace)
    assert X.shape == U.shape and X.shape[0] == 30 and 0 < X.shape[1] <= 12
    assert np.all((U != 0).sum(axis=0) <= 2) and np.any((U != 0).sum(axis=0) == 2)


def test_simulation_scores_every_algorithm_and_uses_doubles_only_where_supported():
    rows = vb.run_simulation(vb.Setting(n_genes=30, pairs=10, cells=25), seed=3)
    by = {r["algorithm"]: r for r in rows}
    assert set(by) == {a.name for a in vb.algorithm_library()}
    assert all(r["status"] == "ok" for r in rows), [r["status"] for r in rows if r["status"] != "ok"]
    assert by["s2g"]["uses_pairs"] and not by["inspre"]["uses_pairs"] and by["s2g"]["n_pair_columns"] > 0
    assert by["s2g"]["aupr_direct"] > 2 * by["s2g"]["base_direct"]


def test_partial_screens_run_for_the_regression_form_algorithms():
    rows = vb.run_simulation(vb.Setting(n_genes=30, selection="effective", cells=25), seed=4)
    assert all(r["status"] == "ok" for r in rows)
    assert {r["n_perturbed"] for r in rows} == {rows[0]["n_perturbed"]} and rows[0]["n_perturbed"] < 30


# -- statistics ----------------------------------------------------------------------------------------------------------

def _rows(scores, seeds=10, **setting):
    """Synthetic rows: ``scores`` maps algorithm -> function(seed) -> AUPR. Base rate 0.05, random ~0.05."""
    out = []
    for seed in range(1, seeds + 1):
        for name, f in scores.items():
            out.append({**vb.Setting(**setting).to_dict(), "seed": seed, "algorithm": name, "status": "ok", "aupr_direct": f(seed),
                        "p_at_k_direct": 0.1, "base_direct": 0.05, "aupr_total": 0.1, "uses_pairs": False})
    return out


def test_verdicts_separate_working_marginal_and_useless_algorithms():
    rng = np.random.default_rng(0)
    def jitter(s):
        return float(np.random.default_rng(s).normal(0, 0.005))

    rows = _rows({
        vb.RANDOM: lambda s: 0.05 + jitter(s),
        vb.HUBNESS: lambda s: 0.12 + jitter(s),
        "good": lambda s: 0.40 + jitter(s),
        "marginal": lambda s: 0.12 + 0.03 + jitter(s) * 0.2,            # beats chance, level with the hubness control
        "useless": lambda s: 0.05 + 0.5 * jitter(s),
        "anti": lambda s: 0.01 + 0.2 * abs(jitter(s)),
    })
    agg = vb.aggregate(rows)
    by = {a["algorithm"]: a for a in agg["algorithms"]}
    assert agg["n_simulations"] == 10
    assert by["good"]["verdict"] == "better" and by["good"]["lift"] == pytest.approx(8.0, rel=0.05)
    assert by["useless"]["verdict"] == "not_better"
    assert by["anti"]["verdict"] == "worse"
    assert by[vb.RANDOM]["verdict"] == "reference" and by[vb.HUBNESS]["verdict"] == "control"
    assert by["marginal"]["verdict"] in ("marginal", "better") and by["marginal"]["q_hub"] >= 0 and rng is not None
    assert by["good"]["win_rate"] == 1.0 and len(by["good"]["values"]) == 10


def test_filters_select_the_matching_simulations_only():
    rows = _rows({vb.RANDOM: lambda s: 0.05, "a": lambda s: 0.3}, pathway="cascade") + \
        _rows({vb.RANDOM: lambda s: 0.05, "a": lambda s: 0.1}, pathway="regulon")
    both = vb.aggregate(rows)
    one = vb.aggregate(rows, {"pathway": "regulon", "mode": "all"})
    assert both["n_settings"] == 2 and both["n_simulations"] == 20
    assert one["n_simulations"] == 10 and {a["algorithm"]: a["mean"] for a in one["algorithms"]}["a"] == pytest.approx(0.1)
    assert vb.aggregate(rows, {"pathway": "lineage"})["algorithms"] == []


def test_failed_algorithms_are_missing_not_scored_as_chance():
    rows = _rows({vb.RANDOM: lambda s: 0.05, "a": lambda s: 0.3})
    for r in rows:
        if r["algorithm"] == "a" and r["seed"] <= 4:
            r["status"], r["aupr_direct"] = "error: ValueError", None
    a = {x["algorithm"]: x for x in vb.aggregate(rows)["algorithms"]}["a"]
    assert a["n"] == 6


# -- view / API ----------------------------------------------------------------------------------------------------------

@pytest.fixture()
def stored(tmp_path, monkeypatch):
    monkeypatch.setenv("GRN_SIM_HOME", str(tmp_path))
    (tmp_path / "benchmarks").mkdir()
    rows = _rows({vb.RANDOM: lambda s: 0.05, vb.HUBNESS: lambda s: 0.1, "a": lambda s: 0.3 + 0.01 * s}, pathway="cascade") + \
        _rows({vb.RANDOM: lambda s: 0.05, vb.HUBNESS: lambda s: 0.1, "a": lambda s: 0.05}, pathway="regulon", mode="knockout")
    (tmp_path / "benchmarks" / "vcc_screen_benchmark.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return tmp_path


def test_answer_reports_levels_aggregate_and_pivot(stored):
    out = view.answer({"mode": ["knockdown"], "compare_by": ["pathway"]})
    assert out["available"] and out["levels"]["pathway"] == ["cascade", "regulon"]
    assert out["aggregate"]["n_simulations"] == 10                       # the knockout setting is filtered out
    assert list(out["pivot"]["levels"]) == ["cascade"]
    assert out["algorithms"]["s2g"]["uses_pairs"] is True and "decision_rule" in out


def test_answer_rejects_an_unknown_comparison_factor(stored):
    with pytest.raises(ValueError):
        view.answer({"compare_by": ["colour"]})


def test_missing_results_file_is_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("GRN_SIM_HOME", str(tmp_path / "empty"))
    monkeypatch.setattr(view, "resolve_data_file", lambda *p: tmp_path / "nowhere.jsonl")
    out = view.answer({})
    assert out["available"] is False and out["aggregate"]["algorithms"] == []


@pytest.mark.parametrize("query,fragment", [({"n_genes": ["5"]}, "n_genes"), ({"n_genes": ["201"]}, "n_genes"),
                                            ({"pathway": ["nope"]}, "pathway"), ({"mode": ["zap"]}, "mode"),
                                            ({"selection": ["x"]}, "selection"), ({"pairs": ["-1"]}, "pairs"),
                                            ({"n_genes": ["many"]}, "integer")])
def test_live_simulation_validates_its_parameters(query, fragment):
    with pytest.raises(ValueError, match=fragment):
        view.parse_simulation(query)


def test_live_simulation_is_saved_and_joins_the_statistics(stored):
    before = view.answer({})["aggregate"]["n_simulations"]
    out = view.simulate({"n_genes": ["20"], "pathway": ["regulon"], "seed": ["7"]})
    assert out["saved"] and out["seed"] == 7 and out["generators"] == ["linear_sem"]
    assert {r["generator"] for r in out["rows"]} == {"linear_sem"} and "s2g" in out["comparison"]
    assert (stored / view.USER_FILE).is_file()
    assert view.answer({})["aggregate"]["n_simulations"] == before + 1


def test_live_comparison_runs_every_generator_on_the_same_graph(stored):
    out = view.simulate({"n_genes": ["16"], "generator": ["all"], "seed": ["5"]}, save=False)
    assert out["generators"] == list(vb.GENERATORS) and not out["errors"] and not out["saved"]
    assert set(out["comparison"]["s2g"]) == set(vb.GENERATORS)
    edges = {r["generator"]: r["n_edges"] for r in out["rows"] if r["algorithm"] == "random"}
    assert len(set(edges.values())) == 1                                  # one graph for all five generators


def test_live_comparison_on_the_panel1_network(stored):
    from grn_sim.core.network import GeneNetwork
    from grn_sim.core.topology import TopologyType

    net = GeneNetwork.create(n=14, k_av=1.8, net_type=TopologyType.POWER_LAW, seed=3)
    out = view.simulate({"generator": ["all"], "source": ["current"], "seed": ["2"]}, current_network=net)
    assert out["setting"]["n_genes"] == 14 and out["setting"]["pathway"] == "panel1" and not out["saved"]
    with pytest.raises(ValueError, match="panel 1"):
        view.simulate({"source": ["current"]}, current_network=None)


def test_endpoints_over_http(app_url, stored):
    with urllib.request.urlopen(app_url + "/api/vcc/screen_benchmark?pathway=cascade&compare_by=mode") as r:
        data = json.load(r)
    assert data["status"] == "success" and data["aggregate"]["n_simulations"] == 10
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(app_url + "/api/vcc/screen_benchmark?compare_by=colour")
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(app_url + "/api/vcc/screen_simulate?n_genes=500")
    assert err.value.code == 400
