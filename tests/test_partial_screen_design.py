"""Partial screens as a simulation option: the design object, effective-gene selection, their own statistics, CLI, API, registry."""
import json
import sys

import numpy as np
import pytest

from grn_sim.perturbseq.design import ScreenDesign, select_perturbed
from grn_sim.perturbseq.pipeline import build_network, run_screen
from grn_sim.perturbseq.realism import NOT_COMPARABLE_UNDER_PARTIAL, realism_score


@pytest.fixture(scope="module")
def net():
    return build_network("replogle_like", 80, seed=3)


def test_design_options_and_validation(net):
    assert ScreenDesign.from_options(None).kind == "full" and ScreenDesign.from_options(0).kind == "full"
    d = ScreenDesign.from_options(10, "random")
    assert d.is_partial and d.label() == "partial_10_random"
    assert ScreenDesign.from_options(5, genes=[4, 2, 2]).genes == (2, 4)
    for bad in (ScreenDesign.partial(0), ScreenDesign.partial(81), ScreenDesign.partial(5, "hubs"), ScreenDesign.explicit([99])):
        with pytest.raises(ValueError):
            bad.validate(80)


def test_effective_selection_picks_only_genes_that_move_other_genes(net):
    from grn_sim.perturbseq.conditions import simulate_conditions
    genes, info = select_perturbed(net, ScreenDesign.partial(15), seed=1)
    assert len(genes) == 15 and info["rule"] == "effective" and info["shortfall"] == 0
    probe = simulate_conditions(net, mode="knockdown", seed=1)
    labels = {c[0]: lab for c, lab in zip([c for c in probe.conditions if len(c) == 1], probe.labels)}
    assert all(labels[g] == "effective" for g in genes)
    again, _ = select_perturbed(net, ScreenDesign.partial(15), seed=1)
    other, _ = select_perturbed(net, ScreenDesign.partial(15), seed=2)
    assert again == genes and other != genes                       # seeded and repeatable


def test_effective_selection_reports_a_shortfall_instead_of_padding(net):
    genes, info = select_perturbed(net, ScreenDesign.partial(80), seed=1)
    assert len(genes) == info["n_selected"] == info["n_effective_available"] < 80
    assert info["shortfall"] == 80 - len(genes) > 0


def test_random_and_explicit_selection(net):
    genes, info = select_perturbed(net, ScreenDesign.partial(12, "random"), seed=1)
    assert len(genes) == 12 and info["rule"] == "random"
    genes, info = select_perturbed(net, ScreenDesign.explicit([7, 3, 9]), seed=1)
    assert genes == [3, 7, 9] and info["rule"] == "explicit"
    assert select_perturbed(net, ScreenDesign.full())[0] is None


@pytest.fixture(scope="module")
def partial_run(net):
    return run_screen(net, cells_per_condition=25, n_perturbed=10, seed=1)


def test_partial_run_perturbs_only_the_chosen_genes_and_stamps_the_design(partial_run):
    run = partial_run
    singles = [c[0] for c in run.screen.conditions if len(c) == 1]
    assert singles == run.settings["perturbed_genes"] and len(singles) == 10
    assert run.adata.n_vars == 80 and run.adata.n_obs == 25 * 10 + 25         # all genes measured; controls = cells per condition
    d = run.adata.uns["screen_design"]
    assert d["label"] == "partial_10_effective" and int(d["n_perturbed"]) == 10 and len(d["perturbed_genes"]) == 10
    assert run.settings["design"] == "partial_10_effective"


def test_partial_statistics_are_their_own_categories(partial_run):
    result = partial_run.realism()
    cats = result["design"]["categories"]
    assert set(cats) == {"design", "identifiability", "coverage", "blocks"}
    i = cats["identifiability"]
    assert i["edges_P_to_P"] + i["edges_P_to_Q"] + i["edges_from_unperturbed"] == i["edges_total"]
    assert abs(i["fraction_edges_identifiable"] - (i["edges_P_to_P"] + i["edges_P_to_Q"]) / i["edges_total"]) < 1e-9
    assert cats["design"]["n_perturbed"] == 10 and cats["design"]["fraction_perturbed_effective"] == 1.0
    assert 0.0 <= cats["coverage"]["fraction_unperturbed_responding"] <= 1.0
    json.dumps(result["design"], default=float)


def test_selection_biased_statistics_are_reported_but_not_scored(partial_run):
    rows = partial_run.realism()["report"]
    flagged = {r["statistic"] for r in rows if not r["comparable"]}
    assert flagged == NOT_COMPARABLE_UNDER_PARTIAL & {r["statistic"] for r in rows}
    assert realism_score(rows) == sum(min(r["distance"], 10.0) for r in rows if r["comparable"])


def test_random_partial_selection_keeps_every_statistic_comparable(net):
    rows = run_screen(net, cells_per_condition=15, n_perturbed=8, perturbed_selection="random", seed=1).realism()["report"]
    assert all(r["comparable"] for r in rows)


def test_full_screens_are_unchanged(net):
    run = run_screen(net, cells_per_condition=10, n_control_cells=40, seed=1)
    assert not run.design.is_partial and "design" not in run.realism()
    assert all(r["comparable"] for r in run.realism()["report"])
    assert run.settings["design"] == "full" and run.settings["perturbed_genes"] is None


def test_cli_partial_screen(tmp_path, monkeypatch, capsys):
    pytest.importorskip("anndata")
    from grn_sim import cli
    monkeypatch.setattr(sys, "argv", ["grn-sim", "perturbseq", "--genes", "40", "--cells", "12", "--controls", "60",
                                      "--perturbed", "6", "--selection", "effective", "--report", str(tmp_path / "r.json")])
    cli.main()
    out = capsys.readouterr().out
    assert "Partial screen: 6 of 40 genes perturbed (effective selection)" in out and "[identifiability]" in out
    assert json.loads((tmp_path / "r.json").read_text())["design"]["describe"]["n_perturbed"] == 6


def test_server_partial_screen_payload_and_validation():
    pytest.importorskip("anndata")
    from grn_sim.web import server
    server._execute_simulation({"num_genes": 30, "topology": "modular", "k_av": 2.5, "seed": 3, "auto_infer": False})
    payload = server._perturb_seq_realism_payload(None, 12, "knockdown", 0, "lna", "replogle_like", True, 60, 8, "effective")
    assert payload["design"]["describe"]["n_perturbed"] == 8 and "identifiability" in payload["design"]["categories"]
    assert not [r for r in payload["report"] if r["statistic"] == "affected_pair_density"][0]["comparable"]
    json.dumps(payload)
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=8, preset="replogle_like", perturbed=5,
                                                          selection="random")
    assert "partial_5_random" in filename and len(data) > 0
    for q in ({"perturbed": ["-1"]}, {"perturbed": ["999999"]}, {"selection": ["hubs"]}):
        with pytest.raises(ValueError):
            server._parse_perturb_seq_extras(q)
    assert server._parse_perturb_seq_extras({"perturbed": ["12"], "selection": ["random"]})["perturbed"] == 12


def test_registry_declares_partial_support_and_builds_adapters():
    from grn_sim.inference import registry
    from grn_sim.inference.partial import PartialAdapter
    flags = {s.key: s.partial for s in registry.specs()}
    assert set(flags.values()) <= {"native", "adapter"}
    assert flags["s2g"] == flags["g2s"] == flags["ridge_cv"] == "adapter" and flags["inspre"] == flags["nir"] == "native"
    assert isinstance(registry.build_partial("s2g"), PartialAdapter)
    assert not isinstance(registry.build_partial("inspre"), PartialAdapter)
    assert {a["key"]: a["partial"] for a in registry.describe()}["s2g"] == "adapter"
