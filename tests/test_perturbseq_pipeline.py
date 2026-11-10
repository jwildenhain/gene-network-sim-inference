"""Phase 5: the one-call Perturb-seq pipeline, the `grn-sim perturbseq` CLI and the studio API around them."""

import json
import sys

import pytest

from grn_sim.perturbseq.pipeline import build_network, run_screen


@pytest.fixture(scope="module")
def small_net():
    return build_network("replogle_like", 30, seed=2)


def test_pipeline_runs_all_stages_and_reports(small_net):
    run = run_screen(small_net, cells_per_condition=20, n_control_cells=120, pairs=2, seed=1)
    assert run.population is not None
    assert run.adata.n_obs == 120 + 20 * (30 + 2)
    assert {"screen", "cells", "measurement"} <= set(run.seconds)
    assert "assignment_status" in run.adata.obs.columns
    result = run.realism()
    names = {r["statistic"] for r in result["report"]}
    assert {"on_target_knockdown", "fraction_with_phenotype", "affected_pair_density"} <= names
    assert result["score"] >= 0 and result["settings"]["pairs"] == 2


def test_pipeline_legacy_and_unmeasured_variants(small_net):
    legacy = run_screen(small_net, cells_per_condition=5, cell_model="legacy", seed=1)
    assert legacy.population is None
    assert "affected_pair_density" in legacy.realism()["statistics"]
    plain = run_screen(small_net, cells_per_condition=5, measurement=False, seed=1)
    assert "assignment_status" not in plain.adata.obs.columns


@pytest.mark.parametrize("kwargs", [{"cell_model": "gillespie"}, {"cells_per_condition": 0}])
def test_pipeline_rejects_bad_settings(small_net, kwargs):
    with pytest.raises(ValueError):
        run_screen(small_net, **kwargs)


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError):
        build_network("nope", 10)


def test_cli_writes_h5ad_and_report(tmp_path, monkeypatch, capsys):
    pytest.importorskip("anndata")
    from grn_sim import cli

    out, report = tmp_path / "screen.h5ad", tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["grn-sim", "perturbseq", "--genes", "25", "--cells", "15", "--controls", "100",
                                      "--out", str(out), "--report", str(report)])
    cli.main()
    assert out.exists() and report.exists()
    data = json.loads(report.read_text())
    assert data["settings"]["n_genes"] == 25 and len(data["report"]) == 7
    assert "Realism vs genome-scale" in capsys.readouterr().out


def test_server_realism_endpoint_and_cached_export():
    pytest.importorskip("anndata")
    from grn_sim.web import server

    server._execute_simulation({"num_genes": 20, "topology": "modular", "k_av": 2.5, "seed": 3, "auto_infer": False})
    payload = server._perturb_seq_realism_payload(None, 15, "knockdown", 0, "lna", "current", True, 100)
    assert payload["status"] == "success" and payload["n_genes"] == 20
    json.dumps(payload)  # NaNs were replaced, so the payload is valid JSON
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=15, n_control_cells=100)
    assert filename.endswith("_lna_15cells.h5ad") and len(data) > 0
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=5, preset="replogle_like")
    assert "replogle_like" in filename
    with pytest.raises(ValueError):
        server._perturb_seq_realism_payload(None, 15, "input", 0, "lna", "current", True, None)


@pytest.mark.parametrize("query", [{"preset": ["nope"]}, {"controls": ["-1"]}, {"controls": ["999999"]}])
def test_server_rejects_bad_extras(query):
    from grn_sim.web import server

    with pytest.raises(ValueError):
        server._parse_perturb_seq_extras(query)


def test_server_extras_defaults():
    from grn_sim.web import server

    extras = server._parse_perturb_seq_extras({})
    assert extras == {"preset": "current", "measurement": True, "n_control_cells": None, "perturbed": 0,
                      "selection": "effective"}
    assert server._parse_perturb_seq_extras({"measurement": ["0"]})["measurement"] is False
    assert set(server.PERTURB_SEQ_PRESETS) == {"current", "replogle_like", "thesis", "cascade", "feedforward", "lineage", "regulon"}
