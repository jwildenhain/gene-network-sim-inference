"""
Tests for Benchmark Dataset Provenance, Raw TSV/ZIP Downloads, and Biophysical Parameter Estimation.
"""

import io
import json
import zipfile
import pytest
import numpy as np

from grn_sim.benchmarks.provenance import (
    get_benchmark_provenance,
    export_benchmark_raw_files,
    estimate_biophysical_parameters,
    BENCHMARK_PROVENANCE,
)
from grn_sim.benchmarks.benchmark_datasets import list_benchmark_datasets, load_benchmark_dataset
from grn_sim.web.server import _build_raw_benchmark_download, _execute_simulation


DATASETS = ["ecoli_sos", "dream4_10gene", "norman_2019_real_16tf", "replogle_2022_p53"]


def test_provenance_catalog_coverage():
    """Verifies all 4 primary datasets have complete scientific provenance."""
    for ds_id in DATASETS:
        prov = get_benchmark_provenance(ds_id)
        assert prov["id"] == ds_id
        assert "name" in prov
        assert "organism" in prov
        assert "technology" in prov
        assert "citations" in prov and len(prov["citations"]) > 0
        assert "genes_annotated" in prov and len(prov["genes_annotated"]) > 0
        assert "how_raw_data_is_used" in prov
        assert "wrangled_direct" in prov["how_raw_data_is_used"]
        assert "biophysical_fit" in prov["how_raw_data_is_used"]


def test_export_benchmark_raw_files(tmp_path):
    """Verifies generation of raw TSV matrices and JSON bundles."""
    for ds_id in DATASETS:
        files = export_benchmark_raw_files(ds_id, out_dir=tmp_path)
        assert files["expression_tsv"].exists()
        assert files["perturbation_tsv"].exists()
        assert files["ground_truth_tsv"].exists()
        assert files["json_bundle"].exists()

        # Check expression TSV content
        content = files["expression_tsv"].read_text()
        lines = content.strip().split("\n")
        assert len(lines) > 1
        headers = lines[0].split("\t")
        assert headers[0] == "Gene"

        # Check JSON bundle content
        with open(files["json_bundle"], "r") as f:
            data = json.load(f)
        assert data["dataset_id"] == ds_id
        assert len(data["gene_names"]) == len(lines) - 1
        assert len(data["X_expression_matrix"]) == len(data["gene_names"])
        assert len(data["U_perturbation_matrix"]) == len(data["gene_names"])
        assert len(data["true_adjacency_matrix"]) == len(data["gene_names"])


def test_estimate_biophysical_parameters():
    """Verifies non-linear Hill ODE parameter estimation from empirical steady states."""
    for ds_id in DATASETS:
        fit = estimate_biophysical_parameters(ds_id)
        assert "params" in fit
        params = fit["params"]
        num_genes = len(fit["gene_names"])

        assert len(fit["basal_v"]) == num_genes
        assert len(fit["degradation_B"]) == num_genes
        assert np.all(np.array(fit["basal_v"]) > 0)
        assert np.all(np.array(fit["degradation_B"]) > 0)

        # Check Ka / Ki Hill saturation matrices
        Ka = np.array(fit["Ka"])
        Ki = np.array(fit["Ki"])
        assert Ka.shape == (num_genes, num_genes)
        assert Ki.shape == (num_genes, num_genes)

        # Check regularized Jacobian estimation
        J = np.array(fit["jacobian_est"])
        assert J.shape == (num_genes, num_genes)
        assert not np.any(np.isnan(J))


def test_build_raw_benchmark_download_all_types():
    """Verifies in-memory download generation across TSV, JSON, and ZIP."""
    for ds_id in DATASETS:
        # TSV expression
        data_expr, fn_expr, mime_expr = _build_raw_benchmark_download(ds_id, "expression_tsv")
        assert len(data_expr) > 100
        assert fn_expr.endswith(".tsv")
        assert mime_expr == "text/tab-separated-values"

        # TSV perturbation
        data_pert, fn_pert, mime_pert = _build_raw_benchmark_download(ds_id, "perturbation_tsv")
        assert len(data_pert) > 100
        assert fn_pert.endswith(".tsv")

        # TSV ground truth
        data_true, fn_true, mime_true = _build_raw_benchmark_download(ds_id, "ground_truth_tsv")
        assert len(data_true) > 50
        assert fn_true.endswith(".tsv")

        # JSON bundle
        data_json, fn_json, mime_json = _build_raw_benchmark_download(ds_id, "json_bundle")
        assert len(data_json) > 500
        assert fn_json.endswith(".json")
        assert mime_json == "application/json"

        # ZIP archive
        data_zip, fn_zip, mime_zip = _build_raw_benchmark_download(ds_id, "zip_bundle")
        assert len(data_zip) > 1000
        assert fn_zip.endswith(".zip")
        assert mime_zip == "application/zip"

        # Inspect ZIP contents
        with zipfile.ZipFile(io.BytesIO(data_zip)) as zf:
            names = zf.namelist()
            assert any("expression.tsv" in n for n in names)
            assert any("perturbation.tsv" in n for n in names)
            assert any("ground_truth.tsv" in n for n in names)
            assert any("bundle.json" in n for n in names)
            assert "README_METADATA.txt" in names
            readme = zf.read("README_METADATA.txt").decode("utf-8")
            assert "GRN-Sim & phpIoRN 2.0" in readme


def test_simulation_loading_modes_distinction():
    """Verifies distinction between Direct Empirical Wrangled Mode and Biophysical Fit Mode."""
    for ds_id in ["ecoli_sos", "norman_2019_real_16tf"]:
        # Mode 1: Wrangled Direct
        res_wrangled = _execute_simulation({
            "dataset_id": ds_id,
            "loading_mode": "wrangled_direct",
        })
        assert res_wrangled["status"] == "success"
        assert res_wrangled["loading_mode"] == "wrangled_direct"
        assert res_wrangled["is_empirical_direct"] is True
        assert res_wrangled["is_biophysical_fit"] is False
        assert "provenance" in res_wrangled and res_wrangled["provenance"] is not None

        # Mode 2: Biophysical Fit
        res_bio = _execute_simulation({
            "dataset_id": ds_id,
            "loading_mode": "biophysical_fit",
        })
        assert res_bio["status"] == "success"
        assert res_bio["loading_mode"] == "biophysical_fit"
        assert res_bio["is_empirical_direct"] is False
        assert res_bio["is_biophysical_fit"] is True
        assert "fitted_parameters" in res_bio and res_bio["fitted_parameters"] is not None
        assert "basal_v" in res_bio["fitted_parameters"]
        assert "degradation_B" in res_bio["fitted_parameters"]
