"""Simulation parameter limits for genome-scale (2000 gene) jobs and the calibrated-network preset."""
import pytest

from grn_sim.web import server


def test_gene_cap_allows_two_thousand_genes():
    assert server._validate_sim_params({"num_genes": 2000})["num_genes"] == 2000
    with pytest.raises(ValueError, match="num_genes"):
        server._validate_sim_params({"num_genes": 2001})


def test_calibration_preset_is_validated():
    assert server._validate_sim_params({"calibration_preset": "replogle_like"}, strict=True)["calibration_preset"] == "replogle_like"
    assert "calibration_preset" not in server._validate_sim_params({"calibration_preset": ""})
    with pytest.raises(ValueError, match="calibration_preset"):
        server._validate_sim_params({"calibration_preset": "nope"})
