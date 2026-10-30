"""Large studio runs: N^2-sized trajectories and gene profiles are served on demand instead of embedded."""

import json

import numpy as np
import pytest

from grn_sim.web import server

PARAMS = {"num_genes": 14, "topology": "modular", "k_av": 2.0, "seed": 9, "auto_infer": False}


def simulate(monkeypatch, threshold):
    monkeypatch.setattr(server, "LARGE_NETWORK_GENES", threshold)
    return server._execute_simulation(dict(PARAMS))


def test_small_runs_still_embed_everything(monkeypatch):
    payload = simulate(monkeypatch, 60)
    assert payload["trajectories_on_demand"] is False
    assert len(payload["perturbation_trajectories"]) == 14 and len(payload["gene_profiles"]) == 14


def test_large_runs_leave_out_the_quadratic_data_and_serve_it_on_request(monkeypatch):
    full = simulate(monkeypatch, 60)
    lazy = simulate(monkeypatch, 10)
    assert lazy["trajectories_on_demand"] is True
    assert lazy["perturbation_trajectories"] == [] and lazy["gene_profiles"] == {}
    assert len(json.dumps(lazy)) < len(json.dumps(full))
    assert lazy["X"] == full["X"], "same network, same seed: only the payload layout differs"

    traj = server._trajectory_payload(None, 3)
    assert traj["perturbed_gene"] == "G4"
    assert np.allclose(traj["trajectory"], full["perturbation_trajectories"][3], atol=1e-3)

    for gene in ("G2", "7"):
        name = gene if gene.startswith("G") else f"G{int(gene) + 1}"
        prof = server._gene_profile_payload(None, gene)
        ref = full["gene_profiles"][name]
        assert prof["name"] == name
        assert prof["in_regulators"] == ref["in_regulators"] and prof["out_targets"] == ref["out_targets"]
        assert prof["expression_across_conditions"] == ref["expression_across_conditions"]
        assert set(prof["regulator_trajectories"]) == set(ref["regulator_trajectories"])
        for key, curve in ref["regulator_trajectories"].items():
            assert np.allclose(prof["regulator_trajectories"][key], curve, atol=1e-3)


def test_large_runs_keep_only_downsampled_trajectories_in_memory(monkeypatch):
    simulate(monkeypatch, 10)
    trajs = server.CURRENT_STATE["experiment"].perturbation_trajectories
    assert all(t.shape[1] <= 50 and t.dtype == np.float32 for t in trajs)


@pytest.mark.parametrize("call", [
    lambda: server._trajectory_payload(None, 99),
    lambda: server._gene_profile_payload(None, "NOPE"),
])
def test_bad_on_demand_requests_are_rejected(monkeypatch, call):
    simulate(monkeypatch, 10)
    with pytest.raises(ValueError):
        call()
