"""The cross-atlas comparison on statistical data: limits, contexts, double-perturbation scope, background runs."""
import json
import multiprocessing
import time
import urllib.request

import numpy as np
import pytest

from grn_sim.benchmarks import vcc_benchmark as vcc


def test_gene_limits_depend_on_the_generator():
    vcc.validate_vcc_params(200, 1.8, 0.15, "knockdown", "linear_sem")
    vcc.validate_vcc_params(32, 1.8, 0.15, "knockdown", "mechanistic")
    vcc.validate_vcc_params(vcc.MAX_GENES, 1.8, 0.15, "knockdown", "mechanistic")
    with pytest.raises(ValueError, match="mechanistic"):
        vcc.validate_vcc_params(vcc.MAX_GENES + 1, 1.8, 0.15, "knockdown", "mechanistic")
    with pytest.raises(ValueError, match="200"):
        vcc.validate_vcc_params(201, 1.8, 0.15, "knockdown", "linear_sem")
    vcc.validate_vcc_params(150, 1.8, 0.15, "knockdown", "boolean")
    with pytest.raises(ValueError, match="150"):
        vcc.validate_vcc_params(151, 1.8, 0.15, "knockdown", "boolean")
    with pytest.raises(ValueError, match="generator"):
        vcc.validate_vcc_params(20, 1.8, 0.15, "knockdown", "ode")


def test_statistical_contexts_share_the_graph_and_the_target_is_rewired():
    ctx = vcc._StatContexts("linear_sem", 30, 1.8, 7, "knockdown")
    sources, target = ctx.contexts(0.3)
    base = np.asarray(sources[0][1].true_matrix)
    assert all(np.array_equal(base, np.asarray(e.true_matrix)) for _, e in sources)
    assert not np.array_equal(base, np.asarray(target.true_matrix))
    assert (np.asarray(target.true_matrix) != 0).sum() == (base != 0).sum()          # edges move, none are lost
    assert not np.allclose(sources[0][1].X, sources[1][1].X)                          # contexts differ in strength
    assert np.allclose(np.diag(target.U), -vcc.KNOCKDOWN_EFFICIENCY)
    # without measurement noise X is exactly the model's relative response
    clean = vcc._StatContexts("linear_sem", 30, 1.8, 7, "knockdown", measurement_noise=0.0)
    exp = clean.experiment(clean._model(clean._graph(0.0, 1, 0)), 0)
    m = exp.network
    assert np.allclose(exp.X[:, 4], np.exp(m.log_mean((4,), np.array([[vcc.KNOCKDOWN_EFFICIENCY]]))[0]) - 1.0)


def test_rewired_contexts_drift_away_from_the_target():
    ctx = vcc._StatContexts("nonlinear_scm", 30, 1.8, 3, "knockdown")
    _, target = ctx.contexts(0.0)
    near, far = ctx.rewired(0.0), ctx.rewired(0.5)
    def drift(e):
        return float(np.mean(np.abs(e.X - target.X)))

    assert drift(near) < drift(far)


@pytest.fixture(scope="module")
def payload_40():
    return vcc.run_vcc_comparative_benchmark(num_genes=40, k_av=1.8, seed=11, perturbation_mode="knockdown",
                                             generator="linear_sem", rewire_fractions=[0.0, 0.3])


def test_a_40_gene_statistical_comparison_screens_every_pair_and_refits_hill(payload_40):
    p = payload_40
    assert p["status"] == "success" and p["parameters"]["generator"] == "linear_sem"
    cs = p["combinatorial_screen"]
    assert len(cs["screened_genes"]) == 40 and cs["all_pairs"] is True
    assert max(max(row) for row in cs["epistasis_matrix"]) < 1e-9                    # log-linear: no interactions
    hill = {m["key"]: m for m in p["parameters"]["hill_models"]}
    assert hill["oracle"]["available"] and hill["null"]["available"]
    assert len(p["summary_table"]) >= 18 and len(p["target_profiles"]) == 40
    json.dumps(p, allow_nan=False)


def test_the_screen_covers_the_most_connected_genes_above_the_generator_limit(monkeypatch):
    monkeypatch.setitem(vcc.SCREEN_MAX_GENES, "nonlinear_scm", 10)
    ctx = vcc._StatContexts("nonlinear_scm", 30, 1.8, 5, "knockdown")
    _, target = ctx.contexts(0.0)
    eps, _, genes = ctx.dual_screen(target)
    out = (np.asarray(target.network.adjacency) != 0).sum(axis=1)
    assert len(genes) == 10 and out[genes].min() >= np.sort(out)[-10]
    others = [g for g in range(30) if g not in genes]
    assert not eps[np.ix_(others, others)].any()


def test_nonlinear_data_do_show_interactions():
    ctx = vcc._StatContexts("nonlinear_scm", 20, 1.8, 5, "knockdown")
    _, target = ctx.contexts(0.0)
    eps, unsteady, genes = ctx.dual_screen(target)
    assert genes == list(range(20)) and unsteady == 0 and eps.max() > 1e-3


def test_the_child_process_entry_point_reports_stages_and_the_payload():
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    args = {"num_genes": 12, "k_av": 1.8, "seed": 3, "perturbation_mode": "knockdown", "generator": "boolean",
            "target_rewire_fraction": 0.15, "rewire_fractions": [0.0, 0.2]}
    proc = ctx.Process(target=vcc.run_in_child, args=(args, q))
    proc.start()
    kinds = []
    payload = None
    deadline = time.time() + 300
    while time.time() < deadline:
        kind, value = q.get(timeout=300)
        kinds.append(kind)
        if kind != "stage":
            payload = value
            break
    proc.join(timeout=30)
    assert kinds[-1] == "done", payload
    assert "stage" in kinds and payload["parameters"]["num_genes"] == 12


def test_large_runs_are_computed_in_the_background(app_url):
    url = (app_url + "/api/benchmark/vcc_atlasshift?num_genes=34&k_av=1.8&seed=5&rewire_fraction=0.15"
           "&perturbation=knockdown&generator=linear_sem")
    first = urllib.request.urlopen(url)
    assert first.status == 202 and json.load(first)["status"] == "running"
    deadline = time.time() + 600
    while time.time() < deadline:
        res = urllib.request.urlopen(url)
        if res.status == 200:
            data = json.load(res)
            assert data["parameters"]["num_genes"] == 34 and data["parameters"]["generator"] == "linear_sem"
            break
        body = json.load(res)
        assert body["state"] == "running" and body["stage"]
        time.sleep(3)
    else:
        pytest.fail("the background run did not finish")
