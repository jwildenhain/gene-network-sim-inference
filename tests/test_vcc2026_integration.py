"""The Virtual Cell Challenge scoring in the Virtual Cell tab: the screen benchmark rows, the two API endpoints and the
cross-atlas payload."""

import json
import urllib.error
import urllib.request

import pytest

from grn_sim.evaluation import vcc2026 as v
from grn_sim.evaluation import vcc_benchmark as vb


def _get(url):
    with urllib.request.urlopen(url, timeout=600) as res:
        return json.loads(res.read())


def test_run_simulation_returns_vcc_scores_and_annotates_the_rows():
    algorithms = [a for a in vb.algorithm_library() if a.name in ("random", "mean_difference", "s2g", "gene_features")]
    rows, vcc = vb.run_simulation(vb.Setting(generator="nb_copula", n_genes=24, cells=30), 5, algorithms, return_vcc=True,
                                  bystanders=50)
    assert len(rows) == 4 and all("vcc_overall" in r and "vcc_pds" in r for r in rows)
    assert vcc["bystanders"] == 50 and vcc["n_genes_scored"] == 74
    by_key = {r["key"]: r for r in vcc["rows"]}
    for r in rows:
        if by_key[r["algorithm"]]["available"]:
            assert r["vcc_overall"] == by_key[r["algorithm"]]["overall"]
        else:
            assert r["vcc_overall"] is None
    json.dumps(vcc, allow_nan=False)


def test_screen_simulate_endpoint_scores_the_challenge_and_stores_only_standard_cell_counts(app_url):
    d = _get(f"{app_url}/api/vcc/screen_simulate?generator=linear_sem&n_genes=20&cells=30&seed=21&save=0&pathway=cascade")
    assert d["status"] == "success" and d["saved"] is False
    vcc = d["vcc_scores"]["linear_sem"]
    assert vcc["metrics"] == list(v.METRICS)
    anchors = vcc["anchors"]
    assert set(anchors["baseline"]) == set(v.METRICS) and anchors["n_genes"] == 20
    keys = {r["key"] for r in vcc["rows"]}
    assert {"baseline", "replicate", "truth_means", "self_prior", "random"} <= keys
    assert all(set(r["members"]) == set(v.METRICS) for r in vcc["rows"] if r["available"])
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{app_url}/api/vcc/screen_simulate?cells=5")
    assert err.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{app_url}/api/vcc/screen_simulate?bystanders=-1")
    assert err.value.code == 400


def test_cross_atlas_payload_carries_the_held_out_predictions_on_the_challenge_scale(app_url):
    d = _get(f"{app_url}/api/benchmark/vcc_atlasshift?num_genes=12&k_av=1.8&seed=2026&perturbation=knockdown&generator=linear_sem&bystanders=0")
    vcc = d["vcc_scores"]
    assert vcc["held_out"] == [f"Gene_{i:02d}" for i in (10, 11, 12)] and vcc["cells_per_perturbation"] == 400
    keys = [r["key"] for r in vcc["rows"]]
    assert "atlasshift" in keys and "truth_means" in keys and keys[-2:] == ["baseline", "replicate"]
    table_keys = {(r["method"], r["key"]) for r in d["response_table"]}
    assert all(any(k == r["key"] for _, k in table_keys) or r["kind"] in ("anchor", "oracle") for r in vcc["rows"])
    unavailable = [r for r in vcc["rows"] if not r["available"]]
    assert all(r["reason"] for r in unavailable)
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{app_url}/api/benchmark/vcc_atlasshift?num_genes=12&bystanders=99999")
    assert err.value.code == 400
