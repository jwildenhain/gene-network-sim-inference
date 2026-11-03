"""Regressions from the full UI/UX test: silent acceptance of bad names and silent loss of stored results."""
import json
import urllib.error
import urllib.request

import pytest

from grn_sim.web import server


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _state(url):
    with urllib.request.urlopen(url + "/api/state", timeout=60) as r:
        return json.loads(r.read())


def test_clean_algorithm_list():
    assert server._clean_algorithm_list(None) == server.DEFAULT_ALGORITHMS
    assert server._clean_algorithm_list("vote") == ["vote"]
    assert server._clean_algorithm_list(["vote", "s2g", "vote"]) == ["vote", "s2g"]
    assert server._clean_algorithm_list(["ridge"]) == ["ridge_cv"]          # legacy key
    assert server._clean_algorithm_list(["score_vote"]) == ["score_vote"]
    for bad in ([], ["nope"], ["vote", "nope"], 5, {"a": 1}):
        with pytest.raises(ValueError):
            server._clean_algorithm_list(bad)


def test_simulate_request_checks_names_except_for_benchmarks():
    assert server._validate_sim_request({"topology": "random"})["topology"] == "random"
    for key in ("topology", "kinetics_engine", "dynamics_preset"):
        with pytest.raises(ValueError, match=key):
            server._validate_sim_request({key: "zzz"})
    assert "topology" not in server._validate_sim_request({"topology": ""})          # blank means "not given"
    assert "topology" not in server._validate_sim_params({"topology": ""})
    # benchmark requests carry dataset-specific names
    assert server._validate_sim_request({"mode": "benchmark_dream4_10gene", "topology": "benchmark_dream4_10gene"})["topology"]


def test_execute_inference_refuses_an_empty_selection(app_url):
    assert _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 1, "topology": "random"})[0] == 200
    assert _post(app_url + "/api/infer", {"algorithms": ["vote", "s2g"]})[0] == 200
    with pytest.raises(ValueError):
        server._execute_inference([])
    with pytest.raises(ValueError):
        server._execute_inference(["nope"])


@pytest.mark.parametrize("body", [{"algorithms": []}, {"algorithms": ["nope"]}, {"algorithms": ["vote", "nope"]}, {"algorithms": 7}])
def test_infer_rejects_bad_algorithm_lists_and_keeps_results(app_url, body):
    assert _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 2, "topology": "random"})[0] == 200
    assert _post(app_url + "/api/infer", {"algorithms": ["vote", "s2g"]})[0] == 200
    before = _state(app_url)
    status, resp = _post(app_url + "/api/infer", body)
    assert status == 400 and "error" in resp
    after = _state(app_url)
    def inferred(s):
        return sorted((s.get("inferred") or s.get("inferred_results") or {}).get("metrics") or {})

    assert inferred(before) == inferred(after) == ["s2g", "vote"]               # results were not wiped


@pytest.mark.parametrize("field, value", [("topology", "zzz"), ("kinetics_engine", "x"), ("dynamics_preset", "wavy")])
def test_simulate_endpoint_rejects_unknown_names(app_url, field, value):
    status, resp = _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 3, field: value})
    assert status == 400 and field in resp["error"]


def test_simulate_endpoint_treats_blank_topology_as_not_given(app_url):
    status, resp = _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 4, "topology": ""})
    assert status == 200 and resp["topology"]                                    # the default was used and recorded


def test_auto_infer_validates_algorithms(app_url):
    status, resp = _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 5, "auto_infer": True, "algorithms": ["nope"]})
    assert status == 400 and "nope" in resp["error"]
    status, resp = _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 5, "auto_infer": True, "algorithms": ["score_vote"]})
    assert status == 200 and list(resp["inferred"]["metrics"]) == ["score_vote"]
