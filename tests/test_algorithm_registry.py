"""The algorithm registry: an algorithm registered once reaches the API, the job queue, scoring, the sweep and the page."""
import json
import urllib.request

import numpy as np
import pytest

from grn_sim.inference import registry
from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.orientation import ADDITIVE_GARDNER, HILL, output_is_target_regulator
from grn_sim.web import server


class _EveryOtherEdge(BaseInferenceAlgorithm):
    """A deliberately silly algorithm: predicts the strongest |X| correlation partner of every gene."""

    def __init__(self, k: int = 1):
        super().__init__(name="dummy_algo")
        self.k = k

    def fit(self, X, U, **kwargs):
        n = X.shape[0]
        c = np.abs(np.corrcoef(X)) - 2 * np.eye(n)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, np.argsort(c[i])[-self.k:]] = 1
        return InferenceResult("dummy_algo", adj.astype(float), adj, 0.0)


@pytest.fixture
def dummy():
    spec = registry.AlgorithmSpec(
        key="dummy_algo", label="Dummy algorithm", long_name="Dummy (test only)", paradigm="Correlation",
        badge="Test", build=lambda ctx: _EveryOtherEdge(k=ctx.k_est), target_row=True, order=5000)
    registry.register(spec)
    yield spec
    registry.unregister("dummy_algo")


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read())


def test_builtins_are_registered_in_order():
    keys = registry.keys()
    assert len(keys) >= 19 and keys[:3] == ["s2g", "g2s", "vote"] and "score_vote" in keys
    assert registry.core_keys() == ["s2g", "g2s", "vote", "nir", "fast", "by_chance"]
    for spec in registry.specs():                       # every built-in can be built for both data models
        for use_reg in (True, False):
            assert spec.build(registry.BuildContext(k_est=3, use_reg=use_reg)) is not None


def test_registration_rules():
    spec = registry.AlgorithmSpec(key="x_one", label="X", build=lambda c: _EveryOtherEdge())
    registry.register(spec)
    try:
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)
        registry.register(spec, replace=True)
    finally:
        registry.unregister("x_one")
    for bad in ("", "Has Space", "UPPER", "dash-ed"):
        with pytest.raises(ValueError):
            registry.register(registry.AlgorithmSpec(key=bad, label="bad", build=lambda c: _EveryOtherEdge()))
    with pytest.raises(KeyError):
        registry.get("never_registered")


def test_orientation_follows_the_registration(dummy):
    assert output_is_target_regulator("dummy_algo", HILL) is True              # target_row=True
    assert output_is_target_regulator("dummy_algo", ADDITIVE_GARDNER) is True
    assert output_is_target_regulator("sfinn", HILL) and not output_is_target_regulator("nir", HILL)


def test_a_registered_algorithm_is_accepted_everywhere(dummy, app_url):
    assert "dummy_algo" in server.known_algorithms()
    assert server._clean_algorithm_list(["dummy_algo", "vote"]) == ["dummy_algo", "vote"]
    assert server.JOB_MANAGER._clean_algorithms(["dummy_algo"]) == ["dummy_algo"]
    assert registry.sweep_metadata()["dummy_algo"] == {"name": "Dummy (test only)", "type": "Correlation", "badge": "Test"}
    listed = {a["key"]: a for a in _get(app_url + "/api/algorithms")["algorithms"]}
    assert listed["dummy_algo"]["label"] == "Dummy algorithm" and "dummy_algo" in _get(app_url + "/api/jobs")["algorithms"]
    # and it actually runs and is scored through the studio
    assert _post(app_url + "/api/simulate", {"num_genes": 8, "seed": 6, "topology": "random"})[0] == 200
    status, resp = _post(app_url + "/api/infer", {"algorithms": ["dummy_algo", "s2g"]})
    assert status == 200 and set(resp["metrics"]) == {"dummy_algo", "s2g"}
    assert 0.0 <= resp["metrics"]["dummy_algo"]["f1_score"] <= 1.0


def test_removed_algorithms_are_refused_again(app_url):
    registry.register(registry.AlgorithmSpec(key="short_lived", label="Short", build=lambda c: _EveryOtherEdge()))
    assert "short_lived" in server.known_algorithms()
    registry.unregister("short_lived")
    with pytest.raises(ValueError, match="unknown algorithm"):
        server._clean_algorithm_list(["short_lived"])


def test_virtual_cell_challenge_picks_up_a_registered_algorithm(dummy):
    from grn_sim.benchmarks import vcc_benchmark as vcc

    assert "dummy_algo" in [k for k, _ in vcc.algorithms()] and "dummy_algo" in [k for k, _ in vcc.ALGORITHMS]
    assert dict(vcc.algorithms())["dummy_algo"] == "Correlation"                    # the spec's paradigm
    assert "dummy_algo" in vcc._build_runners(2) and vcc.fit_input_sign("dummy_algo") == 1.0
    assert any(m.key == "dummy_algo" and m.kind == "algorithm" for m in vcc.hill_models())
    assert "dummy_algo" not in [m.key for m in vcc.response_models()]                # no Jacobian declared: structure only
    registry.register(registry.AlgorithmSpec(key="dummy_algo", label="Dummy algorithm", build=lambda c: _EveryOtherEdge(),
                                             response_sign=-1.0), replace=True)
    assert {m.key: m.sign for m in vcc.response_models()}["dummy_algo"] == -1.0      # declared: joins the response comparison
    assert vcc.RESPONSE_SIGN["dummy_algo"] == -1.0


def test_virtual_cell_challenge_lists_every_builtin_once():
    from grn_sim.benchmarks import vcc_benchmark as vcc

    keys = [k for k, _ in vcc.algorithms()]
    assert len(keys) == len(set(keys)) and set(registry.keys()) <= set(keys) and "score_vote" in keys
    assert {m.key for m in vcc.response_models()} <= set(vcc._build_runners(2))
