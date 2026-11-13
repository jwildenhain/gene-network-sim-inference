"""Ensembles reuse G2S / NIR fits of the same data and settings, and an ensemble still reports its stand-alone run time."""
import numpy as np
import pytest

from grn_sim.inference import _cache, registry
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.delenv("GRN_SIM_NO_RESULT_CACHE", raising=False)
    _cache.clear()
    yield
    _cache.clear()


def _data(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n)), np.eye(n) * 2 + 0.05 * rng.normal(size=(n, n))


def test_a_second_identical_fit_is_a_cache_hit_with_equal_results():
    X, U = _data()
    a = GeneralToSpecific(drop_rule="loo")
    first = a.fit(X, U)
    second = GeneralToSpecific(drop_rule="loo").fit(X, U)
    assert "cache_hit" not in first.diagnostics and second.diagnostics["cache_hit"] is True
    assert second.diagnostics["cached_seconds"] == pytest.approx(first.time_elapsed)
    assert np.array_equal(first.A_est, second.A_est) and np.array_equal(first.adjacency_pred, second.adjacency_pred)
    second.A_est[0, 0] += 1.0                                   # a hit is a copy: changing it does not poison the cache
    assert GeneralToSpecific(drop_rule="loo").fit(X, U).A_est[0, 0] == first.A_est[0, 0]


def test_different_data_settings_or_arguments_do_not_share_an_entry():
    X, U = _data()
    GeneralToSpecific(drop_rule="loo").fit(X, U)
    assert "cache_hit" not in GeneralToSpecific(drop_rule="loo", max_interactions=2).fit(X, U).diagnostics
    X2, U2 = _data(seed=1)
    assert "cache_hit" not in GeneralToSpecific(drop_rule="loo").fit(X2, U2).diagnostics
    assert "cache_hit" not in NIRAlgorithm(k_interactions=3).fit(X, U).diagnostics


def test_vote_after_g2s_reuses_it_and_reports_the_member_time(monkeypatch):
    X, U = _data()
    ctx = registry.BuildContext(k_est=3, use_reg=True, is_sos=False)
    g2s_res = registry.build("g2s", ctx).fit(X, U)
    vote_res = registry.build("vote", ctx).fit(X, U)
    assert vote_res.diagnostics["cached_seconds"] >= g2s_res.time_elapsed * 0.99
    assert registry.build("score_vote", ctx).fit(X, U).diagnostics["cached_seconds"] >= g2s_res.time_elapsed * 0.99


def test_the_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("GRN_SIM_NO_RESULT_CACHE", "1")
    X, U = _data()
    GeneralToSpecific(drop_rule="loo").fit(X, U)
    assert "cache_hit" not in GeneralToSpecific(drop_rule="loo").fit(X, U).diagnostics


def test_the_cache_is_bounded():
    for seed in range(_cache.MAX_ENTRIES + 2):
        GeneralToSpecific(drop_rule="loo").fit(*_data(seed=seed))
    assert len(_cache._CACHE) == _cache.MAX_ENTRIES
