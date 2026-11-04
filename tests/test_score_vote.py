"""ScoreVote: evidence-weighted vote of S2G, G2S and (optionally) NIR."""
import numpy as np
import pytest

from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.orientation import HILL, ADDITIVE_GARDNER, output_is_target_regulator
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.score_vote import ScoreVote


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 40
    A = np.zeros((n, n))
    for j in range(n):
        A[rng.choice(n, 3, replace=False), j] = rng.normal(size=3)
    X = rng.normal(size=(n, n)) + 0.3 * (A.T @ rng.normal(size=(n, n)))
    U = -(A.T @ X) + 0.05 * rng.normal(size=(n, n))
    return X, U


def _make(nir: bool, n_jobs: int):
    return ScoreVote(
        s2g=SimpleToGeneral(n_jobs=1),
        g2s=GeneralToSpecific(max_interactions=6, drop_rule="loo"),
        nir=NIRAlgorithm(k_interactions=3, perturbation_mode="regulator", adaptive_k=True, n_jobs=1) if nir else None,
        k_interactions=3,
        n_jobs=n_jobs,
    )


@pytest.mark.parametrize("nir", [False, True])
def test_top_k_per_target_and_members(data, nir):
    X, U = data
    res = _make(nir, 1).fit(X, U)
    assert res.adjacency_pred.shape == X.shape
    assert (np.abs(res.adjacency_pred).sum(axis=0) <= 3).all()          # at most K regulators per target (column)
    assert res.diagnostics["members"] == (["s2g", "g2s", "nir"] if nir else ["s2g", "g2s"])
    assert np.isfinite(res.A_est).all()


def test_parallel_matches_serial(data):
    X, U = data
    a = _make(True, 1).fit(X, U)
    b = _make(True, 4).fit(X, U)
    assert np.array_equal(a.adjacency_pred, b.adjacency_pred)


def test_opposite_signs_cancel(data):
    X, U = data
    sv = _make(False, 1)
    r1, r2 = sv.s2g.fit(X, U), sv.g2s.fit(X, U)
    z, _ = sv.score(X, U, [r1, r2])
    assert z.shape == X.shape
    assert not np.any((z != 0) & ~((r1.A_est != 0) | (r2.A_est != 0)))   # nothing scored outside the candidate set


def test_threshold_mode_without_k(data):
    X, U = data
    sv = _make(False, 1)
    sv.k_interactions = None
    res = sv.fit(X, U)
    z = sv.last_scores
    assert np.array_equal(res.adjacency_pred != 0, np.abs(z) >= sv.z_threshold)


def test_registered_like_vote():
    for dm in (HILL, ADDITIVE_GARDNER):
        assert output_is_target_regulator("score_vote", dm) == output_is_target_regulator("vote", dm)
    from grn_sim.web.server import KNOWN_ALGORITHMS
    assert "score_vote" in KNOWN_ALGORITHMS
