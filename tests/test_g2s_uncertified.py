"""G2S quick mode for ill-conditioned wide bases: no per-step SVD, same selections as the certified path."""
import numpy as np

from grn_sim.inference import _g2s_fast as fast
from grn_sim.inference.g2s import GeneralToSpecific


def _data(n=240, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n)) @ np.diag(np.logspace(0, 4, n))      # wide scale range, N = M
    U = rng.normal(size=(n, n))
    return X, U


def params(accept):
    return fast.G2SParams(criterion="mdl", min_interactions=1, max_interactions=10, drop_rule="loo", accept_uncertified=accept)


def test_uncertified_steps_replace_refreshes_and_selection_matches(monkeypatch):
    X, U = _data()
    basis = fast.make_shared_basis(X)
    exact = [fast.solve_gene(U[i].copy(), basis, params(False)) for i in range(4)]
    monkeypatch.setattr(fast, "_G_MAX", 1e-14)                        # no step can be certified
    plain = fast.solve_gene(U[0].copy(), basis, params(False))
    quick = [fast.solve_gene(U[i].copy(), basis, params(True)) for i in range(4)]
    assert plain[2]["refresh"] > quick[0][2]["refresh"]               # the flag removes the per-step recomputation
    assert quick[0][2]["uncertified"] > 0
    for e, q in zip(exact, quick):
        assert set(e[0]) == set(q[0])


def test_flag_is_off_by_default_and_needs_a_cap():
    assert GeneralToSpecific().accept_uncertified is False
    X, U = _data(60)
    basis = fast.make_shared_basis(X)
    p = fast.G2SParams(criterion="mdl", min_interactions=1, max_interactions=None, drop_rule="loo", accept_uncertified=True)
    assert "uncertified" not in fast.solve_gene(U[0].copy(), basis, p)[2]


def test_diagnostics_report_count():
    X, U = _data(120)
    res = GeneralToSpecific(criterion="mdl", max_interactions=8, drop_rule="loo", accept_uncertified=True, backend="numpy").fit(X, U)
    assert "uncertified_steps" in res.diagnostics
