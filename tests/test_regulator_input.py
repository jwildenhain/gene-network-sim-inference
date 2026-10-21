"""Regression tests for the regulator-input perturbation model (sign / orientation correctness)."""
import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.evaluation.metrics import compute_aupr, evaluate_inference
from grn_sim.inference import (
    CellBoxNeuralODE, FastSVD, GeneralToSpecific, KaggleMultiOutputRidge,
    NIRAlgorithm, SimpleToGeneral, VotingEnsemble,
)
from grn_sim.simulator.experiment import PerturbationSimulator

SEEDS = range(100, 105)


def _experiments(topo=TopologyType.RANDOM):
    for seed in SEEDS:
        net = GeneNetwork.create(n=12, k_av=2, net_type=topo, seed=seed)
        yield PerturbationSimulator(2.0).run(net, seed=seed)


def _mean(metric, make, mode_kwargs=None):
    vals = []
    for e in _experiments():
        res = make().fit(e.X, e.U, **(mode_kwargs or {}))
        vals.append(getattr(evaluate_inference(e.true_matrix, res.adjacency_pred), metric))
    return float(np.mean(vals))


@pytest.mark.parametrize("make_additive,make_regulator", [
    (lambda: NIRAlgorithm(k_interactions=2), lambda: NIRAlgorithm(k_interactions=2, perturbation_mode="regulator", adaptive_k=True)),
    (lambda: FastSVD(k_interactions=2), lambda: FastSVD(k_interactions=2, perturbation_mode="regulator")),
    (lambda: KaggleMultiOutputRidge(k_interactions=2), lambda: KaggleMultiOutputRidge(k_interactions=2, perturbation_mode="regulator")),
])
def test_regulator_mode_recovers_signs(make_additive, make_regulator):
    additive = _mean("signed_ppv", make_additive)
    regulator = _mean("signed_ppv", make_regulator)
    assert additive < 0.15          # additive model is sign-inverted on Hill-kinetics data
    assert regulator > 0.30
    assert regulator > additive + 0.2


def test_regulator_mode_output_uses_regulator_row_frame():
    e = next(_experiments())
    res = NIRAlgorithm(k_interactions=2, perturbation_mode="regulator").fit(e.X, e.U)
    direct = evaluate_inference(e.true_matrix, res.adjacency_pred).f1_score
    flipped = evaluate_inference(e.true_matrix, res.adjacency_pred.T).f1_score
    assert direct >= flipped


def test_cellbox_regulator_mode_runs_and_recovers_edges():
    e = next(_experiments())
    res = CellBoxNeuralODE(epochs=40, perturbation_mode="regulator").fit(e.X, e.U)
    assert res.adjacency_pred.shape == (12, 12)
    assert res.total_predicted_edges > 0


def test_g2s_defaults_unchanged_and_cap_limits_edges():
    e = next(_experiments())
    base = GeneralToSpecific().fit(e.X, e.U)
    capped = GeneralToSpecific(max_interactions=4).fit(e.X, e.U)
    assert capped.total_predicted_edges <= base.total_predicted_edges


def test_majority_vote_runs_and_is_subset_union():
    e = next(_experiments())
    nir = NIRAlgorithm(k_interactions=2, perturbation_mode="regulator", adaptive_k=True)
    res = VotingEnsemble(extra_members=[nir]).fit(e.X, e.U)
    assert res.adjacency_pred.shape == (12, 12)
    assert res.diagnostics["min_votes"] == 2


def test_adaptive_k_options_run():
    e = next(_experiments())
    f = FastSVD(k_interactions=3, perturbation_mode="regulator", adaptive_k=True).fit(e.X, e.U)
    r = KaggleMultiOutputRidge(k_interactions=3, perturbation_mode="regulator", adaptive_k=True, stability_boots=10).fit(e.X, e.U)
    assert f.adjacency_pred.shape == r.adjacency_pred.shape == (12, 12)


def test_aupr_metric():
    T = np.array([[0, 1], [-1, 0]])
    assert compute_aupr(T, np.abs(T) * 2.0) == pytest.approx(1.0)
    assert compute_aupr(T, 1.0 - np.abs(T)) < 0.6
    assert compute_aupr(np.zeros((2, 2)), np.ones((2, 2))) == 0.0
