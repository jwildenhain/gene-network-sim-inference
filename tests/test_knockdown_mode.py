"""CRISPRi-style knockdown generation (PerturbationSimulator perturbation_mode="knockdown") and the VCC benchmark on top of it."""

import numpy as np
import pytest

from grn_sim.benchmarks import vcc_benchmark as vcc
from grn_sim.benchmarks.hill_response import (
    HillResponseModel,
    mean_profile_baseline,
    pearson,
)
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.kinetics import evaluate_derivatives


def network(n=12, seed=3):
    return GeneNetwork.create(n=n, k_av=1.8, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)


def run(mode, n=12, seed=3, noise=0.0, **kwargs):
    return PerturbationSimulator(perturbation_factor=1.0, measurement_noise=noise, perturbation_mode=mode, **kwargs).run(network(n, seed), seed=seed)


# --- the generator ------------------------------------------------------------------------------------

def test_default_mode_is_the_thesis_input_convention():
    exp = run("input")
    assert np.allclose(np.diag(exp.U), 1.0) and np.count_nonzero(exp.U) == 12
    assert np.abs(np.diag(exp.X)).max() > 0 and np.mean(np.diag(exp.X)) > -0.2, "the gene's own transcript is not repressed"
    assert PerturbationSimulator().perturbation_mode == "input"


def test_knockdown_represses_the_perturbed_gene_itself():
    exp = run("knockdown", knockdown_efficiency=0.8)
    assert np.allclose(np.diag(exp.U), -0.8) and np.count_nonzero(exp.U) == 12, "the design records the efficiency"
    own = np.diag(exp.X)
    assert own.max() < -0.6 and own.min() > -0.95, f"own transcript should fall by roughly 80%, got {own}"
    for j in range(12):
        assert np.argmin(exp.X[:, j]) == j or exp.X[j, j] < -0.6


def test_a_stronger_knockdown_gives_a_larger_own_drop_and_a_larger_downstream_effect():
    weak, strong = run("knockdown", knockdown_efficiency=0.3), run("knockdown", knockdown_efficiency=0.9)
    assert np.mean(np.diag(strong.X)) < np.mean(np.diag(weak.X)) - 0.3
    off = ~np.eye(12, dtype=bool)
    assert np.abs(strong.X[off]).mean() > np.abs(weak.X[off]).mean()


def test_efficiency_spread_varies_the_design_but_stays_in_bounds():
    exp = run("knockdown", knockdown_efficiency=0.8, efficiency_spread=0.1)
    eff = -np.diag(exp.U)
    assert eff.std() > 0.02 and eff.min() >= 0.05 and eff.max() <= 0.99


@pytest.mark.parametrize("kwargs", [{"perturbation_mode": "overexpress"}, {"knockdown_efficiency": 0.0}, {"knockdown_efficiency": 1.0}])
def test_invalid_generator_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        PerturbationSimulator(**kwargs)


def test_unit_production_scale_changes_nothing():
    net = network(8)
    x = np.linspace(0.5, 2.0, 8)
    assert np.allclose(evaluate_derivatives(x, net), evaluate_derivatives(x, net, production_scale=np.ones(8)))
    scale = np.ones(8); scale[2] = 0.2
    d0, d1 = evaluate_derivatives(x, net), evaluate_derivatives(x, net, production_scale=scale)
    assert not np.isclose(d0[2], d1[2]) and np.allclose(np.delete(d0, 2), np.delete(d1, 2))


# --- the model of the generator ---------------------------------------------------------------------

def test_hill_model_in_knockdown_mode_reproduces_training_and_predicts_held_out_with_the_true_structure():
    scores_train, scores_held = [], []
    for seed in (1, 2, 3):
        net = network(12, seed)
        exp = PerturbationSimulator(1.0, 0.0, perturbation_mode="knockdown").run(net, seed=seed)
        held = [9, 10, 11]; seen = list(range(9))
        x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
        model = HillResponseModel(exp.true_matrix.astype(int), x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state, mode="knockdown")
        R = model.response_matrix(exp.U)
        scores_train.append(np.mean([pearson(R[:, j], exp.X[:, j]) for j in seen]))
        scores_held.append(np.mean([pearson(R[:, j], exp.X[:, j]) for j in held]))
    assert np.mean(scores_train) > 0.9 and np.mean(scores_held) > 0.7, (scores_train, scores_held)


def test_off_target_r_leaves_the_perturbed_gene_out():
    truth = np.array([-0.8, 0.3, -0.2, 0.1, 0.0])
    pred = np.array([-0.8, -0.3, 0.2, -0.1, 0.0])          # right target gene, wrong downstream response
    scores = vcc.profile_scores(truth, pred, 0)
    assert scores["pearson_r"] > 0.5 > 0 > scores["offtarget_r"]
    assert vcc.profile_scores(truth, truth, 0)["offtarget_r"] == 1.0


def test_a_target_only_baseline_looks_good_on_all_genes_and_has_no_off_target_skill():
    """The trap this mode exposes: r over all genes mostly measures knowing which gene was hit."""
    exp = run("knockdown", n=16, noise=0.01)
    self_only = np.diag(np.diag(exp.U))
    all_genes = np.mean([pearson(self_only[:, j], exp.X[:, j]) for j in range(16)])
    off_target = np.mean([vcc.profile_scores(exp.X[:, j], self_only[:, j], j)["offtarget_r"] for j in range(16)])
    assert all_genes > 0.5 and off_target == 0.0
    null = mean_profile_baseline(exp.X, range(16))
    assert np.mean([pearson(null[:, j], exp.X[:, j]) for j in range(16)]) < all_genes


# --- the VCC benchmark on knockdown data --------------------------------------------------------------

@pytest.fixture(scope="module")
def knockdown_payload():
    return vcc.run_vcc_comparative_benchmark(num_genes=8, k_av=1.8, perturbation_mode="knockdown")


@pytest.fixture(scope="module")
def input_payload():
    return vcc.run_vcc_comparative_benchmark(num_genes=8, k_av=1.8, perturbation_mode="input")


def rows(payload):
    return {(r["method"], r["key"]): r for r in payload["response_table"]}


def test_payload_reports_the_perturbation_mode(knockdown_payload, input_payload):
    assert knockdown_payload["parameters"]["perturbation_mode"] == "knockdown"
    assert knockdown_payload["parameters"]["knockdown_efficiency"] == vcc.KNOCKDOWN_EFFICIENCY
    assert input_payload["parameters"]["perturbation_mode"] == "input" and input_payload["parameters"]["knockdown_efficiency"] is None


def test_response_table_covers_every_algorithm_with_both_methods(knockdown_payload):
    table = rows(knockdown_payload)
    for model in vcc.RESPONSE_MODELS:
        assert ("linear", model.key) in table
    for key, _ in vcc.ALGORITHMS:
        if key != "atlasshift":
            assert ("hill", key) in table, key
    for key in ("hill_fs", "null", "oracle", "self_prior"):
        assert ("hill", key) in table
    assert ("reference", "atlasshift") in table
    for row in knockdown_payload["response_table"]:
        if row["available"]:
            assert set(row["held_out"]) >= {"pearson_r", "cosine_sim", "rmse", "offtarget_r"} and row["held_out_offtarget_minus_null"] is not None
        else:
            assert row["reason"], f"{row['key']} is unavailable without a reason"


def test_linear_rows_stay_zero_on_held_out_genes_and_the_null_row_is_the_zero_point(knockdown_payload):
    table = rows(knockdown_payload)
    assert table[("linear", "g2s")]["held_out"]["pearson_r"] == 0.0
    null = table[("hill", "null")]
    assert null["held_out_minus_null"] == 0.0 and null["held_out_offtarget_minus_null"] == 0.0


def test_the_target_prior_exists_only_in_knockdown_mode(knockdown_payload, input_payload):
    knock, plain = rows(knockdown_payload)[("hill", "self_prior")], rows(input_payload)[("hill", "self_prior")]
    assert knock["available"] and knock["held_out"]["pearson_r"] > 0.4 and knock["held_out"]["offtarget_r"] == 0.0
    assert not plain["available"] and plain["reason"]


def test_oracle_is_the_upper_bound_among_the_hill_rows(knockdown_payload):
    table = rows(knockdown_payload)
    oracle = table[("hill", "oracle")]["held_out"]["pearson_r"]
    assert oracle > 0.5
    assert all(r["held_out"]["pearson_r"] <= oracle + 0.15 for (m, k), r in table.items() if m == "hill" and r["available"])


def test_every_target_profile_carries_hill_predictions_for_each_available_model(knockdown_payload):
    available = {m["key"] for m in knockdown_payload["parameters"]["hill_models"] if m["available"]}
    assert {"hill_fs", "null", "oracle", "vote"} <= available
    for profile in knockdown_payload["target_profiles"]:
        assert set(profile["hill_predictions"]) == available == set(profile["hill_metrics"])
        assert all(len(v) == 8 for v in profile["hill_predictions"].values())


def test_rewiring_sweep_has_a_hill_series_per_rewire_rate(knockdown_payload):
    decay = knockdown_payload["rewire_decay"]
    assert len(decay["hill_r"]) == len(decay["rewire_rates"]) and all(np.isfinite(decay["hill_r"]))


def test_epistasis_screen_runs_on_knockdown_data(knockdown_payload):
    screen = knockdown_payload["combinatorial_screen"]
    assert np.all(np.isfinite(np.array(screen["epistasis_matrix"])))


def test_unknown_perturbation_mode_is_rejected():
    with pytest.raises(ValueError, match="perturbation"):
        vcc.validate_vcc_params(12, 1.8, 0.15, "overexpress")
    with pytest.raises(ValueError, match="perturbation"):
        vcc.run_vcc_comparative_benchmark(num_genes=8, perturbation_mode="nope")
