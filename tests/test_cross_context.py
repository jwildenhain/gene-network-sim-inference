"""Cross-context transfer predictors (benchmarks/cross_context.py) and their place in the Virtual Cell Challenge benchmark."""

from functools import partial

import numpy as np
import pytest

from grn_sim.benchmarks import cross_context as cc
from grn_sim.benchmarks import vcc_benchmark as vcc
from grn_sim.simulator.experiment import PerturbationSimulator

N = 8
SEEN = tuple(range(6))   # perturbations 6 and 7 are held out in the target context


def problem(seed=0, scale=None, noise=0.02):
    """Three noisy copies of one response matrix as sources; the target is that matrix scaled gene by gene."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(N, N)) * 0.4
    np.fill_diagonal(truth, -0.8)
    scale = np.ones(N) if scale is None else np.asarray(scale, float)
    sources = tuple(truth + noise * rng.normal(size=(N, N)) for _ in range(3))
    target = scale[:, None] * truth
    design = -0.8 * np.eye(N)
    control = np.linspace(0.5, 2.0, N)
    return cc.TransferData(sources, target, design, control, SEEN)


def by_key(key):
    return cc.TRANSFER_BY_KEY[key]


ALL_KEYS = [p.key for p in cc.TRANSFER_PREDICTORS]


# --- the contract ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ALL_KEYS)
def test_every_predictor_gives_a_finite_matrix_of_the_right_shape(key):
    out = cc.predict_matrix(by_key(key), problem())
    assert out.shape == (N, N) and np.all(np.isfinite(out))


@pytest.mark.parametrize("key", ALL_KEYS)
def test_held_out_target_columns_are_never_read(key):
    data = problem()
    garbage = data.target.copy()
    garbage[:, 6:] = np.nan   # a read would poison the result
    base = cc.predict_matrix(by_key(key), data)
    other = cc.predict_matrix(by_key(key), cc.TransferData(data.sources, garbage, data.design, data.control, data.seen))
    assert np.allclose(base, other) and np.all(np.isfinite(other))


@pytest.mark.parametrize("key", ALL_KEYS)
def test_a_seen_column_is_predicted_without_its_own_target_measurements(key):
    data = problem(scale=np.linspace(0.6, 1.8, N))
    base = cc.predict_matrix(by_key(key), data)
    for j in data.seen:
        changed = data.target.copy()
        changed[:, j] = 99.0
        again = cc.predict_matrix(by_key(key), cc.TransferData(data.sources, changed, data.design, data.control, data.seen))
        assert np.allclose(base[:, j], again[:, j]), f"column {j} was predicted from its own target column"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_predictors_are_deterministic(key):
    data = problem()
    assert np.array_equal(cc.predict_matrix(by_key(key), data), cc.predict_matrix(by_key(key), data))


def test_data_validation_rejects_inconsistent_inputs():
    data = problem()
    with pytest.raises(ValueError, match="at least one source"):
        cc.TransferData((), data.target, data.design, data.control, data.seen)
    with pytest.raises(ValueError, match="shape"):
        cc.TransferData((data.sources[0][:4],), data.target, data.design, data.control, data.seen)
    with pytest.raises(ValueError, match="control"):
        cc.TransferData(data.sources, data.target, data.design, data.control[:3], data.seen)
    with pytest.raises(ValueError, match="seen"):
        cc.TransferData(data.sources, data.target, data.design, data.control, (1, 1))


@pytest.mark.parametrize("key", ALL_KEYS)
def test_degenerate_inputs_do_not_crash(key):
    zero = np.zeros((N, N))
    flat = cc.TransferData((zero, zero), zero, zero, np.ones(N), SEEN)
    assert np.all(np.isfinite(cc.predict_matrix(by_key(key), flat)))
    one_seen = cc.TransferData(problem().sources, problem().target, problem().design, np.ones(N), (3,))
    assert np.all(np.isfinite(cc.predict_matrix(by_key(key), one_seen)))
    all_seen = cc.TransferData(problem().sources, problem().target, problem().design, np.ones(N), tuple(range(N)))
    assert np.all(np.isfinite(cc.predict_matrix(by_key(key), all_seen)))


# --- what each predictor does -----------------------------------------------------------------------------

def test_the_reference_is_the_plain_source_average():
    data = problem()
    assert np.allclose(cc.predict_matrix(by_key("atlas_mean"), data), np.mean(data.sources, axis=0))


def test_calibration_recovers_a_known_per_gene_scale_and_the_default_shrinks_towards_one():
    scale = np.array([0.5, 1.5, 1.0, 2.0, 0.7, 1.2, 1.0, 0.9])
    data = problem(scale=scale, noise=0.0)
    train = list(data.seen)
    sharp = cc._fit_atlas_calibrated(data.sources, data.target, data.design, data.control, train, ridge=1e-9)
    assert np.allclose(sharp.scale, scale, atol=1e-3)
    default = by_key("atlas_calibrated").fit(data.sources, data.target, data.design, data.control, train)
    assert np.all(np.abs(default.scale - 1.0) <= np.abs(scale - 1.0) + 1e-9), "shrinkage never overshoots the evidence"
    assert default.scale[0] > sharp.scale[0], "a gene with scale 0.5 is pulled back towards 1"


def test_calibration_gives_more_weight_to_the_source_that_explains_the_target():
    data = problem(noise=0.0)
    bad = data.sources[0] + np.random.default_rng(1).normal(size=(N, N))
    model = cc._fit_atlas_calibrated((data.sources[1], bad), data.target, data.design, data.control, list(data.seen))
    assert model.weights[0] > model.weights[1] and np.isclose(model.weights.sum(), 1.0)


def test_calibration_never_flips_a_sign_or_stretches_beyond_the_limit():
    data = problem(scale=np.full(N, -2.0), noise=0.0)    # the target is the mirror image of the sources
    model = by_key("atlas_calibrated").fit(data.sources, data.target, data.design, data.control, list(data.seen))
    assert np.all(model.scale >= 0.0) and np.all(model.scale <= cc.MAX_SCALE)


def test_the_consensus_gate_keeps_only_entries_all_sources_agree_on():
    a = np.array([[-0.8, 0.5, 0.001], [0.4, -0.8, -0.4], [0.0, 0.3, -0.8]])
    b = a.copy()
    b[0, 1] = -0.5          # sign disagreement
    consensus, gate = cc._consensus_gate((a, b, a), noise_scale=0.5)
    assert gate[0, 1] == 0.0 and gate[0, 0] == 1.0 and gate[1, 2] == 1.0
    assert gate[0, 2] == 0.0, "a response below the consensus noise floor is dropped"
    assert gate[2, 0] == 0.0, "an exact zero has no sign to agree on"


def test_the_global_scale_of_the_summary_model_recovers_a_known_factor():
    data = problem(noise=0.0)
    consensus, gate = cc._consensus_gate(data.sources, noise_scale=0.5)
    target = 1.7 * gate * consensus
    model = cc._fit_transpert_style(data.sources, target, data.design, data.control, list(data.seen))
    assert model.scale == pytest.approx(1.7, rel=1e-6)
    assert np.allclose(model.predict(data.sources, data.design, data.control, [6, 7]), target[:, [6, 7]])


def test_the_regression_learns_a_context_specific_on_target_effect():
    data = problem(noise=0.0)
    target = data.target.copy()
    target[np.arange(N), np.arange(N)] = -1.4      # the on-target drop is stronger in this context than in the sources
    fit = partial(cc._fit_pseudobulk_ridge, ridge=1e-8)
    model = fit(data.sources, target, data.design, data.control, list(data.seen))
    predicted = model.predict(data.sources, data.design, data.control, [6, 7])
    plain = np.mean(data.sources, axis=0)[:, [6, 7]]
    assert abs(predicted[6, 0] - (-1.4)) < abs(plain[6, 0] - (-1.4)) / 2


def test_a_fitted_model_can_be_applied_to_another_context():
    data = problem()
    model = by_key("pseudobulk_ridge").fit(data.sources, data.target, data.design, data.control, list(data.seen))
    out = model.predict(data.sources, 2.0 * data.design, data.control * 1.5, list(range(N)))
    assert out.shape == (N, N) and np.all(np.isfinite(out))


# --- the Virtual Cell Challenge benchmark -----------------------------------------------------------------

@pytest.fixture(scope="module")
def payload():
    return vcc.run_vcc_comparative_benchmark(num_genes=8, k_av=1.8, seed=7, rewire_fractions=[0.0, 0.3], perturbation_mode="knockdown")


def rows(payload):
    return {(r["method"], r["key"]): r for r in payload["response_table"]}


def test_every_transfer_predictor_has_a_row_and_a_model_entry(payload):
    table = rows(payload)
    models = {m["key"]: m for m in payload["parameters"]["transfer_models"]}
    assert list(models) == ALL_KEYS
    for key in ALL_KEYS:
        row = table[("transfer", key)]
        assert row["available"] and models[key]["available"] and models[key]["description"]
        assert set(row["held_out"]) >= {"pearson_r", "cosine_sim", "rmse", "offtarget_r"}


def test_the_source_average_equals_atlasshift_on_seen_columns_but_not_on_held_out_ones(payload):
    table = rows(payload)
    average, atlas = table[("transfer", "atlas_mean")], table[("reference", "atlasshift")]
    assert average["seen"] == atlas["seen"], "same prediction for the seen columns"
    assert atlas["held_out"]["pearson_r"] == 0.0 and average["held_out"]["pearson_r"] != 0.0, \
        "AtlasShift predicts zeros for held-out genes by construction; the source average uses the atlas columns"


def test_target_profiles_carry_transfer_predictions_for_every_gene(payload):
    for profile in payload["target_profiles"]:
        assert set(profile["transfer_predictions"]) == set(ALL_KEYS) == set(profile["transfer_metrics"])
        assert all(len(v) == 8 for v in profile["transfer_predictions"].values())


def test_the_rewiring_sweep_has_a_series_per_calibrated_predictor(payload):
    decay = payload["rewire_decay"]
    for series in ("calibrated_r", "transpert_r", "pseudobulk_r"):
        assert len(decay[series]) == len(decay["rewire_rates"]) and all(np.isfinite(decay[series]))


def test_the_shared_context_builder_reproduces_the_benchmark_target(payload):
    sim = PerturbationSimulator(1.0, 0.01, perturbation_mode="knockdown", knockdown_efficiency=vcc.KNOCKDOWN_EFFICIENCY)
    sources, target = vcc._simulate_contexts(8, 1.8, 7, 0.15, sim)
    assert [name for name, _ in sources] == vcc.SOURCE_NAMES
    for j, profile in enumerate(payload["target_profiles"]):
        assert np.allclose(profile["y_true"], np.round(target.X[:, j], 4))
