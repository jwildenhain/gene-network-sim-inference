"""
Inference is scored in the simulator's orientation, T[j, k] = effect of regulator j on target k.

Covers the linear and CellBox engines (they simulate T, like the Hill engine), the sign convention of the synthetic
benchmark datasets, the datasets whose ground truth is A[target, regulator], and the per-algorithm output
orientation table (grn_sim.inference.orientation) on data from networks with known topology.
"""

import numpy as np
import pytest

from grn_sim.benchmarks.benchmark_datasets import (
    GARDNER_SIGN_DATASETS,
    benchmark_data_model,
    load_benchmark_dataset,
    true_matrix_is_target_regulator,
)
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import NetworkTopology, TopologyType
from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.inference import FastSVD, GeneralToSpecific, NIRAlgorithm, SimpleToGeneral
from grn_sim.inference.orientation import (
    ADDITIVE_GARDNER,
    ADDITIVE_PHYSICAL,
    HILL,
    orient_prediction,
    output_is_target_regulator,
)
from grn_sim.simulator.model_engine import run_simulation_with_engine
from grn_sim.web import server


def _network(T, seed=0):
    topo = NetworkTopology(
        adjacency=T, topology_type=TopologyType.HIERARCHICAL,
        k_av=float(np.sum(T != 0) / T.shape[0]), ratio=0.6,
    )
    return GeneNetwork(topology=topo, rng=np.random.default_rng(seed))


def _random_networks(count, n=10, density=0.2):
    """Signed networks T[regulator, target], alternating cyclic and acyclic."""
    nets = []
    for s in range(count):
        rng = np.random.default_rng(900 + s)
        T = (rng.random((n, n)) < density).astype(int) * rng.choice([-1, 1], (n, n), p=[0.4, 0.6])
        np.fill_diagonal(T, 0)
        if s % 2:
            T = np.triu(T, 1)
            p = rng.permutation(n)
            T = T[np.ix_(p, p)]
        nets.append((T, rng))
    return nets


@pytest.mark.parametrize("engine", ["hill_nonlinear", "linear_jacobian", "cellbox_neural_ode"])
def test_engines_propagate_a_perturbation_along_the_edges_of_T(engine):
    """G1 -> G2 -> G3: perturbing G1 moves G2, perturbing G3 moves nothing upstream."""
    T = np.zeros((3, 3), dtype=int)
    T[0, 1] = 1
    T[1, 2] = 1
    exp = run_simulation_with_engine(_network(T), engine=engine, seed=1)
    assert abs(exp.X[1, 0]) > 0.05
    assert abs(exp.X[2, 1]) > 0.05
    assert abs(exp.X[0, 2]) < 1e-3 and abs(exp.X[1, 2]) < 1e-3
    np.testing.assert_array_equal(exp.true_matrix, T)


def test_synthetic_benchmarks_share_the_gardner_sign():
    for did in ("dream4_10gene", "norman_2019_tf16", "replogle_2022_p53"):
        ds = load_benchmark_dataset(did)
        assert np.all(np.diag(ds["X"]) < 0), did
        assert did in GARDNER_SIGN_DATASETS and benchmark_data_model(did) == ADDITIVE_GARDNER
        assert not true_matrix_is_target_regulator(did)
    assert benchmark_data_model("ecoli_sos") == ADDITIVE_PHYSICAL


def test_real_norman_ground_truth_is_target_regulator():
    """build_real_norman stores A_known[dst, src]: KLF1 -> CNN1 is row CNN1, column KLF1."""
    ds = load_benchmark_dataset("norman_2019_real_16tf")
    names = list(ds["gene_names"])
    A = np.asarray(ds["true_matrix"])
    if "KLF1" in names and "CNN1" in names:
        assert A[names.index("CNN1"), names.index("KLF1")] == 1
    for did in ("ecoli_sos", "norman_2019_real_16tf", "norman_real_16tf", "norman_2019_real_105gene"):
        assert true_matrix_is_target_regulator(did)


def test_orient_prediction_rules():
    m = np.arange(9).reshape(3, 3)
    # Gardner-style algorithms estimate A[target, regulator] on additive data, T on Hill data
    np.testing.assert_array_equal(orient_prediction("s2g", m, ADDITIVE_GARDNER), m.T)
    np.testing.assert_array_equal(orient_prediction("s2g", m, HILL), m)
    # ... and keep A[target, regulator] for a ground truth stored that way (SOS thesis numbers unchanged)
    np.testing.assert_array_equal(orient_prediction("s2g", m, ADDITIVE_PHYSICAL, target_regulator=True), m)
    # regulator-input fits are already transposed to T
    assert not output_is_target_regulator("nir", ADDITIVE_PHYSICAL, perturbation_mode="regulator")
    assert output_is_target_regulator("nir", ADDITIVE_PHYSICAL)
    # GENIE3 returns A[target, regulator] on Hill data; the sweep's "ridge" key is ridge_cv
    # GENIE3 returns A[target, regulator] on X alone (additive data) but runs on X + U on Hill data, where its output
    # matches T[regulator, target] as returned (tests/test_effective_orientation.py)
    assert not output_is_target_regulator("genie3", HILL)
    assert output_is_target_regulator("genie3", ADDITIVE_GARDNER)
    assert output_is_target_regulator("ridge", ADDITIVE_GARDNER) == output_is_target_regulator("ridge_cv", ADDITIVE_GARDNER)
    np.testing.assert_array_equal(orient_prediction("by_chance", m, ADDITIVE_GARDNER, target_regulator=True), m)
    with pytest.raises(ValueError):
        output_is_target_regulator("s2g", "nonsense")


def _recovery(T, pred):
    return evaluate_inference(T, pred).inferred_connections_pct


def test_oriented_gardner_predictions_match_T_on_additive_data():
    """Linear steady state B X = -U from T: oriented predictions recover T, not its transpose.

    Summed over the networks: on a single small network a near-tie can flip with the numpy version (S2G on one of
    the four with numpy 2.5), while the total stays clearly forward (S2G 337 vs 288-293 on numpy 2.0-2.5).
    """
    totals: dict = {}
    for T, rng in _random_networks(4):
        n = T.shape[0]
        k = max(1, int(np.ceil(np.sum(T != 0) / n)))
        U = 2.0 * np.eye(n)
        X = -np.linalg.inv(np.eye(n) - 0.5 * T.T) @ U + rng.normal(0, 0.02, (n, n))
        for name, algo, u_in in (
            ("s2g", SimpleToGeneral(), U),
            ("g2s", GeneralToSpecific(), U),
            ("fast", FastSVD(k_interactions=k), -U),
        ):
            pred = orient_prediction(name, algo.fit(X, u_in).adjacency_pred, ADDITIVE_GARDNER)
            pair = totals.setdefault(name, [0.0, 0.0])
            pair[0] += _recovery(T, pred)
            pair[1] += _recovery(T.T, pred)
    for name, (forward, reverse) in totals.items():
        assert forward > reverse, (name, forward, reverse)


def test_oriented_predictions_match_T_on_hill_data():
    # GENIE3 is left out: it regresses each gene on the others across the perturbation columns, where regulator and
    # target co-vary in both directions, so its data-level orientation is weak and flips with the scikit-learn
    # version (forward wins on 16/20 networks with scikit-learn 1.6, 5/20 with 1.9). Its output orientation
    # (W[target, regulator]) is fixed by construction.
    totals = {"s2g": [0.0, 0.0], "nir": [0.0, 0.0]}
    for i, (T, _rng) in enumerate(_random_networks(4)):
        n = T.shape[0]
        k = max(1, int(np.ceil(np.sum(T != 0) / n)))
        exp = run_simulation_with_engine(_network(T, i), engine="hill_nonlinear", seed=i)
        for name, algo in (
            ("s2g", SimpleToGeneral()),
            ("nir", NIRAlgorithm(k_interactions=k, perturbation_mode="regulator", adaptive_k=True)),
        ):
            res = algo.fit(exp.X, exp.U)
            pred = orient_prediction(name, res.adjacency_pred, HILL,
                                     perturbation_mode=res.diagnostics.get("perturbation_mode", "additive"))
            totals[name][0] += _recovery(T, pred)
            totals[name][1] += _recovery(T.T, pred)
    for name, (forward, reverse) in totals.items():
        assert forward > reverse, (name, forward, reverse)


@pytest.mark.parametrize("run, expected", [
    ({"mode": "benchmark_dream4_10gene", "settings": {"loading_mode": "wrangled_direct"}}, ADDITIVE_GARDNER),
    ({"mode": "sos"}, ADDITIVE_PHYSICAL),
    ({"mode": "insilico", "settings": {"dataset_source": "norman_2019_real_16tf", "loading_mode": "wrangled_direct"}}, ADDITIVE_PHYSICAL),
    ({"mode": "benchmark_ecoli_sos", "settings": {"dataset_source": "ecoli_sos", "loading_mode": "biophysical_fit"}}, HILL),
    ({"mode": "insilico", "settings": {"kinetics_engine": "linear_jacobian"}}, ADDITIVE_PHYSICAL),
    ({"mode": "insilico", "settings": {"kinetics_engine": "biophysical"}}, HILL),
    ({"mode": "insilico", "settings": {}}, HILL),
])
def test_inference_data_model_of_a_run(run, expected):
    assert server._inference_data_model(run) == expected
