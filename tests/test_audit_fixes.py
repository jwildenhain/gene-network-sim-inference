"""Regressions for the UI/UX audit: request validation, archived sweep runs, oscillating networks."""

import copy

import numpy as np
import pytest

from grn_sim.web import server


@pytest.mark.parametrize("params, field", [
    ({"seed": -5}, "seed"),
    ({"seed": 2 ** 40}, "seed"),
    ({"seed": "abc"}, "seed"),
    ({"num_genes": 3}, "num_genes"),
    ({"num_genes": 100000}, "num_genes"),
    ({"noise": 2}, "noise"),
    ({"act_ratio": -0.1}, "act_ratio"),
    ({"k_av": float("nan")}, "k_av"),
])
def test_simulation_parameters_are_range_checked(params, field):
    with pytest.raises(ValueError, match=field):
        server._validate_sim_params(params)


def test_validation_keeps_what_the_simulator_accepted_before():
    cleaned = server._validate_sim_params({"seed": "7", "num_genes": 12.0, "topology": "whatever", "k_av": "1.5", "seed2": 1})
    assert cleaned["seed"] == 7 and cleaned["num_genes"] == 12 and cleaned["topology"] == "whatever"
    assert server._validate_sim_params({"seed": ""}) == {}  # empty means "not given", as before
    assert server._validate_sim_params({"seed": 0})["seed"] == 0  # 0 is a real seed, not "use 42"
    # benchmark data sets ignore the gene count, so it is not range-checked for them
    assert server._validate_sim_params({"mode": "sos", "num_genes": 3})["num_genes"] == 3
    with pytest.raises(ValueError, match="topology"):
        server._validate_sim_params({"topology": "whatever"}, strict=True)


def test_archived_sweep_runs_get_edge_counts_and_an_edge_list():
    server._execute_simulation({"num_genes": 8, "seed": 3, "topology": "random", "k_av": 2.0})
    payload = copy.deepcopy(server.SIMULATION_RUN_STORE[server.CURRENT_STATE["run_id"]]["sim_payload"])
    truth = np.array(payload["true_matrix"])
    for key in ("total_edges", "activations", "inhibitions", "edge_list", "edge_params", "node_params"):
        payload.pop(key, None)  # what the benchmark sweep stores: matrices only
    server._ensure_graph_views(payload)
    assert payload["total_edges"] == int(np.count_nonzero(truth))
    assert payload["activations"] + payload["inhibitions"] == payload["total_edges"]
    assert len(payload["edge_list"]) == payload["total_edges"]
    assert payload["kinetics_stored"] is False
    first = payload["edge_list"][0]
    assert first["key"] == f"{first['source_idx']}-{first['target_idx']}" and first["sign"] in (1, -1)


def test_runs_that_already_have_edge_details_are_left_alone():
    server._execute_simulation({"num_genes": 6, "seed": 4, "topology": "random", "k_av": 2.0})
    payload = copy.deepcopy(server.SIMULATION_RUN_STORE[server.CURRENT_STATE["run_id"]]["sim_payload"])
    before = copy.deepcopy(payload["edge_list"])
    server._ensure_graph_views(payload)
    assert payload["edge_list"] == before and "kinetics_stored" not in payload


def test_perturb_seq_refuses_an_oscillating_network_instead_of_hanging():
    server._execute_simulation({"num_genes": 12, "seed": 4321, "topology": "random", "dynamics_preset": "oscillating"})
    assert server.CURRENT_STATE["steady_state_diagnostics"]["oscillating_detected"]
    with pytest.raises(ValueError, match="oscillates"):
        server._perturb_seq_realism_payload(None, 5, "knockdown", 0, "lna", "current", True, None)


def test_limit_cycle_detector_separates_cycles_from_slow_convergence():
    from grn_sim.simulator.steady_state import SteadyStateSolver

    solver = SteadyStateSolver()
    n = solver.CYCLE_MIN_CHUNKS
    t = np.arange(4 * n)
    cycle = [(np.array([1.0 + 0.5 * np.sin(k)]), 0.3) for k in t]
    assert solver._is_limit_cycle(cycle)
    converging = [(np.array([1.0 + 0.5 * 0.7 ** k]), 0.3 * 0.7 ** k) for k in t]
    assert not solver._is_limit_cycle(converging)
    assert not solver._is_limit_cycle(cycle[: 2 * n - 1])  # too early to tell
    drifting_but_improving = [(np.array([1.0 + 0.05 * k]), 0.3 * 0.9 ** k) for k in t]
    assert not solver._is_limit_cycle(drifting_but_improving)


@pytest.mark.parametrize("engine", ["linear_jacobian", "cellbox_neural_ode"])
def test_notears_no_longer_returns_an_empty_graph_on_extreme_range_data(engine):
    """Expression spanning thousands used to freeze NOTEARS at W = 0 (an empty prediction, F1 = 0)."""
    server._execute_simulation({"num_genes": 12, "seed": 14, "topology": "random", "k_av": 1.5, "noise": 0.02,
                                "kinetics_engine": engine})
    exp = server.CURRENT_STATE["experiment"]
    assert float(np.max(np.abs(exp.X))) > 100  # the regime that failed
    metrics = server._execute_inference(["notears"])["metrics"]["notears"]
    assert metrics["true_positives"] + metrics["false_positives"] > 0
    assert metrics["f1_score"] > 0.1


def test_notears_is_untouched_on_ordinary_data():
    from grn_sim.inference.notears import NOTEARSAlgorithm

    server._execute_simulation({"num_genes": 12, "seed": 11, "topology": "power_law", "k_av": 1.5, "noise": 0.02})
    exp = server.CURRENT_STATE["experiment"]
    assert float(np.max(np.abs(exp.X - exp.X[:, :1]))) < NOTEARSAlgorithm.RESCALE_ABOVE
    a = NOTEARSAlgorithm().fit(exp.X, exp.U)
    b = NOTEARSAlgorithm().fit(exp.X.copy(), exp.U.copy())
    assert np.array_equal(a.adjacency_pred, b.adjacency_pred) and np.allclose(a.A_est, b.A_est)
