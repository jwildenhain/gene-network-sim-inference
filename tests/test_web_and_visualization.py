"""
Tests for graph export, 9-state differential visualization, and web server API.
"""

import numpy as np
from grn_sim.visualization.graph_export import (
    export_graphviz_dot,
    export_9state_graphviz_dot,
    compute_new_stats_9state,
    render_graphviz_svg,
    export_graph_json,
    export_comparison_graph_json,
    compute_detailed_confusion_matrix,
)
from grn_sim.web.server import (
    _execute_simulation,
    _execute_inference,
    _build_sos_full_studio,
    _load_archived_run,
    SIMULATION_RUN_STORE,
)
from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator


def test_graphviz_dot_export():
    adj = np.array([
        [0,  1, 0],
        [-1, 0, 1],
        [0, -1, 0],
    ])
    dot = export_graphviz_dot(adj, gene_names=["lexA", "recA", "ssb"])
    assert 'digraph "GRN"' in dot
    assert '"recA" -> "lexA"' in dot
    assert 'arrowhead="normal"' in dot   # Activation
    assert 'arrowhead="tee"' in dot      # Inhibition


def test_9state_stats_and_dot():
    true_adj = np.array([
        [0,  1, -1],
        [1,  0,  0],
        [0, -1,  0],
    ])
    pred_adj = np.array([
        [1,  1,  0],
        [-1, 0, -1],
        [0,  1,  0],
    ])
    res = compute_new_stats_9state(true_adj, pred_adj)
    st = res["tmp_statmat"]
    assert "power" in st
    assert "specificity" in st
    assert "signed_ppv" in st
    assert st["total_errors"] > 0

    a_stats = np.array(res["a_stats"], dtype=int)
    dot9 = export_9state_graphviz_dot(a_stats, ["G1", "G2", "G3"])
    assert 'digraph "GRN_Comparison"' in dot9
    svg = render_graphviz_svg(dot9, engine="dot", matrix=a_stats, gene_names=["G1", "G2", "G3"], is_9state=True)
    assert "<svg" in svg


def test_graph_json_export():
    adj = np.array([
        [0,  1, 0],
        [-1, 0, 0],
        [0,  0, 0],
    ])
    g = export_graph_json(adj, gene_names=["G1", "G2", "G3"])
    assert len(g["nodes"]) == 3
    assert len(g["links"]) == 2
    assert g["links"][0]["type"] in ["activation", "inhibition"]


def test_comparison_graph_json():
    true_adj = np.array([
        [0,  1],
        [-1, 0]
    ])
    pred_adj = np.array([
        [1,  1],
        [1,  0]
    ])
    comp = export_comparison_graph_json(true_adj, pred_adj, ["G1", "G2"])
    statuses = [l["status"] for l in comp["links"]]
    assert "TP_CORRECT" in statuses
    assert "TP_SIGN_ERR" in statuses
    assert "FP" in statuses
    assert "state_counts" in comp


def test_detailed_confusion_matrix():
    true_adj = np.array([
        [1, -1, 0],
        [0,  1, 0],
        [-1, 0, 0]
    ])
    pred_adj = true_adj.copy()
    cmat = compute_detailed_confusion_matrix(true_adj, pred_adj)
    assert cmat["overall_error_pct"] == 0.0
    assert len(cmat["row_errors"]) == 3
    assert len(cmat["col_errors"]) == 3


def test_end_to_end_web_pipeline():
    sim_resp = _execute_simulation({"num_genes": 8, "topology": "power_law", "k_av": 1.5, "seed": 42})
    assert sim_resp["status"] == "success"
    assert len(sim_resp["node_params"]) == 8
    assert len(sim_resp["basal_trajectory"]) == 8
    assert len(sim_resp["perturbation_trajectories"]) == 8

    inf_resp = _execute_inference(["s2g", "g2s", "vote"])
    assert inf_resp["status"] == "success"
    assert "s2g" in inf_resp["tmp_statmat"]
    assert "aic_profile" in inf_resp

    run_id = sim_resp["run_id"]
    assert run_id in SIMULATION_RUN_STORE
    loaded = _load_archived_run(run_id)
    assert loaded["status"] == "success"
    assert loaded["run_id"] == run_id
    assert loaded["settings"]["num_genes"] == 8
    assert loaded["settings"]["topology"] == "power_law"
    assert "s2g" in loaded["inferred"]["tmp_statmat"]

    sos_resp = _build_sos_full_studio()
    assert sos_resp["status"] == "success"
    assert len(sos_resp["gene_names"]) == 9

