"""
Tests for graph export, visualization, and web server API.
"""

import numpy as np
import json
from grn_sim.visualization.graph_export import (
    export_graphviz_dot,
    export_graph_json,
    export_comparison_graph_json,
    compute_detailed_confusion_matrix,
)
from grn_sim.web.server import GRNWebHandler, CURRENT_STATE
from grn_sim.core.topology import generate_topology, TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator


def test_graphviz_dot_export():
    adj = np.array([
        [0,  1, 0],
        [-1, 0, 1],
        [0, -1, 0],
    ])
    dot = export_graphviz_dot(adj, gene_names=["lexA", "recA", "ssb"])
    assert "digraph \"GRN\"" in dot
    assert "\"recA\" -> \"lexA\"" in dot
    assert "arrowhead=\"vee\"" in dot   # Activation
    assert "arrowhead=\"tee\"" in dot   # Inhibition


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
    # Inferred has 1 correct TP, 1 sign error, and 1 FP
    pred_adj = np.array([
        [1,  1],
        [1,  0]
    ])
    comp = export_comparison_graph_json(true_adj, pred_adj, ["G1", "G2"])
    statuses = [l["status"] for l in comp["links"]]
    assert "TP_CORRECT" in statuses
    assert "TP_SIGN_ERR" in statuses
    assert "FP" in statuses


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
    # Simulate a small network
    network = GeneNetwork.create(
        n=6,
        k_av=1.5,
        net_type=TopologyType.RANDOM,
        seed=42,
    )
    simulator = PerturbationSimulator(measurement_noise=0.01)
    experiment = simulator.run(network, seed=42)

    assert experiment.X.shape == (6, 6)
    dot = export_graphviz_dot(network.topology.adjacency)
    assert "digraph" in dot
