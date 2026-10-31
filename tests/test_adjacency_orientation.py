"""
The studio reports every network in the simulator's orientation, T[j, k] = effect of regulator j on target k
(NetworkTopology; the Hill kinetics read the regulators of gene k from column k).

Covers the node inspector, edge parameters, gene profiles (server and client mirror), graph exports, archived runs
saved before the fix, and the E. coli SOS ground truth, which is stored the other way round (A[target, regulator]).
"""

import copy
import json
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest

from grn_sim.benchmarks.benchmark_datasets import (
    load_benchmark_dataset,
    regulator_target_matrix,
    true_matrix_is_target_regulator,
)
from grn_sim.benchmarks.provenance import estimate_biophysical_parameters
from grn_sim.benchmarks.sos_pathway import GENE_NAMES as SOS_GENES, SOS_GROUND_TRUTH
from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import NetworkTopology, TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.kinetics import KineticsKernel
from grn_sim.web import server

NAMES = ["G1", "G2", "G3"]


@pytest.fixture(scope="module")
def chain():
    """G1 -> G2 (activation), G2 -| G3 (inhibition), simulated with the Hill engine."""
    T = np.zeros((3, 3), dtype=int)
    T[0, 1] = 1
    T[1, 2] = -1
    topo = NetworkTopology(adjacency=T, topology_type=TopologyType.HIERARCHICAL, k_av=2 / 3, ratio=0.5)
    net = GeneNetwork(topology=topo, rng=np.random.default_rng(7))
    exp = PerturbationSimulator(perturbation_factor=2.0, measurement_noise=0.0).run(net, seed=7)
    kin = server._extract_kinetics_and_trajectories(net, exp, NAMES)
    return net, exp, kin


def test_chain_node_params(chain):
    _, _, kin = chain
    nodes = {n["name"]: n for n in kin["node_params"]}
    assert (nodes["G1"]["in_degree"], nodes["G1"]["out_degree"]) == (0, 1)
    assert (nodes["G2"]["in_degree"], nodes["G2"]["out_degree"]) == (1, 1)
    assert (nodes["G3"]["in_degree"], nodes["G3"]["out_degree"]) == (1, 0)
    assert nodes["G1"]["activators_in"] == [] and nodes["G1"]["inhibitors_in"] == []
    assert nodes["G2"]["activators_in"] == ["G1"] and nodes["G2"]["inhibitors_in"] == []
    assert nodes["G3"]["activators_in"] == [] and nodes["G3"]["inhibitors_in"] == ["G2"]


def test_chain_edge_params_carry_the_hill_constants_of_that_edge(chain):
    net, _, kin = chain
    p = net.params
    assert set(kin["edge_params"]) == {"0-1", "1-2"}
    assert [e["key"] for e in kin["edge_list"]] == ["0-1", "1-2"]

    act = kin["edge_params"]["0-1"]
    assert (act["source"], act["target"], act["source_idx"], act["target_idx"]) == ("G1", "G2", 0, 1)
    assert act["sign"] == 1 and act["param_symbol_n"] == "n_A"
    assert act["hill_coeff"] == round(float(p.n_k[0, 1]), 4)
    assert act["half_sat_K"] == round(float(p.Ka[0, 1]), 4)

    inh = kin["edge_params"]["1-2"]
    assert (inh["source"], inh["target"], inh["source_idx"], inh["target_idx"]) == ("G2", "G3", 1, 2)
    assert inh["sign"] == -1 and inh["param_symbol_K"] == "K_I"
    assert inh["hill_coeff"] == round(float(p.n_i[1, 2]), 4)
    assert inh["half_sat_K"] == round(float(p.Ki[1, 2]), 4)


def test_edge_params_match_the_edges_the_kinetics_integrate(chain):
    net, _, kin = chain
    k = KineticsKernel(net)
    kernel_edges = {(int(s), int(d)): (float(m), float(km) ** (1.0 / float(m))) for s, d, m, km in zip(k.src, k.dst, k.m, k.k_m)}
    reported = {(e["source_idx"], e["target_idx"]): (e["hill_coeff"], e["half_sat_K"]) for e in kin["edge_list"]}
    assert set(reported) == set(kernel_edges)
    for edge, (n, K) in reported.items():
        assert n == pytest.approx(kernel_edges[edge][0], abs=1e-4)
        assert K == pytest.approx(kernel_edges[edge][1], abs=1e-4)


def test_chain_gene_profiles(chain):
    _, exp, kin = chain
    prof = kin["gene_profiles"]
    assert prof["G1"]["in_regulators"] == []
    assert [(t["name"], t["sign"]) for t in prof["G1"]["out_targets"]] == [("G2", 1)]
    assert [(r["name"], r["sign"], r["type"]) for r in prof["G2"]["in_regulators"]] == [("G1", 1, "activation")]
    assert [(t["name"], t["sign"]) for t in prof["G2"]["out_targets"]] == [("G3", -1)]
    assert [(r["name"], r["sign"], r["type"]) for r in prof["G3"]["in_regulators"]] == [("G2", -1, "inhibition")]
    assert prof["G3"]["out_targets"] == []
    # the regulator curve of G3 is G3's trajectory when its regulator G2 is perturbed
    assert prof["G3"]["regulator_trajectories"]["G2"] == kin["perturbation_trajectories"][1][2]
    flags = {c["perturbed_gene"]: c["is_regulator_perturbation"] for c in prof["G3"]["expression_across_conditions"]}
    assert flags == {"G1": False, "G2": True, "G3": False}


def test_large_network_in_degrees_match_the_topology():
    """The reported failure: a modular network gives every gene a regulator, yet half showed in_degree 0."""
    net = GeneNetwork.create(n=60, k_av=2.0, net_type=TopologyType.MODULAR, ratio=0.5, subdiv=4, seed=3)
    exp = PerturbationSimulator(perturbation_factor=2.0, measurement_noise=0.0).run(net, seed=3)
    kin = server._extract_kinetics_and_trajectories(net, exp, [f"G{i + 1}" for i in range(net.n)], on_demand=True)
    assert [n["in_degree"] for n in kin["node_params"]] == net.topology.in_degrees.tolist()
    assert [n["out_degree"] for n in kin["node_params"]] == net.topology.out_degrees.tolist()


# --- studio runs: graphs and the client-side profile ---------------------------------------------------------


@pytest.fixture
def studio_state():
    saved = dict(server.CURRENT_STATE)
    try:
        yield
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)


def _links(graph):
    return {(link["source_idx"], link["target_idx"]): link["sign"] for link in graph["links"]}


def _dot_edges(dot):
    return set(re.findall(r'"([^"]+)" -> "([^"]+)"', dot))


def test_studio_graphs_draw_regulator_to_target(studio_state):
    sim = server._execute_simulation({"num_genes": 10, "topology": "power_law", "k_av": 1.8, "seed": 11})
    T = np.asarray(server.CURRENT_STATE["network"].topology.adjacency)
    names = sim["gene_names"]
    expected = {(int(j), int(k)): int(np.sign(T[j, k])) for j, k in zip(*np.nonzero(T))}
    assert expected, "the test network needs edges"

    assert sim["true_matrix_orientation"] == "regulator_target" and sim["regulatory_matrix"] is None
    assert _links(sim["graph"]) == expected
    assert {(e["source_idx"], e["target_idx"]) for e in sim["edge_list"]} == set(expected)
    assert _dot_edges(sim["dot_hierarchical"]) == {(names[j], names[k]) for j, k in expected}
    for node in sim["graph"]["nodes"]:
        assert node["in_degree"] == int(np.sum(T[:, node["index"]] != 0))

    inferred = server._execute_inference(["s2g"])
    links = {(link["source_idx"], link["target_idx"]): link for link in inferred["comparison_graphs"]["s2g"]["links"]}
    for edge, sign in expected.items():  # every true edge appears (TP, FN or sign error) with its true sign
        assert links[edge]["true_sign"] == sign


def _client_profile(payload, idx):
    """Runs index.html's buildClientGeneProfile in Node on ``payload``."""
    html = (Path(server.__file__).parent / "static" / "index.html").read_text()
    start = html.index("function buildClientGeneProfile(")
    depth, i = 0, html.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(html[i], 0)
        i += 1
        if depth == 0:
            break
    script = html[start:i] + f"\nconsole.log(JSON.stringify(buildClientGeneProfile({json.dumps(payload)}, {idx})));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_client_gene_profile_mirrors_the_server(chain):
    _, exp, kin = chain
    payload = {
        "gene_names": NAMES,
        "true_matrix": np.asarray(chain[0].topology.adjacency).astype(int).tolist(),
        "X": np.asarray(exp.X).tolist(),
        "perturbation_trajectories": kin["perturbation_trajectories"],
        "basal_trajectory": kin["basal_trajectory"],
    }
    for i, g in enumerate(NAMES):
        client = _client_profile(payload, i)
        assert client["in_regulators"] == kin["gene_profiles"][g]["in_regulators"]
        assert client["out_targets"] == kin["gene_profiles"][g]["out_targets"]

    # SOS payloads: true_matrix is A[target, regulator]; the client must use regulatory_matrix
    sos = {"gene_names": SOS_GENES, "true_matrix": SOS_GROUND_TRUTH.tolist(), "regulatory_matrix": SOS_GROUND_TRUTH.T.tolist()}
    recF = _client_profile(sos, SOS_GENES.index("recF"))
    assert [r["name"] for r in recF["in_regulators"]] == ["rpoD", "rpoS"]


# --- the SOS ground truth (A[target, regulator]) ------------------------------------------------------------


def test_only_sos_is_stored_target_regulator():
    assert true_matrix_is_target_regulator("ecoli_sos")
    for ds in ("dream4_10gene", "norman_2019_tf16", "replogle_2022_p53"):
        assert not true_matrix_is_target_regulator(ds)
    A = load_benchmark_dataset("ecoli_sos")["true_matrix"]
    T = regulator_target_matrix("ecoli_sos", A)
    lexA, recA = SOS_GENES.index("lexA"), SOS_GENES.index("recA")
    assert A[recA, lexA] == -1 and T[lexA, recA] == -1  # LexA represses recA
    dream = load_benchmark_dataset("dream4_10gene")["true_matrix"]
    assert np.array_equal(regulator_target_matrix("dream4_10gene", dream), dream)


@pytest.mark.parametrize("ds_id", ["ecoli_sos", "dream4_10gene"])
def test_biophysical_fit_parameters_follow_the_simulator_orientation(ds_id):
    ds = load_benchmark_dataset(ds_id)
    T = regulator_target_matrix(ds_id, ds["true_matrix"])
    fit = estimate_biophysical_parameters(ds_id)
    p = fit["params"]
    assert np.array_equal(p.Ka > 0, T > 0) and np.array_equal(p.Ki > 0, T < 0)
    X = np.asarray(ds["X"], dtype=float)
    for j, k in zip(*np.nonzero(T)):  # K sits at the regulator's level
        K = p.Ka[j, k] if T[j, k] > 0 else p.Ki[j, k]
        assert K == pytest.approx(max(0.15, float(np.median(np.abs(X[j, :]))) + 0.1))


def test_sos_mode_reports_lexa_as_regulator(studio_state):
    sim = server._execute_simulation({"mode": "sos", "seed": 42})
    lexA, recA, recF = SOS_GENES.index("lexA"), SOS_GENES.index("recA"), SOS_GENES.index("recF")
    assert sim["true_matrix_orientation"] == "target_regulator"
    T = np.asarray(sim["regulatory_matrix"])
    assert np.array_equal(T, np.asarray(sim["true_matrix"]).T)
    assert np.array_equal(server.CURRENT_STATE["network"].topology.adjacency, T)

    nodes = {n["name"]: n for n in sim["node_params"]}
    assert "lexA" in nodes["recA"]["inhibitors_in"]
    assert nodes["recF"]["in_degree"] == 2 and nodes["recF"]["out_degree"] == 6  # rpoD, rpoS -> recF -> 6 SOS genes
    assert sim["edge_params"][f"{lexA}-{recA}"]["source"] == "lexA"
    assert _links(sim["graph"])[(lexA, recA)] == -1 and (recA, lexA) in _links(sim["graph"])
    assert ("rpoS", "recF") in _dot_edges(sim["dot_hierarchical"])
    assert ("recF", "rpoS") not in _dot_edges(sim["dot_hierarchical"])

    profile = server._gene_profile_payload(None, "recF")
    assert [r["name"] for r in profile["in_regulators"]] == ["rpoD", "rpoS"]

    inferred = server._execute_inference(["s2g"])
    links = {(link["source_idx"], link["target_idx"]): link for link in inferred["comparison_graphs"]["s2g"]["links"]}
    assert links[(lexA, recA)]["true_sign"] == -1
    assert links[(SOS_GENES.index("rpoD"), recF)]["true_sign"] == 1


# --- runs archived before the fix ---------------------------------------------------------------------------


def _as_archived_before_the_fix(sim):
    """What the same run looked like when the studio read rows as regulators (every edge reversed)."""
    old = copy.deepcopy(sim)
    old.pop("true_matrix_orientation")
    old.pop("regulatory_matrix", None)
    server._repair_reversed_network_views(old, np.asarray(old["true_matrix"]).T, old["gene_names"])
    return old


def test_archived_runs_from_before_the_fix_are_repaired(studio_state, no_graphviz_dot):
    sim = server._execute_simulation({"num_genes": 7, "topology": "power_law", "k_av": 1.8, "seed": 4})
    inferred = server._execute_inference(["s2g"])
    old = _as_archived_before_the_fix(sim)
    old["inferred"] = copy.deepcopy(inferred)
    old["svg_hierarchical"] = '<svg><g class="graph"><title>stale graphviz drawing</title></g></svg>'
    assert old["node_params"] != sim["node_params"] or not any(n["in_degree"] for n in sim["node_params"])

    server._ensure_graph_views(old)
    assert old["true_matrix_orientation"] == "regulator_target"
    for key in ("node_params", "edge_params", "edge_list", "gene_profiles", "graph"):
        assert old[key] == sim[key], key
    assert _dot_edges(old["dot_hierarchical"]) == _dot_edges(sim["dot_hierarchical"])
    assert "stale graphviz drawing" not in old["svg_hierarchical"]
    assert old["inferred"]["comparison_graphs"]["s2g"]["links"] == inferred["comparison_graphs"]["s2g"]["links"]

    repaired = copy.deepcopy(old)
    server._ensure_graph_views(old)
    assert old == repaired, "the repair runs once"


@pytest.fixture
def no_graphviz_dot(monkeypatch):
    """Built-in renderer only, so the redrawn SVGs are deterministic."""
    import grn_sim.visualization.graph_export as ge
    monkeypatch.setattr(ge.shutil, "which", lambda *_: None)
