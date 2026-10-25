"""The hierarchical view comes in two versions - "layered" (regulators on top) and the compact "grid"."""

import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from grn_sim.visualization import graph_export
from grn_sim.visualization.graph_export import render_graphviz_svg
from grn_sim.web import server

NS = "{http://www.w3.org/2000/svg}"
CHAIN = np.zeros((6, 6), dtype=int)
for k in range(5):
    CHAIN[k + 1, k] = 1  # 0 -> 1 -> ... -> 5


def node_rows(svg):
    root = ET.fromstring(svg)
    return sorted({round(float(e.get("cy"))) for e in root.iter(NS + "ellipse")})


def arrow_heads(svg):
    return len([e for e in ET.fromstring(svg).iter() if e.get("data-arrowhead")])


@pytest.fixture
def no_graphviz(monkeypatch):
    monkeypatch.setattr(graph_export.shutil, "which", lambda *_: None)


def test_unknown_layout_is_rejected(no_graphviz):
    with pytest.raises(ValueError):
        render_graphviz_svg("", engine="dot", matrix=CHAIN, layout="radial")


def test_layered_and_grid_are_different_arrangements_of_the_same_edges(no_graphviz):
    layered = render_graphviz_svg("", engine="dot", matrix=CHAIN, layout="layered")
    grid = render_graphviz_svg("", engine="dot", matrix=CHAIN, layout="grid")

    assert layered != grid
    assert len(node_rows(layered)) == 6, "the 6-gene chain has one row per level"
    assert len(node_rows(grid)) < 6, "the grid packs nodes into a few shared rows regardless of direction"
    assert arrow_heads(layered) == arrow_heads(grid) == 5


def test_default_layout_is_layered(no_graphviz):
    assert render_graphviz_svg("", engine="dot", matrix=CHAIN) == render_graphviz_svg("", engine="dot", matrix=CHAIN, layout="layered")


def test_grid_keeps_the_fixed_fit_to_container_box(no_graphviz):
    root = ET.fromstring(render_graphviz_svg("", engine="dot", matrix=CHAIN, layout="grid"))
    assert root.get("viewBox") == "0.0 0.0 560.0 420.0"
    assert (root.get("width"), root.get("height")) == ("100%", "100%")


def test_grid_never_calls_graphviz_but_layered_does(monkeypatch):
    calls = []
    monkeypatch.setattr(graph_export.shutil, "which", lambda name: f"/fake/{name}")

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"<svg>graphviz</svg>", stderr=b"")

    monkeypatch.setattr(graph_export.subprocess, "run", fake_run)

    assert render_graphviz_svg("digraph{}", engine="dot", matrix=CHAIN, layout="layered") == "<svg>graphviz</svg>"
    assert len(calls) == 1
    grid = render_graphviz_svg("digraph{}", engine="dot", matrix=CHAIN, layout="grid")
    assert len(calls) == 1, "the grid is always drawn by the built-in renderer"
    assert "graphviz" not in grid and arrow_heads(grid) == 5


def test_spring_view_ignores_the_layout_option(no_graphviz):
    assert render_graphviz_svg("", engine="neato", matrix=CHAIN, layout="grid") == render_graphviz_svg("", engine="neato", matrix=CHAIN)


def test_9state_graphs_support_both_layouts(no_graphviz):
    states = np.zeros((4, 4), dtype=int)
    states[1, 0], states[2, 1], states[3, 2] = 1, -1, 2
    for layout in ("layered", "grid"):
        svg = render_graphviz_svg("", engine="dot", matrix=states, is_9state=True, layout=layout)
        assert arrow_heads(svg) == 3


def test_server_payloads_carry_both_versions(no_graphviz, monkeypatch, tmp_path):
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    saved = dict(server.CURRENT_STATE)
    try:
        sim = server._execute_simulation({"num_genes": 8, "topology": "power_law", "k_av": 1.8, "seed": 3})
        assert sim["svg_hierarchical"] and sim["svg_hierarchical_grid"]
        assert sim["svg_hierarchical"] != sim["svg_hierarchical_grid"]
        assert server.CURRENT_STATE["svg_hierarchical_grid"] == sim["svg_hierarchical_grid"]
        assert arrow_heads(sim["svg_hierarchical"]) == arrow_heads(sim["svg_hierarchical_grid"]) == sim["total_edges"]

        inferred = server._execute_inference(["s2g", "vote"])
        assert set(inferred["svgs_dot_grid"]) == set(inferred["svgs_dot"]) == {"s2g", "vote"}
        assert all(inferred["svgs_dot"][a] != inferred["svgs_dot_grid"][a] for a in inferred["svgs_dot"])
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)


# --- archived runs ------------------------------------------------------------------------------------

ARROWLESS = '<svg viewBox="0 0 560 420" width="100%" height="100%"><line x1="1" y1="1" x2="9" y2="9" stroke="#10b981"/></svg>'
GRAPHVIZ_SVG = '<svg width="62pt"><g id="graph0" class="graph"><title>from graphviz</title></g></svg>'


@pytest.fixture
def archived_run(no_graphviz, monkeypatch, tmp_path):
    """A freshly simulated + inferred run, returned as (sim_payload, inferred_payload)."""
    import copy
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "archive.sqlite"))
    saved = dict(server.CURRENT_STATE)
    try:
        sim = server._execute_simulation({"num_genes": 8, "topology": "power_law", "k_av": 1.8, "seed": 5})
        inferred = server._execute_inference(["s2g", "vote"])
        yield copy.deepcopy(sim), copy.deepcopy(inferred)
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)


def legacy_copy(sim, inferred):
    """What a run archived by an older version looks like: no grid versions, arrow-less built-in SVGs."""
    sim = dict(sim)
    sim["svg_hierarchical"], sim["svg_spring"] = ARROWLESS, ARROWLESS
    sim.pop("svg_hierarchical_grid", None)
    inferred = dict(inferred)
    inferred["svgs_dot"] = {a: ARROWLESS for a in inferred["svgs_dot"]}
    inferred["svgs_neato"] = {a: ARROWLESS for a in inferred["svgs_neato"]}
    inferred.pop("svgs_dot_grid", None)
    sim["inferred"] = inferred
    return sim


def test_ensure_graph_views_upgrades_a_legacy_run(archived_run):
    sim, inferred = archived_run
    legacy = legacy_copy(sim, inferred)
    server._ensure_graph_views(legacy)

    assert arrow_heads(legacy["svg_hierarchical"]) == arrow_heads(legacy["svg_spring"]) == sim["total_edges"]
    assert arrow_heads(legacy["svg_hierarchical_grid"]) == sim["total_edges"]
    assert legacy["svg_hierarchical"] != legacy["svg_hierarchical_grid"]
    for alg in ("s2g", "vote"):
        assert arrow_heads(legacy["inferred"]["svgs_dot"][alg]) > 0
        assert arrow_heads(legacy["inferred"]["svgs_neato"][alg]) > 0
        assert legacy["inferred"]["svgs_dot_grid"][alg]


def test_ensure_graph_views_keeps_graphviz_drawings_and_is_idempotent(archived_run):
    sim, inferred = archived_run
    legacy = legacy_copy(sim, inferred)
    legacy["svg_hierarchical"] = GRAPHVIZ_SVG
    legacy["inferred"]["svgs_dot"]["s2g"] = GRAPHVIZ_SVG

    server._ensure_graph_views(legacy)
    assert legacy["svg_hierarchical"] == GRAPHVIZ_SVG, "SVGs drawn by Graphviz are kept as stored"
    assert legacy["inferred"]["svgs_dot"]["s2g"] == GRAPHVIZ_SVG
    assert legacy["svg_hierarchical_grid"], "but the grid version is still added next to them"

    import copy
    first = copy.deepcopy(legacy)
    server._ensure_graph_views(legacy)
    assert legacy == first


def test_ensure_graph_views_ignores_payloads_without_a_network():
    assert server._ensure_graph_views({"true_matrix": [], "gene_names": []}) == {"true_matrix": [], "gene_names": []}
    assert server._ensure_graph_views({}) == {}


def test_loading_a_legacy_run_from_the_database_adds_the_new_versions(archived_run):
    from grn_sim.web import database
    sim, inferred = archived_run
    legacy = legacy_copy(sim, inferred)
    legacy_inferred = legacy.pop("inferred")
    database.save_simulation_to_db(
        "legacy-run-1", legacy, legacy_inferred, history_entry={"run_id": "legacy-run-1", "timestamp": "2020-01-01"}
    )
    server.SIMULATION_RUN_STORE.pop("legacy-run-1", None)

    loaded = server._load_archived_run("legacy-run-1")
    assert loaded["svg_hierarchical_grid"]
    assert arrow_heads(loaded["svg_hierarchical"]) == sim["total_edges"]
    assert loaded["inferred"]["svgs_dot_grid"].keys() == loaded["inferred"]["svgs_dot"].keys()
    assert server.CURRENT_STATE["svg_hierarchical_grid"] == loaded["svg_hierarchical_grid"]
    server.SIMULATION_RUN_STORE.pop("legacy-run-1", None)


def test_archived_runs_stored_without_a_status_or_response_matrix_still_load_as_successes(archived_run):
    """Sweep-generated archive rows carry no "status"; the page ignores a load response unless it says success."""
    from grn_sim.web import database
    sim, inferred = archived_run
    sweep_like = legacy_copy(sim, inferred)
    sweep_inferred = sweep_like.pop("inferred")
    sweep_like.pop("status", None)
    database.save_simulation_to_db(
        "sweep-like-run-1", sweep_like, sweep_inferred, history_entry={"run_id": "sweep-like-run-1", "timestamp": "2020-01-02"}
    )
    server.SIMULATION_RUN_STORE.pop("sweep-like-run-1", None)
    loaded = server._load_archived_run("sweep-like-run-1")
    assert loaded["status"] == "success"

    no_matrix = dict(sweep_like)
    no_matrix.pop("X", None)
    database.save_simulation_to_db(
        "sweep-like-run-2", no_matrix, sweep_inferred, history_entry={"run_id": "sweep-like-run-2", "timestamp": "2020-01-03"}
    )
    server.SIMULATION_RUN_STORE.pop("sweep-like-run-2", None)
    loaded = server._load_archived_run("sweep-like-run-2")
    assert loaded["status"] == "success" and loaded["response_type"] == "responsive" and isinstance(loaded["response_dynamics"], dict)
    for run_id in ("sweep-like-run-1", "sweep-like-run-2"):
        server.SIMULATION_RUN_STORE.pop(run_id, None)


def test_archived_runs_report_the_stored_classification_and_flag_a_disagreement(archived_run):
    """The archive table shows the stored label; the detail view must not contradict it silently."""
    from grn_sim.web import database
    sim, inferred = archived_run
    assert sim["response_type"] != "oscillating", "the fixture run converges, so its trajectories alone say so"

    stored = legacy_copy(sim, inferred)
    stored_inferred = stored.pop("inferred")
    stored["response_type"] = "oscillating"           # what an older sweep stored for the same data
    stored["response_dynamics"] = {**(stored.get("response_dynamics") or {}), "response_type": "oscillating"}
    database.save_simulation_to_db("stored-osc-1", stored, stored_inferred, history_entry={"run_id": "stored-osc-1", "timestamp": "2020-02-01"})
    server.SIMULATION_RUN_STORE.pop("stored-osc-1", None)
    loaded = server._load_archived_run("stored-osc-1")
    assert loaded["response_type"] == "oscillating"
    assert "Stored as 'oscillating'" in loaded["classification_note"] and loaded["response_dynamics"]["max_tail_fluctuation"] >= 0

    agree = legacy_copy(sim, inferred)
    agree_inferred = agree.pop("inferred")
    database.save_simulation_to_db("stored-agree-1", agree, agree_inferred, history_entry={"run_id": "stored-agree-1", "timestamp": "2020-02-02"})
    server.SIMULATION_RUN_STORE.pop("stored-agree-1", None)
    loaded = server._load_archived_run("stored-agree-1")
    assert loaded["response_type"] == sim["response_type"] and "classification_note" not in loaded
    for run_id in ("stored-osc-1", "stored-agree-1"):
        server.SIMULATION_RUN_STORE.pop(run_id, None)


# --- real-world benchmark payload -----------------------------------------------------------------------

def test_benchmark_payload_carries_every_view(no_graphviz):
    payload = server._build_benchmark_studio("ecoli_sos")
    assert payload["svg_true_dot"] and payload["svg_true_neato"] and payload["svg_true_dot_grid"]
    assert payload["svg_true_dot_grid"] != payload["svg_true_dot"]
    assert payload["true_graph"]["nodes"] and payload["true_graph"]["links"]
    algs = set(payload["svgs_dot"])
    assert algs and set(payload["svgs_dot_grid"]) == set(payload["svgs_neato"]) == set(payload["comparison_graphs"]) == algs
