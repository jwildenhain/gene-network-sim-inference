"""Above STORED_SVG_MAX_GENES the per-algorithm comparison SVGs are not rendered or stored; the data the page draws from is."""
from grn_sim.web import server


def _run(tmp_path, monkeypatch, genes):
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    saved = dict(server.CURRENT_STATE)
    try:
        server._execute_simulation({"num_genes": genes, "topology": "modular", "k_av": 1.8, "seed": 5})
        return server._execute_inference(["by_chance"])
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)


def test_small_network_keeps_its_svgs(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, 12)
    assert out["svgs_dot"]["by_chance"] and out["svgs_neato"]["by_chance"] and out["svgs_dot_grid"]["by_chance"]


def test_large_network_stores_graph_data_but_no_svgs(tmp_path, monkeypatch):
    genes = server.STORED_SVG_MAX_GENES + 20
    out = _run(tmp_path, monkeypatch, genes)
    assert out["svgs_dot"] == {} and out["svgs_dot_grid"] == {} and out["svgs_neato"] == {}
    assert len(out["comparison_graphs"]["by_chance"]["nodes"]) == genes  # what the page draws (as modules)
    assert out["dots"]["by_chance"] and out["pred_matrices"]["by_chance"]["sparse"] is True


def test_no_graph_is_rendered_above_the_limit_anywhere(tmp_path, monkeypatch):
    """Generation, inference and reloading from the archive never call the SVG renderer for a large network."""
    monkeypatch.setenv("RGN_DB_PATH", str(tmp_path / "t.sqlite"))
    genes = server.STORED_SVG_MAX_GENES + 20
    calls = []
    real = server.render_graphviz_svg
    monkeypatch.setattr(server, "render_graphviz_svg", lambda *a, **k: calls.append(1) or real(*a, **k))
    saved = dict(server.CURRENT_STATE)
    try:
        sim = server._execute_simulation({"num_genes": genes, "topology": "modular", "k_av": 1.8, "seed": 5})
        assert sim["svg_hierarchical"] == sim["svg_hierarchical_grid"] == sim["svg_spring"] == ""
        assert sim["graph"]["nodes"] and sim["dot_hierarchical"]          # what the page draws from is still there
        run_id = server.CURRENT_STATE["run_id"]
        server._execute_inference(["by_chance", "mean_difference"])
        server.SIMULATION_RUN_STORE.clear()                               # reload from the archive, as after a restart
        loaded = server._load_archived_run(run_id)
        assert loaded["svg_hierarchical"] == "" and loaded["svg_spring"] == ""
        assert loaded["inferred"]["svgs_dot"] == {}
        assert calls == [], f"{len(calls)} graph render(s) above the limit"
    finally:
        server.CURRENT_STATE.clear()
        server.CURRENT_STATE.update(saved)
