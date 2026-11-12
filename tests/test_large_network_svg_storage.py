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
