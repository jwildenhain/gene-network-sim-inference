"""Large payload columns are stored compressed, read back transparently; legacy duplicate columns are skipped for big networks."""
import json
import sqlite3

import numpy as np

from grn_sim.web import database as db


def test_pack_round_trip_and_threshold():
    small = json.dumps({"a": 1})
    assert db.pack_json(small) == small and db.unpack_json(small) == small
    big = json.dumps({"x": list(np.random.default_rng(0).random(200_000))})
    packed = db.pack_json(big)
    assert isinstance(packed, bytes) and len(packed) < len(big)
    assert db.unpack_json(packed) == big
    assert db.unpack_json(sqlite3.Binary(packed)) == big


def _payload(genes, extra=0):
    return {"num_genes": genes, "topology": "modular", "seed": 1, "svg_hierarchical": "<svg/>", "pert_text": "p\tq",
            "msmt_text": "m\tn", "true_matrix": [[0, 1], [0, 0]], "X": [[0.5] * 4] * 3,
            "padding": "x" * extra}


def test_big_run_is_compressed_in_the_file_and_loads_unchanged(tmp_path):
    path = str(tmp_path / "t.sqlite")
    inferred = {"metrics": {"s2g": {"f1_score": 0.5}}, "tmp_statmat": {"s2g": {"specificity": 99.0, "power": 40.0}},
                "dots": {"s2g": "digraph {}"}}
    db.save_simulation_to_db("run-big", _payload(300, extra=3_000_000), inferred, {"timestamp": "t"}, db_path=path)
    db.save_simulation_to_db("run-small", _payload(12), inferred, {"timestamp": "t"}, db_path=path)
    con = sqlite3.connect(path)
    kinds = dict(con.execute("select simid, typeof(sim_payload_json) from nw"))
    assert kinds == {"run-big": "blob", "run-small": "text"}
    big_size = con.execute("select length(sim_payload_json) from nw where simid='run-big'").fetchone()[0]
    assert big_size < 100_000  # 3 MB of repeated text compresses to a few kB
    # legacy duplicate columns: empty above the gene limit, still filled for small networks
    assert con.execute("select mpet, mmsmt, mdot, mat from nw where simid='run-big'").fetchone() == ("", "", "", "")
    assert con.execute("select mpet, mmsmt from nw where simid='run-small'").fetchone() == ("p\tq", "m\tn")
    assert con.execute("select dmf from alg where simid='run-big'").fetchone()[0] == ""
    assert con.execute("select dmf from alg where simid='run-small'").fetchone()[0] == "digraph {}"
    loaded = db.load_run_from_db("run-big", db_path=path)
    assert loaded["sim_payload"]["padding"] == "x" * 3_000_000 and loaded["inferred_payload"] == inferred
    assert db.load_run_from_db("run-small", db_path=path)["sim_payload"]["num_genes"] == 12
