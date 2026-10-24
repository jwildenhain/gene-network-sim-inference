"""Viewing a benchmark dataset must not fail because an old export file cannot be overwritten."""

import json
import os
import stat
import threading
import urllib.request

import pytest

from grn_sim.benchmarks import provenance
from grn_sim.web import server

pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


def test_export_replaces_a_file_this_process_cannot_write(tmp_path):
    first = provenance.export_benchmark_raw_files("dream4_10gene", out_dir=tmp_path)
    expr = first["expression_tsv"]
    expr.write_text("stale")
    expr.chmod(stat.S_IRUSR | stat.S_IRGRP)  # like the root-owned files the Docker container leaves behind

    again = provenance.export_benchmark_raw_files("dream4_10gene", out_dir=tmp_path)
    assert again["expression_tsv"] == expr
    assert expr.read_text().startswith("Gene\tPert_")
    assert not list(tmp_path.glob(".*.tmp")), "no temporary files are left behind"


def test_export_falls_back_to_a_temp_folder_when_the_folder_is_read_only(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        files = provenance.export_benchmark_raw_files("dream4_10gene", out_dir=locked)
    finally:
        locked.chmod(stat.S_IRWXU)
    for path in files.values():
        assert path.exists() and path.parent != locked
    assert json.loads(files["json_bundle"].read_text())["dataset_id"] == "dream4_10gene"


def test_dataset_api_survives_an_unwritable_export_and_downloads_work(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    seeded = provenance.export_benchmark_raw_files("dream4_10gene", out_dir=raw)
    for path in seeded.values():
        path.chmod(stat.S_IRUSR)
    monkeypatch.setattr(provenance, "RAW_DIR", raw)
    saved = server.BENCHMARK_CACHE.pop("bench_studio_dream4_10gene", None)

    httpd = server.ThreadedHTTPServer(("127.0.0.1", 0), server.GRNWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/api/benchmark/dataset?id=dream4_10gene") as resp:
            payload = json.loads(resp.read())
        assert payload["status"] == "success" and payload["raw_files"]["json_bundle"].endswith("_bundle.json")

        with urllib.request.urlopen(f"{base}/api/benchmark/download_raw?id=dream4_10gene&file_type=expression_tsv") as resp:
            assert resp.read().decode().startswith("Gene\tPert_")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.BENCHMARK_CACHE.pop("bench_studio_dream4_10gene", None)
        if saved is not None:
            server.BENCHMARK_CACHE["bench_studio_dream4_10gene"] = saved
        for path in seeded.values():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
