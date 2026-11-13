"""HTTP-level tests for the web server: cross-origin guard, request parsing, shared-state locking, CLI."""

import http.client
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from grn_sim.web import database, server


@pytest.fixture(scope="module")
def base_url():
    httpd = server.ThreadedHTTPServer(("127.0.0.1", 0), server.GRNWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def request(url, method="GET", headers=None, body=None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def post_json(base_url, path, payload, **headers):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return request(base_url + path, "POST", {"Content-Type": "application/json", **headers}, body)


# --- cross-origin guard / CORS -------------------------------------------------------------------

def test_requests_from_other_origins_are_rejected(base_url):
    evil = {"Origin": "http://evil.example"}
    assert request(base_url + "/api/status", headers=evil)[0] == 403
    code, _, body = post_json(base_url, "/api/history/load", {"run_id": "x"}, **evil)
    assert code == 403 and "Cross-origin" in json.loads(body)["error"]


def test_same_origin_and_non_browser_requests_are_served(base_url):
    assert request(base_url + "/api/status", headers={"Origin": base_url})[0] == 200
    assert request(base_url + "/api/status")[0] == 200  # curl / scripts send no Origin
    assert post_json(base_url, "/api/history/load", {"run_id": "missing"}, Origin=base_url)[0] == 404


def test_cross_origin_post_has_no_side_effects(base_url):
    before = dict(server.CURRENT_STATE)
    post_json(base_url, "/api/simulate", {"num_genes": 5, "seed": 1}, Origin="http://evil.example")
    assert server.CURRENT_STATE == before


def test_no_cors_headers_by_default(base_url):
    _, headers, _ = request(base_url + "/api/status")
    assert "Access-Control-Allow-Origin" not in headers


def test_configured_cors_origin_is_allowed_and_echoed(base_url, monkeypatch):
    monkeypatch.setenv("GRN_CORS_ORIGIN", "https://studio.example")
    code, headers, _ = request(base_url + "/api/status", headers={"Origin": "https://studio.example"})
    assert code == 200 and headers["Access-Control-Allow-Origin"] == "https://studio.example"
    assert request(base_url + "/api/status", headers={"Origin": "https://other.example"})[0] == 403


# --- request parsing -----------------------------------------------------------------------------

@pytest.mark.parametrize("body", [b"{not json", b"[1, 2, 3]", b'"text"', b"\xff\xfe"])
def test_malformed_post_bodies_get_400(base_url, body):
    code, _, payload = post_json(base_url, "/api/history/load", body)
    assert code == 400 and "Invalid request body" in json.loads(payload)["error"]


def test_bad_content_length_gets_400(base_url):
    host, port = base_url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.putrequest("POST", "/api/history/load")
    conn.putheader("Content-Length", "abc")
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 400
    conn.close()


@pytest.mark.parametrize("cells", ["0", "-5", "999999", "abc"])
@pytest.mark.parametrize("path", ["/api/export/perturb_seq_h5ad?", "/api/download?format=h5ad&"])
def test_h5ad_cell_count_is_bounded(base_url, path, cells):
    code, _, body = request(f"{base_url}{path}cells_per_condition={cells}")
    assert code == 400 and "error" in json.loads(body)


def test_vcc_endpoint_rejects_bad_parameters_and_never_emits_nan(base_url, monkeypatch):
    for qs in ("rewire_fraction=2", "rewire_fraction=nan", "num_genes=2", "num_genes=abc", "k_av=0"):
        code, _, body = request(f"{base_url}/api/benchmark/vcc_atlasshift?{qs}")
        assert code == 400 and json.loads(body)["status"] == "error", qs

    monkeypatch.setattr(server, "run_vcc_comparative_benchmark", lambda **kw: {"status": "success", "bad": float("nan")})
    saved = dict(server.BENCHMARK_CACHE)
    try:
        code, _, body = request(f"{base_url}/api/benchmark/vcc_atlasshift?seed=7&refresh=1")
        assert code == 500 and "error" in json.loads(body)  # parseable JSON, not a bare NaN
    finally:
        server.BENCHMARK_CACHE.clear()
        server.BENCHMARK_CACHE.update(saved)


# --- shared state --------------------------------------------------------------------------------

def test_state_mutators_hold_the_state_lock():
    for fn in (server._execute_simulation, server._execute_inference, server._load_archived_run):
        assert hasattr(fn, "__wrapped__"), f"{fn.__name__} must be wrapped by @_state_locked"


def test_state_locked_serialises_callers():
    active, peak = [0], [0]

    @server._state_locked
    def critical():
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.03)
        active[0] -= 1

    threads = [threading.Thread(target=critical) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1


def test_concurrent_identical_vcc_requests_compute_once(monkeypatch):
    runs = []

    def slow_benchmark(**kw):
        runs.append(kw)
        time.sleep(0.2)
        return {"status": "success"}

    monkeypatch.setattr(server, "run_vcc_comparative_benchmark", slow_benchmark)
    saved = dict(server.BENCHMARK_CACHE)
    server.BENCHMARK_CACHE.clear()
    try:
        threads = [threading.Thread(target=server._build_vcc_benchmark_payload, args=(8, 1.5, 3)) for _ in range(5)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(runs) == 1
    finally:
        server.BENCHMARK_CACHE.clear()
        server.BENCHMARK_CACHE.update(saved)


# --- command line --------------------------------------------------------------------------------

def test_cli_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    cli = server._parse_cli([])
    assert (cli.host, cli.port) == ("127.0.0.1", 8080)


def test_cli_accepts_positional_port_flags_and_environment(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert server._parse_cli(["8085"]).port == 8085  # Dockerfile: python -m grn_sim.web.server 8080
    cli = server._parse_cli(["--host", "0.0.0.0", "--port", "9000"])  # README
    assert (cli.host, cli.port) == ("0.0.0.0", 9000)
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8123")
    cli = server._parse_cli([])
    assert (cli.host, cli.port) == ("0.0.0.0", 8123)


# --- database ------------------------------------------------------------------------------------

def test_stored_zero_f1_is_not_recomputed(tmp_path):
    db = str(tmp_path / "t.sqlite")
    base = {"best_alg_key": "vote", "best_power_pct": 80, "metrics_summary": {"vote": {"ppv": 50}}}
    for run_id, entry in (("zero", {**base, "run_id": "zero", "f1_score": 0.0}),
                          ("missing", {**base, "run_id": "missing"})):
        database.save_simulation_to_db(run_id, {"num_genes": 5}, history_entry=entry, db_path=db)

    entries = {e["run_id"]: e for e in database.load_all_history_from_db(db)}
    assert entries["zero"]["f1_score"] == 0.0
    assert entries["missing"]["f1_score"] == pytest.approx(2 * 0.8 * 0.5 / (0.8 + 0.5), abs=1e-4)


# --- clients that disconnect mid-response ------------------------------------------------------------

@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
def test_tolerant_writer_swallows_client_disconnects(error):
    class Raw:
        def __init__(self):
            self.calls = 0

        def write(self, data):
            self.calls += 1
            raise error()

        def flush(self):
            raise error()

    raw = Raw()
    writer = server._TolerantWriter(raw)
    assert writer.write(b"abc") == 3 and writer.client_gone
    assert writer.write(b"more") == 4 and raw.calls == 1, "nothing more is sent to a closed connection"
    writer.flush()  # no exception


def test_tolerant_writer_passes_normal_writes_through():
    import io
    raw = io.BytesIO()
    writer = server._TolerantWriter(raw)
    assert writer.write(b"hello") == 5
    writer.flush()
    assert raw.getvalue() == b"hello" and not writer.client_gone and writer.getvalue() == b"hello"


def test_aborted_downloads_do_not_log_tracebacks(base_url, capfd):
    import socket
    import struct
    host, port = base_url.removeprefix("http://").split(":")
    for _ in range(5):
        s = socket.create_connection((host, int(port)))
        s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close with a reset
        s.close()
    time.sleep(0.5)
    err = capfd.readouterr().err
    assert "Traceback" not in err and "BrokenPipe" not in err
    assert request(base_url + "/api/status")[0] == 200, "the server keeps serving"


@pytest.mark.parametrize("path", ["/api/export/perturb_seq_h5ad?", "/api/download?format=h5ad&"])
def test_h5ad_downloads_are_zipped_unless_zip_is_switched_off(base_url, path):
    import io
    import zipfile

    code, headers, body = request(f"{base_url}{path}cells_per_condition=2")
    assert code == 200 and headers["Content-Type"] == "application/zip"
    name = headers["Content-Disposition"].split('filename="')[1].rstrip('"')
    assert name.endswith(".h5ad.zip")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        assert z.namelist() == [name[:-len(".zip")]]
        assert z.testzip() is None
        inner = z.read(z.namelist()[0])
    assert inner[:8] == b"\x89HDF\r\n\x1a\n"                       # the file inside is the HDF5 / AnnData file itself

    code, headers, raw = request(f"{base_url}{path}cells_per_condition=2&zip=0")
    assert code == 200 and headers["Content-Disposition"].endswith('.h5ad"') and raw[:8] == b"\x89HDF\r\n\x1a\n"


def test_a_download_that_does_not_exist_for_the_run_is_a_404_with_a_reason_not_an_empty_file(base_url, monkeypatch):
    monkeypatch.setitem(server.CURRENT_STATE, "svg_hierarchical", "")
    monkeypatch.setitem(server.CURRENT_STATE, "dot_hierarchical", "")
    for fmt in ("svg", "dot"):
        code, _, body = request(f"{base_url}/api/download?format={fmt}")
        assert code == 404 and f".{fmt}" in json.loads(body)["error"]


def test_downloads_of_other_runs_are_not_answered_with_the_current_run(base_url, monkeypatch):
    """A run id that is not in memory used to get the studio's current run's file; archived runs are loaded, unknown ids are 404."""
    sid = "iorn-dl-test"
    sim = {"run_id": sid, "num_genes": 3, "gene_names": ["G1", "G2", "G3"], "true_matrix": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
           "msmt_text": "ARCHIVED-MSMT", "pert_text": "ARCHIVED-PERT", "svg_hierarchical": "<svg/>", "dot_hierarchical": "digraph{}",
           "X": [[0.0] * 3] * 3, "U": [[0.0] * 3] * 3}
    database.save_simulation_to_db(run_id=sid, sim_payload=sim, inferred_payload={"metrics": {}}, history_entry={"run_id": sid})
    monkeypatch.setitem(server.CURRENT_STATE, "msmt_text", "CURRENT-MSMT")
    monkeypatch.setitem(server.CURRENT_STATE, "run_id", "current-run")
    server.SIMULATION_RUN_STORE.pop(sid, None)
    code, _, body = request(f"{base_url}/api/download?format=msmt&run_id={sid}")
    assert code == 200 and body == b"ARCHIVED-MSMT"
    assert server.CURRENT_STATE["run_id"] == "current-run" and server.CURRENT_STATE["msmt_text"] == "CURRENT-MSMT"
    assert request(f"{base_url}/api/download?format=msmt")[2] == b"CURRENT-MSMT"        # no id: the current run, as before
    for fmt in ("msmt", "pert", "svg", "dot", "h5ad"):
        code, _, body = request(f"{base_url}/api/download?format={fmt}&run_id=iorn-does-not-exist")
        assert code == 404 and "unknown run" in json.loads(body)["error"], fmt
    code, _, body = request(f"{base_url}/api/export/perturb_seq_h5ad?run_id=iorn-does-not-exist")
    assert code == 404
