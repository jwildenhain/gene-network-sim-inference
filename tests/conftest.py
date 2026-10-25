"""Shared fixtures for the browser tests: the real web handler on a free port, and headless Chromium."""

import atexit
import os
import shutil
import tempfile
import threading

import pytest

# In-process tests must never write to the real simulation archive (data/rgn_db.sqlite): every simulation the web
# handler runs is archived, so a test session used to add dozens of runs to it. Set before any test module is
# imported (the server may archive a default run at start-up); export RGN_DB_PATH yourself to choose another file.
if "RGN_DB_PATH" not in os.environ:
    _archive_dir = tempfile.mkdtemp(prefix="grn-sim-test-archive-")
    os.environ["RGN_DB_PATH"] = os.path.join(_archive_dir, "tests.sqlite")
    atexit.register(shutil.rmtree, _archive_dir, True)


@pytest.fixture(scope="module")
def app_url():
    from grn_sim.web import server

    httpd = server.ThreadedHTTPServer(("127.0.0.1", 0), server.GRNWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as exc:  # browsers not installed
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()
