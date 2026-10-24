"""Shared fixtures for the browser tests: the real web handler on a free port, and headless Chromium."""

import threading

import pytest


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
