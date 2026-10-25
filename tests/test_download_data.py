"""scripts/download_data.py: check the bundled data and fetch the optional raw Norman 2019 data (offline tests)."""

import hashlib
import http.server
import importlib.util
import io
import threading
import urllib.error
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
spec = importlib.util.spec_from_file_location("download_data", SCRIPT)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)


def make_zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


GOOD_ZIP = make_zip({"norman/": b"", "norman/perturb_processed.h5ad": b"H5AD" * 1000, "norman/go.csv": b"a,b\n"})
GOOD_SHA = hashlib.sha256(GOOD_ZIP).hexdigest()


@pytest.fixture
def server():
    """Serves GOOD_ZIP (or whatever `payload` is set to) on localhost."""
    state = {"payload": GOOD_ZIP, "hits": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["payload"])))
            self.end_headers()
            self.wfile.write(state["payload"])

        def log_message(self, *args):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/norman.zip"
    yield state
    httpd.shutdown()
    httpd.server_close()


# --- the bundled data ------------------------------------------------------------------------------------

def test_this_repository_has_all_required_bundled_data():
    results = dd.check_bundled(dd.ROOT)
    missing = [label for label, ok, _, _, required in results if required and not ok]
    assert missing == []


def test_an_empty_folder_reports_everything_missing(tmp_path):
    results = dd.check_bundled(tmp_path)
    assert results and not any(ok for _, ok, _, _, _ in results)


def test_main_check_succeeds_and_changes_nothing(capsys):
    assert dd.main([]) == 0
    out = capsys.readouterr().out
    assert "Norman 2019 benchmark (16 TF)" in out and "MISSING" not in out and "bundled data is all the web studio needs" in out


def test_main_reports_missing_data_with_a_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(dd, "check_bundled", lambda root=dd.ROOT: [("Norman 2019 benchmark (16 TF)", False, 0, 1, True)])
    assert dd.main([]) == 1
    out = capsys.readouterr().out
    assert "MISSING" in out and "git clone" in out


# --- downloading the raw Norman data ---------------------------------------------------------------------

def test_download_unpack_and_checksum(tmp_path, server):
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, quiet=True)
    h5ad = tmp_path / dd.NORMAN_H5AD
    assert h5ad.read_bytes() == b"H5AD" * 1000
    assert (tmp_path / "data/downloads/norman/go.csv").exists()
    assert (tmp_path / dd.NORMAN_ZIP).exists(), "the zip is kept unless --clean"
    assert not list(tmp_path.rglob("*.part")), "no partial files are left behind"


def test_second_run_downloads_nothing(tmp_path, server):
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, quiet=True)
    hits = server["hits"]
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, quiet=True)
    assert server["hits"] == hits


def test_force_downloads_again_and_clean_removes_the_zip(tmp_path, server):
    dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, quiet=True)
    before = server["hits"]
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, force=True, keep_zip=False, quiet=True)
    assert server["hits"] == before + 1, "--force fetches the archive again"
    assert not (tmp_path / dd.NORMAN_ZIP).exists(), "--clean deletes the zip after unpacking"
    assert (tmp_path / dd.NORMAN_H5AD).exists()


def test_an_existing_zip_is_reused_without_downloading(tmp_path, server):
    zip_path = tmp_path / dd.NORMAN_ZIP
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(GOOD_ZIP)
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=GOOD_SHA, quiet=True)
    assert server["hits"] == 0 and (tmp_path / dd.NORMAN_H5AD).exists()


def test_checksum_mismatch_warns_by_default_and_fails_when_strict(tmp_path, server, capsys):
    assert dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256="0" * 64)
    assert "checksum differs" in capsys.readouterr().out

    other = tmp_path / "strict"
    other.mkdir()
    assert not dd.fetch_norman(root=other, url=server["url"], expected_sha256="0" * 64, strict=True, quiet=True)


def test_dry_run_changes_nothing(tmp_path, server, capsys):
    assert dd.fetch_norman(root=tmp_path, url=server["url"], dry_run=True)
    assert server["hits"] == 0 and not (tmp_path / "data").exists()
    assert "would download" in capsys.readouterr().out


def test_failed_download_leaves_no_files(tmp_path):
    def broken(url):
        raise urllib.error.URLError("no network")

    assert not dd.fetch_norman(root=tmp_path, url="http://x/norman.zip", opener=broken, quiet=True)
    assert not list(tmp_path.rglob("*.part")) and not (tmp_path / dd.NORMAN_H5AD).exists()


def test_unsafe_archive_is_rejected(tmp_path, server):
    evil = make_zip({"norman/perturb_processed.h5ad": b"x", "../escape.txt": b"nope"})
    server["payload"] = evil
    assert not dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=None, quiet=True)
    assert not (tmp_path / "data" / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()


def test_archive_without_the_matrix_is_reported(tmp_path, server):
    server["payload"] = make_zip({"norman/go.csv": b"a,b"})
    assert not dd.fetch_norman(root=tmp_path, url=server["url"], expected_sha256=None, quiet=True)


def test_not_enough_disk_space_is_reported_before_downloading(tmp_path, server, monkeypatch):
    monkeypatch.setattr(dd.shutil, "disk_usage", lambda p: type("U", (), {"free": int(0.5e9)})())
    assert not dd.fetch_norman(root=tmp_path, url=server["url"], quiet=True)
    assert server["hits"] == 0


def test_build_needs_the_raw_matrix_first(tmp_path, capsys):
    assert not dd.build_norman(root=tmp_path)
    assert "--norman first" in capsys.readouterr().out


def test_dry_run_of_a_build_that_follows_a_download_does_not_need_the_raw_matrix_yet(tmp_path, capsys):
    """`--all --dry-run` on a fresh checkout: the matrix is missing only because the dry run did not download it."""
    assert dd.build_norman(root=tmp_path, dry_run=True, will_download=True)
    assert "would rebuild" in capsys.readouterr().out
    assert not dd.build_norman(root=tmp_path, dry_run=True), "a build without a download still needs the raw matrix"
    assert "--norman first" in capsys.readouterr().out


def test_dry_run_of_everything_is_safe_on_this_machine(capsys):
    assert dd.main(["--all", "--dry-run"]) == 0
    assert "norman" in capsys.readouterr().out.lower()


def test_expected_checksum_matches_the_zip_this_project_was_built_with():
    local = dd.ROOT.parent / "data" / "downloads" / "norman.zip"
    if not local.exists():
        pytest.skip("the original norman.zip is not on this machine")
    assert dd.sha256_of(local) == dd.NORMAN_SHA256
