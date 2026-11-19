"""Check that every page rendered from stored data is current (run it before pushing).

A regenerated corpus (the Tab 4 screen benchmark, the archive snapshot, the scale snapshot) silently makes the pages
derived from it stale; each page has its own test, but those are easy to miss when you select tests by name. This runs
every renderer in ``--check`` mode and exits 1 when any page differs from a fresh render.

    python scripts/check_derived_docs.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CHECKS = (
    ("render_top5_results.py", "--check"),
    ("render_audit_page.py", "--check"),
    ("render_archive_leaderboard.py", "render", "--check"),
    ("render_scale_benchmark.py", "render", "--check"),
    ("generate_algorithm_docs.py", "--check"),
)
TESTS = ("tests/test_statgen_report.py", "tests/test_vcc_findings.py")


def main() -> int:
    failed = []
    for name, *args in CHECKS:
        r = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True)
        print(f"{'ok  ' if r.returncode == 0 else 'STALE'} {name} {' '.join(args)}")
        if r.returncode:
            failed.append(name)
    tests = [t for t in TESTS if (ROOT / t).exists()]
    if tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=ROOT, capture_output=True, text=True)
        print(f"{'ok  ' if r.returncode == 0 else 'STALE'} {' '.join(tests)}")
        if r.returncode:
            failed.append("tests")
    if failed:
        print("\nout of date: " + ", ".join(failed) + " (re-render with the script named, then commit)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
