#!/usr/bin/env python3
"""
Check, and if needed fetch, the data files of GRN-Sim. Standard library only, so it runs before `pip install`.

Everything the web studio needs is committed to the repository: a plain `git clone` brings the benchmark
datasets, the MATLAB thesis archives and the Norman/Replogle/DREAM4/SOS exports. The one thing that is not
included is the raw Norman et al. 2019 Perturb-seq matrix (about 160 MB to download, 2.2 GB unpacked). It is
only needed to *rebuild* src/grn_sim/data/benchmarks/norman_2019_real_16tf.json and ..._105gene.json, which are already bundled.

    python scripts/download_data.py                    # check that the bundled data is complete
    python scripts/download_data.py --norman           # also download + unpack the raw Norman 2019 data
    python scripts/download_data.py --norman --build   # ... and rebuild the Norman benchmark JSONs from it
    python scripts/download_data.py --all --dry-run    # show what would happen, change nothing

Source of the raw data: Norman et al., Science 2019 (K562 CRISPRa Perturb-seq), processed h5ad as distributed
with GEARS, Harvard Dataverse doi:10.7910/DVN/Q2ZV3E.
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

NORMAN_URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
NORMAN_SHA256 = "c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877"  # norman.zip, 168,758,985 bytes
NORMAN_ZIP = Path("data/downloads/norman.zip")
NORMAN_H5AD = Path("data/downloads/norman/perturb_processed.h5ad")
NORMAN_NEEDS_GB = 2.6  # zip + unpacked matrix

# (what it is, glob relative to the project root, minimum number of files, required?)
BUNDLED = [
    ("Norman 2019 benchmark (16 TF)", "src/grn_sim/data/benchmarks/norman_2019_real_16tf.json", 1, True),
    ("Norman 2019 benchmark (105 genes)", "src/grn_sim/data/benchmarks/norman_2019_real_105gene.json", 1, True),
    ("MATLAB thesis archives", "src/grn_sim/data/matlab_benchmarks/*.mat", 6, True),
    ("MATLAB overall-statistics archive", "src/grn_sim/data/matlab_benchmarks/overallstats/*.mat", 20, True),
    ("E. coli SOS microarray data", "src/grn_sim/data/SOSdata.txt", 1, True),
    ("Raw benchmark exports", "data/benchmarks/raw/*", 1, False),  # regenerated on demand
]


def check_bundled(root: Path = ROOT) -> List[Tuple[str, bool, int, int, bool]]:
    """Returns (label, ok, files found, files needed, required) for every bundled data group."""
    results = []
    for label, pattern, minimum, required in BUNDLED:
        found = len(list(root.glob(pattern)))
        results.append((label, found >= minimum, found, minimum, required))
    return results


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, opener: Callable = urllib.request.urlopen, quiet: bool = False) -> Path:
    """Streams ``url`` to ``dest`` (through a .part file, so an interrupted download never looks complete)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with opener(url) as response, open(part, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if not quiet:
                pct = f" ({done * 100 // total}%)" if total else ""
                print(f"\r  downloaded {done / 1e6:8.1f} MB{pct}", end="", flush=True)
        if total and done != total:
            part.unlink(missing_ok=True)
            raise IOError(f"incomplete download: got {done} of {total} bytes")
    part.replace(dest)
    if not quiet:
        print()
    return dest


def unpack(zip_path: Path, into: Path) -> None:
    """Extracts a zip, refusing any member that would land outside ``into``."""
    into = into.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (into / member.filename).resolve()
            if into != target and into not in target.parents:
                raise ValueError(f"unsafe path in archive: {member.filename}")
        archive.extractall(into)


def fetch_norman(
    root: Path = ROOT,
    url: str = NORMAN_URL,
    expected_sha256: Optional[str] = NORMAN_SHA256,
    force: bool = False,
    dry_run: bool = False,
    keep_zip: bool = True,
    strict: bool = False,
    opener: Callable = urllib.request.urlopen,
    quiet: bool = False,
) -> bool:
    """Makes sure data/downloads/norman/perturb_processed.h5ad exists. Returns True on success."""
    say = (lambda *a, **k: None) if quiet else print
    h5ad, archive = root / NORMAN_H5AD, root / NORMAN_ZIP

    if h5ad.exists() and not force:
        say(f"[norman] already present: {h5ad.relative_to(root)} ({h5ad.stat().st_size / 1e9:.2f} GB) - nothing to do")
        return True

    free_gb = shutil.disk_usage(root).free / 1e9
    if free_gb < NORMAN_NEEDS_GB and not dry_run:
        say(f"[norman] only {free_gb:.1f} GB free on this disk, about {NORMAN_NEEDS_GB} GB are needed")
        return False

    if dry_run:
        step = "reuse the existing zip" if archive.exists() and not force else f"download {url}"
        say(f"[norman] would {step}, unpack it into {h5ad.parent.relative_to(root)}/ (needs ~{NORMAN_NEEDS_GB} GB)")
        return True

    if archive.exists() and not force:
        say(f"[norman] using the existing {archive.relative_to(root)}")
    else:
        say(f"[norman] downloading {url}")
        try:
            download(url, archive, opener=opener, quiet=quiet)
        except Exception as exc:  # network, HTTP or incomplete download
            say(f"[norman] download failed: {exc}")
            return False

    if expected_sha256:
        actual = sha256_of(archive)
        if actual != expected_sha256:
            say(f"[norman] checksum differs from the file this project was built with\n"
                f"         expected {expected_sha256}\n         got      {actual}")
            if strict:
                return False
            say("[norman] continuing (the dataset may have been re-published); use --strict to stop instead")
        else:
            say("[norman] checksum OK")

    say(f"[norman] unpacking into {h5ad.parent.parent.relative_to(root)}/")
    try:
        unpack(archive, h5ad.parent.parent)
    except (zipfile.BadZipFile, ValueError) as exc:
        say(f"[norman] could not unpack: {exc}")
        return False
    if not h5ad.exists():
        say(f"[norman] unpacked, but {h5ad.relative_to(root)} is missing from the archive")
        return False
    if not keep_zip:
        archive.unlink()
    say(f"[norman] ready: {h5ad.relative_to(root)}")
    return True


def build_norman(root: Path = ROOT, dry_run: bool = False, will_download: bool = False) -> bool:
    """Rebuilds the bundled norman_2019_real_*.json benchmarks from the raw matrix (needs the package installed).

    ``will_download``: the raw matrix is fetched by the same invocation, so in a dry run its absence is expected."""
    if not (root / NORMAN_H5AD).exists():
        if dry_run and will_download:
            print("[build] would rebuild the Norman benchmark JSONs once the raw matrix has been downloaded")
            return True
        print("[build] the raw Norman matrix is missing - run with --norman first")
        return False
    if dry_run:
        print("[build] would run grn_sim.benchmarks.build_real_norman.build_real_norman_datasets()")
        return True
    try:
        sys.path.insert(0, str(root / "src"))
        from grn_sim.benchmarks.build_real_norman import build_real_norman_datasets
    except ImportError as exc:
        print(f"[build] cannot import the project ({exc}); run `pip install -e .` first")
        return False
    build_real_norman_datasets()
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the top of this file for details.")
    parser.add_argument("--norman", action="store_true", help="download + unpack the raw Norman 2019 Perturb-seq data (~160 MB)")
    parser.add_argument("--build", action="store_true", help="rebuild the Norman benchmark JSONs from the raw data")
    parser.add_argument("--all", action="store_true", help="--norman and --build")
    parser.add_argument("--force", action="store_true", help="download / unpack again even if the files exist")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be done")
    parser.add_argument("--strict", action="store_true", help="treat a checksum mismatch as an error")
    parser.add_argument("--clean", action="store_true", help="delete the zip after unpacking")
    parser.add_argument("--url", default=NORMAN_URL, help="alternative location of norman.zip (mirror)")
    args = parser.parse_args(argv)
    want_norman, want_build = args.norman or args.all, args.build or args.all

    print(f"Project: {ROOT}")
    all_ok = True
    for label, ok, found, needed, required in check_bundled():
        status = "ok" if ok else ("MISSING" if required else "absent (created on demand)")
        print(f"  [{status:>9}] {label}: {found} file(s)" + ("" if ok else f", expected at least {needed}"))
        all_ok &= ok or not required
    if not all_ok:
        print("Some bundled data is missing: re-clone the repository (`git clone ...`) or run `git checkout -- data`.")

    ok = True
    if want_norman:
        ok &= fetch_norman(url=args.url, force=args.force, dry_run=args.dry_run, strict=args.strict, keep_zip=not args.clean)
    if want_build and ok:
        ok &= build_norman(dry_run=args.dry_run, will_download=want_norman)
    if not (want_norman or want_build):
        print("The bundled data is all the web studio needs. Use --norman only to rebuild the Norman benchmarks.")
    return 0 if (all_ok and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
