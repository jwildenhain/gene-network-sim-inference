"""
Where GRN-Sim finds its read-only data and where it keeps files it writes.

Two kinds of files, kept apart so the package works the same from a source checkout, a ``pip install`` and Docker:

* **Bundled data** (read-only, shipped inside the package): the MATLAB thesis archives, the E. coli SOS microarray
  data, the Norman 2019 benchmark JSONs and the precomputed overall-statistics sweep. See :func:`bundled_path`.
* **User data** (writable): the simulation archive (SQLite), raw downloads, regenerated benchmark exports. See
  :func:`user_data_dir`.

User data location, first match wins:

1. ``$GRN_SIM_HOME``
2. ``<checkout>/data`` when running from a source checkout (``pyproject.toml`` next to ``src/``), which keeps the
   layout this project has always used
3. the platform's per-user data folder (``~/.local/share/grn-sim``, ``~/Library/Application Support/grn-sim``,
   ``%LOCALAPPDATA%\\grn-sim``)

The simulation archive can additionally be pointed elsewhere with ``$RGN_DB_PATH``.
"""

import os
import sys
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_DATA_DIR = PACKAGE_DIR / "data"

ENV_HOME = "GRN_SIM_HOME"
ENV_DB_PATH = "RGN_DB_PATH"


def source_checkout_root() -> Optional[Path]:
    """The repository root when the package is imported from ``<root>/src/grn_sim``, otherwise ``None``."""
    root = PACKAGE_DIR.parent.parent
    if (root / "pyproject.toml").is_file() and (root / "src" / "grn_sim").is_dir():
        return root
    return None


def _platform_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "grn-sim"


def user_data_dir() -> Path:
    """Folder for everything the package writes (not created here; callers create it when they write)."""
    configured = os.environ.get(ENV_HOME)
    if configured:
        return Path(configured).expanduser()
    root = source_checkout_root()
    return root / "data" if root is not None else _platform_data_dir()


def archive_db_path() -> Path:
    """The SQLite simulation archive (``$RGN_DB_PATH`` or ``<user data>/rgn_db.sqlite``)."""
    configured = os.environ.get(ENV_DB_PATH)
    return Path(configured).expanduser() if configured else user_data_dir() / "rgn_db.sqlite"


def downloads_dir() -> Path:
    """Raw third-party downloads (e.g. the Norman 2019 Perturb-seq matrix, about 2 GB unpacked)."""
    return user_data_dir() / "downloads"


def exports_dir() -> Path:
    """Where the standalone benchmark exports (TSV / JSON bundles) are regenerated."""
    return user_data_dir() / "benchmarks" / "raw"


def bundled_path(*parts: str) -> Path:
    """A file shipped with the package, e.g. ``bundled_path("matlab_benchmarks", "x.mat")``."""
    return BUNDLED_DATA_DIR.joinpath(*parts)


def resolve_data_file(*parts: str) -> Path:
    """
    A data file for reading: the user's copy under :func:`user_data_dir` when present (a rebuilt or updated
    file overrides the shipped one), otherwise the bundled copy. The bundled path is returned when neither
    exists, so error messages name the expected location.
    """
    user_copy = user_data_dir().joinpath(*parts)
    return user_copy if user_copy.is_file() else bundled_path(*parts)


def resolve_data_dir(*parts: str) -> Path:
    """Like :func:`resolve_data_file` for a folder: the user's folder when it exists, otherwise the bundled one."""
    user_copy = user_data_dir().joinpath(*parts)
    return user_copy if user_copy.is_dir() else bundled_path(*parts)
