"""
Server side of the Virtual Cell tab's results panel: loads the stored screen-benchmark simulations
(``benchmarks/vcc_screen_benchmark.jsonl``, written by ``scripts/run_vcc_benchmark.py``) and answers filtered, aggregated
queries (see :mod:`grn_sim.evaluation.vcc_benchmark` for the statistics).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.paths import resolve_data_file, user_data_dir

FILE_PARTS = ("benchmarks", "vcc_screen_benchmark.jsonl")
USER_FILE = "vcc_screen_user.jsonl"           # simulations run from the studio, kept in the user data folder
MAX_LIVE_GENES = 200
_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {"key": None, "rows": []}


def _user_path() -> Path:
    return user_data_dir() / USER_FILE


def load_rows() -> List[Dict[str, Any]]:
    """The bundled benchmark rows plus the user's own simulations, cached on the files' (path, mtime, size)."""
    paths = [resolve_data_file(*FILE_PARTS), _user_path()]
    stamps = []
    for path in paths:
        try:
            st = path.stat()
            stamps.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            stamps.append((str(path), 0, 0))
    key = tuple(stamps)
    with _LOCK:
        if _CACHE["key"] != key:
            rows = []
            for path in paths:
                if not path.is_file():
                    continue
                for line in path.read_text().splitlines():
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except ValueError:
                            continue
            _CACHE.update(key=key, rows=rows)
        return _CACHE["rows"]


def parse_simulation(query: Dict[str, List[str]]) -> Tuple[vb.Setting, int]:
    """The setting and seed of a live simulation request, validated (raises ``ValueError``)."""
    def one(name: str, default: str) -> str:
        return str(query.get(name, [default])[0]).strip() or default

    def integer(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(one(name, str(default)))
        except ValueError:
            raise ValueError(f"{name} must be an integer") from None
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")
        return value

    pathway, mode, selection = one("pathway", "replogle_like"), one("mode", "knockdown"), one("selection", "full")
    if pathway not in vb.PATHWAYS:
        raise ValueError(f"pathway must be one of {', '.join(vb.PATHWAYS)}")
    if mode not in vb.MODES:
        raise ValueError(f"mode must be one of {', '.join(vb.MODES)}")
    if selection not in vb.SELECTIONS:
        raise ValueError(f"selection must be one of {', '.join(vb.SELECTIONS)}")
    n = integer("n_genes", 100, 12, MAX_LIVE_GENES)
    pairs = integer("pairs", 0, 0, 2 * n)
    seed = integer("seed", int(time.time()) % 100000 + 1000, 1, 10 ** 9)
    return vb.Setting(pathway=pathway, n_genes=n, mode=mode, pairs=pairs, selection=selection), seed


def simulate(query: Dict[str, List[str]], save: bool = True) -> Dict[str, Any]:
    """
    Runs one new simulation (network, screen, every algorithm) and, with ``save``, adds it to the user's rows so it joins the
    statistics. Raises ``RuntimeError`` when another simulation is already running.
    """
    setting, seed = parse_simulation(query)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("another screen simulation is running; wait for it to finish")
    try:
        t = time.time()
        rows = vb.run_simulation(setting, seed)
        if save:
            path = _user_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    finally:
        _RUN_LOCK.release()
    return {"setting": setting.to_dict(), "seed": seed, "rows": rows, "seconds": round(time.time() - t, 1),
            "saved": bool(save), "aggregate": vb.aggregate(rows, {})}


def parse_filters(query: Dict[str, List[str]]) -> Tuple[Dict[str, str], str]:
    """``(filters, compare_by)`` from a parsed query string; every factor accepts ``all``."""
    filters: Dict[str, str] = {}
    for f in vb.FACTORS:
        v = str(query.get(f, ["all"])[0]).strip()
        filters[f] = v or "all"
    compare = str(query.get("compare_by", [""])[0]).strip()
    if compare and compare not in vb.FACTORS:
        raise ValueError(f"compare_by must be one of {', '.join(vb.FACTORS)}, got {compare!r}")
    return filters, compare


def answer(query: Dict[str, List[str]]) -> Dict[str, Any]:
    filters, compare = parse_filters(query)
    rows = load_rows()
    out: Dict[str, Any] = {
        "available": bool(rows),
        "levels": vb.factor_levels(rows),
        "labels": {"pathway": vb.PATHWAYS, "mode": vb.MODES, "selection": vb.SELECTIONS},
        "filters": filters,
        "aggregate": vb.aggregate(rows, filters),
        "n_rows": len(rows),
        "algorithms": {a.name: {"label": a.label, "family": a.family, "uses_pairs": a.uses_pairs}
                       for a in vb.algorithm_library()},
        "decision_rule": {"min_lift": vb.MIN_LIFT, "alpha": vb.ALPHA, "test": "paired one-sided Wilcoxon signed-rank vs the "
                          "random ranking of the same simulation, Benjamini-Hochberg across algorithms"},
    }
    if compare:
        pivot: Dict[str, Any] = {}
        for level in out["levels"][compare]:
            sub = vb.aggregate(rows, {**filters, compare: str(level)})
            if sub["n_simulations"]:
                pivot[str(level)] = {"n_simulations": sub["n_simulations"],
                                     "algorithms": {a["algorithm"]: {k: a[k] for k in ("mean", "sd", "lift", "verdict", "n", "win_rate", "q_value", "q_hub")}
                                                    for a in sub["algorithms"]}}
        out["pivot"] = {"factor": compare, "levels": pivot}
    return out
