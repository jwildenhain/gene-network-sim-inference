"""The statistics behind tab 5 (Overall Statistics), computed from the archive and kept in it.

The page used to read a JSON file written once by ``parameter_sweep.run_systematic_sweep`` (69 networks, the algorithms that
existed that day). Since then algorithms were added and every archived run was completed, so the numbers are now rebuilt from
the archive: the 69 networks of the sweep design (``parameter_sweep.SIZE_VALUES`` ... , run ids as that sweep names them), each
with every registered algorithm, read from the stored metrics. Nothing is simulated or fitted here.

The result has the same structure the sweep produced (the page is unchanged) and is stored in the archive's ``overall_stats``
table, one row per snapshot name, so the page reads it with one key lookup and a stale snapshot is visible by its timestamp.

    python -m grn_sim.benchmarks.overall_stats [--db path] [--no-store]
"""
import datetime
import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grn_sim.benchmarks.parameter_sweep import K_VALUES, NOISE_VALUES, SIZE_VALUES, TOPOLOGY_DEFS
from grn_sim.inference import registry
from grn_sim.web.database import get_db_path, init_db, unpack_json

SNAPSHOT = "sweep"
_ABBREV = {"random": "ran", "power_law": "pow", "hierarchical": "hie"}
_TABLE = """CREATE TABLE IF NOT EXISTS overall_stats (
    name TEXT PRIMARY KEY,
    built_at TEXT NOT NULL,
    n_runs INTEGER NOT NULL,
    n_algorithms INTEGER NOT NULL,
    payload_json TEXT NOT NULL)"""


def design() -> List[Dict[str, Any]]:
    """The sweep's networks: run id, axis ('size', 'conn', 'noise', 'dyn'), topology and the axis value."""
    out: List[Dict[str, Any]] = []

    def add(axis, topo, value, run_id):
        out.append({"run_id": run_id, "axis": axis, "topology": topo, "value": value})

    for n in SIZE_VALUES:
        for t, (name, _, _) in enumerate(TOPOLOGY_DEFS):
            add("size", name, n, f"sweep-size-{name}-{_ABBREV[name]}-n{n}-k18-s{100 + n * 10 + t}")
    for k in K_VALUES:
        for t, (name, _, _) in enumerate(TOPOLOGY_DEFS):
            add("conn", name, k, f"sweep-conn-{name}-{_ABBREV[name]}-n16-k{int(k * 10)}-s{200 + int(k * 20) + t}")
    for z in NOISE_VALUES:
        for t, (name, _, _) in enumerate(TOPOLOGY_DEFS):
            add("noise", name, z, f"sweep-noise-{name}-{_ABBREV[name]}-n16-k18-s{300 + int(z * 100) + t}")
    for t, (name, _, _) in enumerate(TOPOLOGY_DEFS):
        a = _ABBREV[name]
        add("dyn", name, "responsive", f"sweep-dyn-resp-{name}-{a}-n12-k18-s{400 + t}")
        add("dyn", name, "flat", f"sweep-dyn-flat-{name}-{a}-n12-k14-s{500 + t}")
        add("dyn", name, "oscillating", f"sweep-dyn-osc-{name}-{a}-n12-k24-s{600 + t}")
    return out


def _run_metrics(con: sqlite3.Connection, run_id: str) -> Optional[Dict[str, Dict[str, float]]]:
    row = con.execute("SELECT inferred_payload_json FROM nw WHERE simid = ?", (run_id,)).fetchone()
    if not row or not row[0]:
        return None
    payload = json.loads(unpack_json(row[0]) or "{}")
    metrics, statmat = payload.get("metrics") or {}, payload.get("tmp_statmat") or {}
    out = {}
    for alg, m in metrics.items():
        s = statmat.get(alg)
        if s is None:
            continue
        out[alg] = {"f1": float(m["f1_score"]), "total_errors": float(m["total_errors"]), "power": float(s["power"]),
                    "specificity": float(s["specificity"]), "ppv": float(s["ppv"]), "signed_ppv": float(s["signed_ppv"]),
                    "fdr": float(s["fdr"]), "runtime": float(m.get("runtime_ms", 0.0)) / 1000.0}
    return out


def _mean(xs, nd):
    return round(float(np.mean(xs)), nd)


def build_payload(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Aggregates the sweep networks stored in the archive; raises LookupError when they are missing or incomplete."""
    keys = [s.key for s in registry.specs()]
    meta = registry.sweep_metadata()
    con = sqlite3.connect(f"file:{init_db(db_path or get_db_path())}?mode=ro", uri=True)
    try:
        runs = []
        for d in design():
            m = _run_metrics(con, d["run_id"])
            if m is None:
                raise LookupError(f"sweep network {d['run_id']} is not in the archive")
            lack = [k for k in keys if k not in m]
            if lack:
                raise LookupError(f"sweep network {d['run_id']} lacks algorithms {lack}")
            runs.append({**d, "m": m})
    finally:
        con.close()

    def curve(axis: str, values) -> Dict[str, Dict[str, List[float]]]:
        out = {k: {"power": [], "fdr": [], "f1": [], "total_errors": [], "runtime": []} for k in keys}
        for v in values:
            sel = [r for r in runs if r["axis"] == axis and r["value"] == v]
            for k in keys:
                col = lambda f: [r["m"][k][f] for r in sel]  # noqa: E731
                out[k]["power"].append(_mean(col("power"), 1))
                out[k]["fdr"].append(_mean(col("fdr"), 1))
                out[k]["f1"].append(_mean(col("f1"), 3))
                out[k]["total_errors"].append(_mean(col("total_errors"), 1))
                out[k]["runtime"].append(_mean(col("runtime"), 4))
        return out

    size, conn, noise = curve("size", SIZE_VALUES), curve("conn", K_VALUES), curve("noise", NOISE_VALUES)

    leaderboard = []
    for k in keys:
        rows = [r["m"][k] for r in runs]
        leaderboard.append({
            "algorithm_key": k, "algorithm_name": meta[k]["name"], "method_type": meta[k]["type"], "standing_badge": meta[k]["badge"],
            "runs_evaluated": len(rows), "mean_f1": _mean([x["f1"] for x in rows], 3),
            "mean_power_pct": _mean([x["power"] for x in rows], 1), "mean_specificity_pct": _mean([x["specificity"] for x in rows], 1),
            "mean_ppv_pct": _mean([x["ppv"] for x in rows], 1), "mean_signed_ppv_pct": _mean([x["signed_ppv"] for x in rows], 1),
            "mean_fdr_pct": _mean([x["fdr"] for x in rows], 1), "mean_total_errors": _mean([x["total_errors"] for x in rows], 1),
            "mean_runtime_seconds": _mean([x["runtime"] for x in rows], 4)})
    leaderboard.sort(key=lambda x: (x["mean_f1"], -x["mean_total_errors"]), reverse=True)
    for i, item in enumerate(leaderboard):
        item["rank"] = i + 1

    arch: Dict[str, Dict[str, Any]] = {}
    for name, _, label in TOPOLOGY_DEFS:
        sel = [r for r in runs if r["topology"] == name]
        per = {k: {"mean_f1": _mean([r["m"][k]["f1"] for r in sel], 3), "mean_power": _mean([r["m"][k]["power"] for r in sel], 1),
                   "mean_fdr": _mean([r["m"][k]["fdr"] for r in sel], 1),
                   "mean_specificity": _mean([r["m"][k]["specificity"] for r in sel], 1),
                   "mean_errors": _mean([r["m"][k]["total_errors"] for r in sel], 1)} for k in keys}
        best = max(per, key=lambda k: per[k]["mean_f1"])
        mean_of = lambda f: _mean([v[f] for v in per.values()], 1)  # noqa: E731
        arch[name] = {"key": name, "label": label, "runs_evaluated": len(sel),
                      "mean_f1": _mean([v["mean_f1"] for v in per.values()], 3), "mean_power_pct": mean_of("mean_power"),
                      "mean_fdr_pct": mean_of("mean_fdr"), "mean_specificity_pct": mean_of("mean_specificity"),
                      "mean_total_errors": mean_of("mean_errors"), "best_algorithm_key": best,
                      "best_algorithm_name": meta[best]["name"], "best_algorithm_f1": per[best]["mean_f1"], "algorithms": per}

    return {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"archive: {len(runs)} sweep networks, every registered algorithm, read from the stored metrics",
        "total_sweep_runs": len(runs),
        "total_algorithmic_evaluations": len(runs) * len(keys),
        "algorithms_evaluated_count": len(keys),
        "architectures_evaluated": [t[0] for t in TOPOLOGY_DEFS],
        "algorithm_leaderboard": leaderboard,
        "architecture_summary": arch,
        "topology_sweep": {
            "topologies": [t[0] for t in TOPOLOGY_DEFS],
            "summary_by_topology": {n: {"mean_f1": a["mean_f1"], "mean_power": a["mean_power_pct"], "mean_fdr": a["mean_fdr_pct"],
                                        "mean_specificity": a["mean_specificity_pct"], "mean_errors": a["mean_total_errors"]}
                                    for n, a in arch.items()},
            "algorithms_by_topology": {k: {n: arch[n]["algorithms"][k] for n in arch} for k in keys},
        },
        "size_sweep": {"sizes": list(SIZE_VALUES), "algorithms": size},
        "runtime_sweep": {"sizes": list(SIZE_VALUES), "algorithms": {k: size[k]["runtime"] for k in keys}},
        "connectivity_sweep": {"k_values": list(K_VALUES), "algorithms": conn},
        "noise_sweep": {"noise_levels": list(NOISE_VALUES), "algorithms": noise},
    }


def store(payload: Dict[str, Any], db_path: Optional[str] = None, name: str = SNAPSHOT) -> None:
    path = init_db(db_path or get_db_path())
    con = sqlite3.connect(path)
    try:
        con.execute(_TABLE)
        con.execute("INSERT OR REPLACE INTO overall_stats (name, built_at, n_runs, n_algorithms, payload_json) VALUES (?,?,?,?,?)",
                    (name, payload["timestamp"], payload["total_sweep_runs"], payload["algorithms_evaluated_count"],
                     json.dumps(payload)))
        con.commit()
    finally:
        con.close()


def load_stored(db_path: Optional[str] = None, name: str = SNAPSHOT) -> Optional[Dict[str, Any]]:
    path = init_db(db_path or get_db_path())
    con = sqlite3.connect(path)
    try:
        con.execute(_TABLE)
        row = con.execute("SELECT payload_json FROM overall_stats WHERE name = ?", (name,)).fetchone()
    finally:
        con.close()
    return json.loads(row[0]) if row else None


def current(db_path: Optional[str] = None, refresh: bool = False) -> Tuple[Dict[str, Any], str]:
    """(payload, where it came from): the stored snapshot, rebuilt (and stored) when missing, stale or ``refresh`` is set.

    A snapshot is stale when the registry has algorithms it does not list. Raises LookupError when the archive cannot supply
    the sweep networks (the caller falls back to the bundled file)."""
    if not refresh:
        snap = load_stored(db_path)
        if snap is not None and snap.get("algorithms_evaluated_count") == len(registry.specs()):
            return snap, "stored"
    payload = build_payload(db_path)
    store(payload, db_path)
    return payload, "rebuilt"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None)
    ap.add_argument("--no-store", action="store_true")
    a = ap.parse_args()
    pl = build_payload(a.db)
    if not a.no_store:
        store(pl, a.db)
    print(f"{pl['total_sweep_runs']} networks x {pl['algorithms_evaluated_count']} algorithms; top: "
          + ", ".join(f"{x['algorithm_name']} {x['mean_f1']}" for x in pl["algorithm_leaderboard"][:5]))
