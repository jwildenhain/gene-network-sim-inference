"""Scale benchmark numbers of docs/SCALE_BENCHMARK.md, from the simulation archive.

``snapshot`` reads the archive (read-only) and writes ``docs/results/scale_benchmark.json``: per algorithm F1, AUPR and run
time on the 500 / 1000 / 2000 gene networks of the scale run and the mean over the small (12-16 gene) archived runs.
``render`` puts the table between the ``scale-table`` markers of the page; ``tests/test_scale_benchmark_doc.py`` fails when
the page and the snapshot differ. The snapshot is committed so the page can be checked without the archive.

    python scripts/render_scale_benchmark.py snapshot data/rgn_db.sqlite     # needs the archive (never a live write target)
    python scripts/render_scale_benchmark.py render                          # rewrite the table in the page
    python scripts/render_scale_benchmark.py --check                         # exit 1 when the page is out of date
"""
import argparse
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SNAPSHOT = ROOT / "docs" / "results" / "scale_benchmark.json"
PAGE = ROOT / "docs" / "SCALE_BENCHMARK.md"
START, END = "<!-- scale-table:start -->", "<!-- scale-table:end -->"
SCALE_RUNS = {"500": "iorn-838581", "1000": "iorn-592020", "2000": "iorn-846813"}   # one network per size, seed 7
CONTROLS = ("by_chance", "gene_features")
SMALL_GENES = (12, 16)
LABELS = {"s2g": "S2G", "g2s": "G2S", "vote": "Voting ensemble", "score_vote": "Score vote", "nir": "NIR", "fast": "FAST",
          "ridge_cv": "RidgeCV", "atlasshift": "AtlasShift", "causalgrn": "CausalGRN", "sfinn": "SFINN", "notears": "NOTEARS",
          "genie3": "GENIE3", "grnboost": "GRNBoost", "inspre": "inspre", "cellbox": "CellBox", "cpa_lite": "CPA-Lite",
          "dmdgrn": "DMD-GRN", "deeprig": "DeepRIG", "arachne": "ARACNE", "netdeconv": "NetDeconv", "pearson": "Pearson",
          "mean_difference": "Mean difference"}


def _metrics(con: sqlite3.Connection, simid: str) -> Dict[str, Any]:
    from grn_sim.web.database import unpack_json
    raw = con.execute("SELECT inferred_payload_json FROM nw WHERE simid = ?", (simid,)).fetchone()[0]
    return json.loads(unpack_json(raw) or "{}").get("metrics", {})


def snapshot(db: Path) -> Dict[str, Any]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows: Dict[str, Dict[str, Any]] = {}
    previous = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    removed: Dict[str, str] = dict(previous.get("removed_runs", {}))
    for size, simid in SCALE_RUNS.items():
        if con.execute("SELECT 1 FROM nw WHERE simid = ?", (simid,)).fetchone() is None:
            # the run was taken out of the archive (2000 genes, 2026-11-13); its numbers stay as recorded in the snapshot
            removed[size] = simid
            for alg, d in previous.get("algorithms", {}).items():
                if size in d:
                    rows.setdefault(alg, {})[size] = d[size]
            continue
        for alg, m in _metrics(con, simid).items():
            rows.setdefault(alg, {})[size] = {"f1": round(m["f1_score"], 3), "aupr": round(m.get("aupr") or 0.0, 3),
                                              "seconds": round(m["runtime_ms"] / 1000.0, 1)}
    small: Dict[str, List[List[float]]] = {}
    n_runs = 0
    for (simid,) in con.execute("SELECT simid FROM nw WHERE nz BETWEEN ? AND ?", SMALL_GENES).fetchall():
        m = _metrics(con, simid)
        if not m:
            continue
        n_runs += 1
        for alg, v in m.items():
            small.setdefault(alg, []).append([v["f1_score"], v.get("aupr")])      # AUPR is None in rescored runs
    for alg, vals in small.items():
        aupr = [v[1] for v in vals if isinstance(v[1], (int, float))]
        rows.setdefault(alg, {})["small"] = {"f1": round(st.mean(v[0] for v in vals), 3),
                                             "aupr": round(st.mean(aupr), 3) if aupr else 0.0, "runs": len(vals),
                                             "aupr_runs": len(aupr)}
    return {"runs": SCALE_RUNS, "removed_runs": removed, "small_genes": list(SMALL_GENES), "small_runs_with_results": n_runs, "algorithms": rows}


def render_table(snap: Dict[str, Any]) -> str:
    algs = snap["algorithms"]
    order = sorted((a for a in algs if a not in CONTROLS and "2000" in algs[a]),
                   key=lambda a: (-algs[a]["2000"]["aupr"], a))
    lo, hi = snap["small_genes"]
    sizes = ("500", "1000", "2000")

    def cell(d: Optional[Dict[str, Any]]) -> str:
        return "-" if not d else f"{d['f1']:.2f} / {d['aupr']:.2f}"

    def secs(d: Optional[Dict[str, Any]]) -> str:
        return "-" if not d else f"{d['seconds']:,.0f}"

    L = [f"| algorithm | {lo}-{hi} genes (archive mean) | 500 genes | 1000 genes | 2000 genes | seconds at 500 / 1000 / 2000 |",
         "|---|---|---|---|---|---|"]
    for a in order:
        d = algs[a]
        L.append(f"| {LABELS.get(a, a)} | {cell(d.get('small'))} | " + " | ".join(cell(d.get(s)) for s in sizes)
                 + " | " + " / ".join(secs(d.get(s)) for s in sizes) + " |")
    L.append("")
    L.append("Cells are F1 / AUPR. Controls (`by_chance`, `gene_features`) are not listed. Ranked by AUPR at 2000 genes. Seconds are as stored: "
             "different hosts, core counts and image versions, so compare orders of magnitude only. "
             "Networks: " + ", ".join(f"{s} genes `{snap['runs'][s]}`" for s in sizes)
             + ("; " + ", ".join(f"the {s}-gene run `{r}` was removed from the archive on 2026-11-13 (work limited to 1000 genes), its numbers are kept from the snapshot"
                                  for s, r in sorted(snap.get("removed_runs", {}).items())) if snap.get("removed_runs") else "")
             + f"; the small-network column is the mean over {snap['small_runs_with_results']} archived runs "
             f"({lo}-{hi} genes, algorithms that were run on them).")
    return "\n".join(L)


def render_page(snap: Dict[str, Any]) -> str:
    text = PAGE.read_text()
    a, b = text.index(START), text.index(END)
    return text[:a + len(START)] + "\n" + render_table(snap) + "\n" + text[b:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=("snapshot", "render"), default="render")
    ap.add_argument("db", nargs="?", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        ok = render_page(json.loads(SNAPSHOT.read_text())) == PAGE.read_text()
        if not ok:
            print(f"{PAGE} is out of date: run scripts/render_scale_benchmark.py render")
        return 0 if ok else 1
    if args.command == "snapshot":
        if not args.db:
            print("give the archive path")
            return 2
        SNAPSHOT.write_text(json.dumps(snapshot(args.db), indent=1, sort_keys=True) + "\n")
        print(f"wrote {SNAPSHOT}")
        return 0
    PAGE.write_text(render_page(json.loads(SNAPSHOT.read_text())))
    print(f"wrote {PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
