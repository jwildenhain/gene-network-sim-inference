"""Audit of every registered algorithm on one calibrated network: accuracy, time and peak memory.

Each algorithm runs in its own fresh process (so one algorithm's memory cannot hide behind another's, and a crash or an
out-of-memory kill only loses that row) through the same code path as the studio (`_execute_inference`), without touching
the archive. Prints one JSON line per algorithm and a table at the end.

    python scripts/audit_algorithms.py --genes 200 --seed 7                    # all algorithms
    python scripts/audit_algorithms.py --genes 500 --algs s2g,nir,inspre --out audit500.json
    python scripts/audit_algorithms.py --run-id iorn-838581                       # on an archived run instead (nothing generated)
"""
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import json, resource, sys, time
sys.path.insert(0, {src!r})
from grn_sim.web import server
server.save_simulation_to_db = lambda *a, **k: None
t0 = time.time()
if {run_id!r}:
    import types
    import numpy as np
    sim = server._load_archived_run({run_id!r})
    server.CURRENT_STATE["experiment"] = types.SimpleNamespace(X=np.array(sim["X"], dtype=float), U=np.array(sim["U"], dtype=float))
    sim = {{"total_edges": int(np.count_nonzero(np.array(sim["true_matrix"])))}}
else:
    sim = server._execute_simulation({{"num_genes": {genes}, "topology": "modular", "seed": {seed},
                                      "calibration_preset": "replogle_like"}})
t_sim = time.time() - t0
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
t1 = time.time()
out = server._execute_inference([{alg!r}])
t_fit = time.time() - t1
m = out["metrics"][{alg!r}]
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print("RESULT " + json.dumps({{"alg": {alg!r}, "genes": {genes}, "seconds": round(m["runtime_ms"] / 1000.0, 2),
      "wall_seconds": round(t_fit, 2), "peak_mb_before": round(rss_before), "peak_mb_after": round(rss_after),
      "extra_mb": round(rss_after - rss_before), "recall": m["recall"], "precision": m["precision"], "f1": m["f1_score"],
      "aupr": m["aupr"], "true_edges": int(sim.get("total_edges", 0)), "predicted": int(m["true_positives"] + m["false_positives"])}}))
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--run-id", default="", help="audit an archived run (read-only) instead of generating a network")
    ap.add_argument("--algs", default="", help="comma separated; default: every registered algorithm")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per algorithm")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    src = str(Path(__file__).resolve().parents[1] / "src")
    sys.path.insert(0, src)
    from grn_sim.inference import registry
    algs = [a for a in args.algs.split(",") if a] or [s.key for s in registry.specs()]
    rows = []
    for alg in algs:
        code = CHILD.format(src=src, genes=args.genes, seed=args.seed, alg=alg, run_id=args.run_id)
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
            row = json.loads(line[7:]) if line else {"alg": alg, "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:200],
                                                    "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            row = {"alg": alg, "error": f"timeout after {args.timeout} s"}
        row["wall_total_s"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(f"\n{'algorithm':16s} {'time s':>8s} {'extra MB':>9s} {'peak MB':>8s} {'recall':>7s} {'prec':>6s} {'f1':>6s} {'aupr':>6s} {'pred':>6s}")
    for r in rows:
        if "error" in r:
            print(f"{r['alg']:16s} ERROR {r['error']}")
        else:
            print(f"{r['alg']:16s} {r['seconds']:8.1f} {r['extra_mb']:9d} {r['peak_mb_after']:8d} {r['recall']:7.3f} "
                  f"{r['precision']:6.3f} {r['f1']:6.3f} {r['aupr']:6.3f} {r['predicted']:6d}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
