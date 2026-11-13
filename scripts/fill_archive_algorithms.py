"""Queues the missing algorithms on every archived run, through the studio's job queue (so results are stored the usual way).

For each run up to ``--max-genes`` genes, the algorithms of the registry that the run's payload lacks are submitted as one
inference job on the archived network. Nothing that is already stored is recomputed. The archive is only read here (read-only
connection); the server writes the results. Resumable: run it again and it submits what is still missing.

    python scripts/fill_archive_algorithms.py --db data/rgn_db.sqlite --url http://localhost:8085 [--max-genes 1000] [--dry-run]
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def missing_by_run(db: Path, max_genes: int) -> Dict[str, List[str]]:
    from grn_sim.inference import registry
    from grn_sim.web.database import unpack_json
    full = [s.key for s in registry.specs()]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    for simid, raw in con.execute("SELECT simid, inferred_payload_json FROM nw WHERE nz <= ? ORDER BY nz, simid", (max_genes,)).fetchall():
        have = set(json.loads(unpack_json(raw) or "{}").get("metrics", {})) if raw and len(raw) > 2 else set()
        todo = [k for k in full if k not in have]
        if todo:
            out[simid] = todo
    return out


def call(url: str, path: str, body=None):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--url", default="http://localhost:8085")
    ap.add_argument("--max-genes", type=int, default=1000)
    ap.add_argument("--active", type=int, default=100, help="jobs queued at a time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    todo = missing_by_run(args.db, args.max_genes)
    print(f"{len(todo)} runs lack algorithms; {sum(len(v) for v in todo.values())} (run, algorithm) pairs", flush=True)
    if args.dry_run:
        return 0
    pending = list(todo.items())
    mine: Dict[str, str] = {}      # job id -> run id
    failed: Dict[str, str] = {}
    while pending or mine:
        _, listing = call(args.url, "/api/jobs")
        state = {j["id"]: j for j in listing["jobs"]}
        for jid in list(mine):
            j = state.get(jid)
            if j is None or j["status"] in ("done", "failed", "cancelled"):
                if j is not None and j["status"] != "done":
                    failed[mine[jid]] = j.get("error") or j["status"]
                elif j is not None and (j.get("summary") or {}).get("algorithms_failed"):
                    failed[mine[jid]] = json.dumps(j["summary"]["algorithms_failed"])[:300]
                del mine[jid]
        while pending and len(mine) < args.active:
            simid, algs = pending.pop(0)
            code, res = call(args.url, "/api/jobs", {"run_id": simid, "algorithms": algs, "label": f"fill {simid}"})
            if code == 202:
                mine[res["jobs"][0]["id"]] = simid
            else:
                failed[simid] = f"submit {code}: {res.get('error')}"
        print(time.strftime("%H:%M:%S"), f"queued/running {len(mine)}, to submit {len(pending)}, problems {len(failed)}", flush=True)
        time.sleep(30)
    print("problems:", json.dumps(failed, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
