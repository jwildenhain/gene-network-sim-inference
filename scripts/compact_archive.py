"""Shrinks an existing archive: compresses the big payload columns and drops what nothing reads.

For every run of more than ``LEGACY_COLUMNS_MAX_GENES`` genes (100):
  * the per-algorithm comparison SVGs (``svgs_dot``, ``svgs_dot_grid``, ``svgs_neato``) are removed (the page draws
    such graphs as modules from ``comparison_graphs``; see ``STORED_SVG_MAX_GENES`` in ``web/server.py``);
  * dense ``pred_matrices`` become coordinate lists (``evaluation/prediction_codec``);
  * the write-only legacy columns ``nw.mdot/mnet/mpet/mmsmt/mat`` and ``alg.dmf`` are emptied.
Payload columns above 1 M characters are then stored compressed (``web/database.pack_json``). Smaller runs are left alone.

Nothing is changed unless ``--apply`` is given, and ``--apply`` first copies the database to ``--backup`` (a path that
must not exist yet). After the rewrite every changed run is read back and compared with the original (identical apart
from the intended removals), the file is vacuumed, and the sizes before and after are printed. Do not run it against an
archive a server is writing to.

    python scripts/compact_archive.py path/to/rgn_db.sqlite                      # dry run: what would shrink
    python scripts/compact_archive.py path/to/rgn_db.sqlite --apply --backup path/to/rgn_db.before-compact.sqlite
"""
import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grn_sim.evaluation.prediction_codec import decode_prediction, encode_prediction  # noqa: E402
from grn_sim.web.database import LEGACY_COLUMNS_MAX_GENES, pack_json, unpack_json  # noqa: E402

SVG_KEYS = ("svgs_dot", "svgs_dot_grid", "svgs_neato")


def compact_inferred(inferred: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(inferred)
    for key in SVG_KEYS:
        out.pop(key, None)
    preds = out.get("pred_matrices")
    if isinstance(preds, dict):
        out["pred_matrices"] = {alg: encode_prediction(decode_prediction(p)) for alg, p in preds.items()}
    return out


def same_after_compaction(before: Dict[str, Any], after: Dict[str, Any]) -> Optional[str]:
    """None when ``after`` is ``before`` minus the intended removals, else a description of the difference."""
    for key in set(before) | set(after):
        if key in SVG_KEYS:
            if key in after:
                return f"{key} still present"
            continue
        if key == "pred_matrices":
            if set(before.get(key) or {}) != set(after.get(key) or {}):
                return "pred_matrices algorithms differ"
            for alg, pred in (before.get(key) or {}).items():
                if not np.array_equal(decode_prediction(pred), decode_prediction(after[key][alg])):
                    return f"pred_matrices[{alg}] differs"
            continue
        if before.get(key) != after.get(key):
            return f"{key} differs"
    return None


def human(n: float) -> str:
    return f"{n / 1e6:,.0f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path)
    ap.add_argument("--apply", action="store_true", help="rewrite the database (needs --backup)")
    ap.add_argument("--backup", type=Path, help="copy of the database to make before rewriting (must not exist)")
    args = ap.parse_args()
    if not args.db.exists():
        print(f"no such database: {args.db}")
        return 2
    if args.apply:
        if not args.backup or args.backup.exists():
            print("--apply needs --backup <new path>")
            return 2
        print(f"backing up to {args.backup} ...", flush=True)
        shutil.copy2(args.db, args.backup)

    size_before = args.db.stat().st_size
    con = sqlite3.connect(str(args.db))
    ids = [r[0] for r in con.execute("SELECT simid FROM nw").fetchall()]
    saved_total, changed = 0, 0
    t0 = time.time()
    for n, simid in enumerate(ids, 1):
        nz, sim_raw, inf_raw = con.execute(
            "SELECT nz, sim_payload_json, inferred_payload_json FROM nw WHERE simid = ?", (simid,)).fetchone()
        old_bytes = len(sim_raw) + len(inf_raw)
        if old_bytes < 1_000_000:
            continue  # small runs stay plain text
        sim_text, inf_text = unpack_json(sim_raw), unpack_json(inf_raw)
        big = nz > LEGACY_COLUMNS_MAX_GENES
        inferred = json.loads(inf_text) if inf_text else {}
        new_inferred = compact_inferred(inferred) if big else inferred
        new_sim = pack_json(sim_text)
        new_inf = pack_json(json.dumps(new_inferred))
        new_bytes = len(new_sim) + len(new_inf)
        if new_bytes >= old_bytes * 0.98:
            continue  # nothing to gain (already compact)
        saved_total += max(0, old_bytes - new_bytes)
        changed += 1
        print(f"[{n}/{len(ids)}] {simid} {nz} genes: {human(old_bytes)} -> {human(new_bytes)}", flush=True)
        if not args.apply:
            continue
        if big:
            con.execute("UPDATE nw SET mdot = '', mnet = '', mpet = '', mmsmt = '', mat = '' WHERE simid = ?", (simid,))
            con.execute("UPDATE alg SET dmf = '' WHERE simid = ?", (simid,))
        con.execute("UPDATE nw SET sim_payload_json = ?, inferred_payload_json = ? WHERE simid = ?", (new_sim, new_inf, simid))
        con.commit()
        # read back and compare with what was there before
        row = con.execute("SELECT sim_payload_json, inferred_payload_json FROM nw WHERE simid = ?", (simid,)).fetchone()
        if unpack_json(row[0]) != sim_text:
            raise SystemExit(f"{simid}: simulation payload changed on the round trip; restore from {args.backup}")
        diff = same_after_compaction(inferred, json.loads(unpack_json(row[1])))
        if diff and big:
            raise SystemExit(f"{simid}: inferred payload differs beyond the intended removals ({diff}); restore from {args.backup}")
    print(f"\n{changed} runs {'rewritten' if args.apply else 'would shrink'}, about {human(saved_total)} of payload saved "
          f"({time.time() - t0:.0f} s)")
    if args.apply:
        print("vacuuming ...", flush=True)
        con.execute("VACUUM")
        con.close()
        print(f"file: {human(size_before)} -> {human(args.db.stat().st_size)}")
    else:
        con.close()
        print(f"file now {human(size_before)}; run with --apply --backup <path> to rewrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
