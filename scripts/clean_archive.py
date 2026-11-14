"""Archive clean-up: keep runs up to --max-genes, one key per algorithm, and verify every run has every algorithm.

  * removes runs with more than ``--max-genes`` genes (rows of ``nw`` and ``alg``);
  * renames the legacy ``alg.algid`` "ridge" to "ridge_cv" (drops it when the run has both), so the comparison tables do not
    list one algorithm under two keys;
  * ``--verify``: for every run, the algorithm keys of the payload metrics and of the ``alg`` table are the same and equal
    the registry (all algorithms), and the history entry's ``response_type`` equals the ``nw.response_type`` column; prints the
    runs that are not (nothing is changed);
  * ``--sync-response-types``: sets the history entry's ``response_type`` to the column's value where they differ (inference jobs
    on archived runs used to write "responsive" into the entry of flat and oscillating runs, which emptied the curated views).

Dry run by default; ``--apply`` copies the database to ``--backup`` first (a path that must not exist) and must not run
while a server writes to the archive.

    python scripts/clean_archive.py data/rgn_db.sqlite --max-genes 1000                  # dry run
    python scripts/clean_archive.py data/rgn_db.sqlite --max-genes 1000 --apply --backup ~/grn-sim-archive-backups/x.sqlite
    python scripts/clean_archive.py data/rgn_db.sqlite --verify
"""
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LEGACY_KEYS = {"ridge": "ridge_cv"}


def plan(con: sqlite3.Connection, max_genes: int) -> Dict[str, List]:
    too_big = [r[0] for r in con.execute("SELECT simid FROM nw WHERE nz > ?", (max_genes,))]
    renames, drops = [], []
    for simid, uid, algid in con.execute("SELECT simid, uid, algid FROM alg").fetchall():
        new = LEGACY_KEYS.get(algid)
        if not new:
            continue
        has_new = con.execute("SELECT 1 FROM alg WHERE simid = ? AND uid = ? AND algid = ?", (simid, uid, new)).fetchone()
        (drops if has_new else renames).append((simid, uid, algid))
    return {"remove_runs": too_big, "rename": renames, "drop_duplicate": drops}


def apply(con: sqlite3.Connection, p: Dict[str, List]) -> None:
    for simid in p["remove_runs"]:
        con.execute("DELETE FROM alg WHERE simid = ?", (simid,))
        con.execute("DELETE FROM nw WHERE simid = ?", (simid,))
    for simid, uid, algid in p["drop_duplicate"]:
        con.execute("DELETE FROM alg WHERE simid = ? AND uid = ? AND algid = ?", (simid, uid, algid))
    for simid, uid, algid in p["rename"]:
        con.execute("UPDATE alg SET algid = ? WHERE simid = ? AND uid = ? AND algid = ?", (LEGACY_KEYS[algid], simid, uid, algid))
    con.commit()


def verify(con: sqlite3.Connection) -> List[str]:
    from grn_sim.inference import registry
    from grn_sim.web.database import unpack_json
    full: Set[str] = {s.key for s in registry.specs()}
    table: Dict[str, Set[str]] = {}
    for simid, algid in con.execute("SELECT simid, algid FROM alg"):
        table.setdefault(simid, set()).add(algid)
    bad = []
    for simid, nz, raw in con.execute("SELECT simid, nz, inferred_payload_json FROM nw").fetchall():
        payload = set(json.loads(unpack_json(raw) or "{}").get("metrics", {})) if raw and len(raw) > 2 else set()
        have = table.get(simid, set())
        if payload != full or have != full:
            bad.append(f"{simid} ({nz} genes): payload missing {sorted(full - payload)}, alg table missing {sorted(full - have)}, "
                       f"alg table extra {sorted(have - full)}, payload extra {sorted(payload - full)}")
    return bad


def response_type_mismatches(con: sqlite3.Connection) -> List[tuple]:
    """(simid, column value, history-entry value) of the runs whose history entry names another dynamics type than the column."""
    out = []
    for simid, col, hist in con.execute("SELECT simid, response_type, history_entry_json FROM nw").fetchall():
        entry = json.loads(hist or "{}") if hist and hist != "{}" else None
        if entry is not None and col and entry.get("response_type") != col:
            out.append((simid, col, entry.get("response_type")))
    return out


def sync_response_types(con: sqlite3.Connection) -> int:
    """Makes the history entries agree with the column; returns how many were changed."""
    changed = 0
    for simid, col, _ in response_type_mismatches(con):
        entry = json.loads(con.execute("SELECT history_entry_json FROM nw WHERE simid = ?", (simid,)).fetchone()[0])
        entry["response_type"] = col
        con.execute("UPDATE nw SET history_entry_json = ? WHERE simid = ?", (json.dumps(entry), simid))
        changed += 1
    con.commit()
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path)
    ap.add_argument("--max-genes", type=int, default=1000)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", type=Path)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sync-response-types", action="store_true", help="needs --apply --backup")
    args = ap.parse_args()
    if args.verify:
        con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
        bad = verify(con) + [f"{sid}: history entry says {entry!r}, the column says {col!r}" for sid, col, entry in response_type_mismatches(con)]
        total = con.execute("SELECT COUNT(*) FROM nw").fetchone()[0]
        for line in bad:
            print(line)
        print(f"{total - len(bad)} of {total} runs have every algorithm under one key in the payload and the alg table")
        return 1 if bad else 0
    if args.apply:
        if not args.backup or args.backup.exists():
            print("--apply needs --backup <new path>")
            return 2
        shutil.copy2(args.db, args.backup)
        print(f"backed up to {args.backup}")
    con = sqlite3.connect(str(args.db))
    if args.sync_response_types:
        print(f"{sync_response_types(con) if args.apply else len(response_type_mismatches(con))} history entries "
              f"{'changed' if args.apply else 'would change'} to the column's response type")
        con.close()
        return 0
    p = plan(con, args.max_genes)
    print(f"remove {len(p['remove_runs'])} runs over {args.max_genes} genes {p['remove_runs']}; rename {len(p['rename'])} legacy "
          f"alg keys; drop {len(p['drop_duplicate'])} duplicate legacy keys")
    if args.apply:
        apply(con, p)
        con.execute("VACUUM")
        print("applied and vacuumed")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
