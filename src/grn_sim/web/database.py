"""
SQLite Database Persistence for GRN-Sim / phpIoRN.
Mirrors the original MySQL `rgn_db` schema (`dumpdb.sql`, `runsim.php`, `showpre.php`):
- Table `nw`: stores network parameters and generated graphs/matrices (`simid`, `uid`, `nz`, `nt`, `av_k`, `ratio`, `mdot`, `mnet`, `mpet`, `mmsmt`, `mat`, plus JSON payloads and steady-state / oscillation diagnostics).
- Table `alg`: stores reverse-engineering algorithm results (`simid`, `algid`, `uid`, `spec`, `sens`, `dmf`, `dmo`, plus full confusion matrix and 9-state statistics).
"""

import json
import os
import sqlite3
from typing import Dict, Any, List, Optional


DEFAULT_DB_PATH = os.environ.get(
    "RGN_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "rgn_db.sqlite"),
)


def get_db_path() -> str:
    """Returns the active SQLite database file path and ensures its parent directory exists."""
    db_path = os.environ.get("RGN_DB_PATH", DEFAULT_DB_PATH)
    parent = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(parent, exist_ok=True)
    if (os.path.exists(db_path) and not os.access(db_path, os.W_OK)) or not os.access(parent, os.W_OK):
        return "/tmp/rgn_db_test.sqlite"
    return db_path


def init_db(db_path: Optional[str] = None) -> str:
    """Initializes the SQLite database tables (`nw` and `alg`) mirroring `rgn_db` (`dumpdb.sql`)."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nw (
                simid TEXT NOT NULL,
                uid TEXT NOT NULL DEFAULT 'jwil315',
                created_at TEXT NOT NULL,
                nz INTEGER NOT NULL DEFAULT 0,
                nt TEXT NOT NULL DEFAULT '',
                av_k REAL NOT NULL DEFAULT 0.0,
                ratio REAL NOT NULL DEFAULT 0.0,
                noise REAL NOT NULL DEFAULT 0.0,
                bio_var REAL NOT NULL DEFAULT 0.0,
                seed INTEGER NOT NULL DEFAULT 42,
                steady_state INTEGER NOT NULL DEFAULT 1,
                oscillating INTEGER NOT NULL DEFAULT 0,
                mdot TEXT NOT NULL DEFAULT '',
                mnet TEXT NOT NULL DEFAULT '',
                mpet TEXT NOT NULL DEFAULT '',
                mmsmt TEXT NOT NULL DEFAULT '',
                mat TEXT NOT NULL DEFAULT '',
                settings_json TEXT NOT NULL DEFAULT '{}',
                sim_payload_json TEXT NOT NULL DEFAULT '{}',
                inferred_payload_json TEXT NOT NULL DEFAULT '{}',
                history_entry_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (simid, uid)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS alg (
                simid TEXT NOT NULL,
                algid TEXT NOT NULL,
                uid TEXT NOT NULL DEFAULT 'jwil315',
                spec REAL NOT NULL DEFAULT 0.0,
                sens REAL NOT NULL DEFAULT 0.0,
                total_errors INTEGER NOT NULL DEFAULT 0,
                dmf TEXT NOT NULL DEFAULT '',
                dmo TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (uid, algid, simid)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return path


def save_simulation_to_db(
    run_id: str,
    sim_payload: Dict[str, Any],
    inferred_payload: Optional[Dict[str, Any]] = None,
    history_entry: Optional[Dict[str, Any]] = None,
    uid: str = "jwil315",
    db_path: Optional[str] = None,
) -> None:
    """
    Persists or updates a simulation run and its reverse-engineering results
    in the `nw` and `alg` tables (matching `ActionProcess::perform` in `runsim.php`).
    """
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        created_at = (history_entry or {}).get("timestamp") or sim_payload.get("timestamp") or ""
        nz = int(sim_payload.get("num_genes", 0))
        nt = str(sim_payload.get("topology", "power_law"))
        av_k = float(sim_payload.get("k_average", 0.0))
        ratio = float(sim_payload.get("act_ratio", 0.5))
        noise = float(sim_payload.get("noise", 0.0))
        bio_var = float(sim_payload.get("bio_var", 0.0))
        seed = int(sim_payload.get("seed", 42))

        ss_diag = sim_payload.get("steady_state_diagnostics") or {}
        steady_state = 1 if ss_diag.get("is_valid_steady_state", True) else 0
        oscillating = 1 if ss_diag.get("oscillating_detected", False) else 0

        mdot = str(sim_payload.get("svg_hierarchical", ""))
        mnet = str(sim_payload.get("svg_spring", ""))
        mpet = str(sim_payload.get("pert_text", ""))
        mmsmt = str(sim_payload.get("msmt_text", ""))
        mat = json.dumps(sim_payload.get("true_matrix", []))

        settings_json = json.dumps(sim_payload.get("settings", {}))
        sim_json = json.dumps(sim_payload)
        inf_json = json.dumps(inferred_payload or {})
        hist_json = json.dumps(history_entry or {})

        cur.execute(
            """
            INSERT OR REPLACE INTO nw (
                simid, uid, created_at, nz, nt, av_k, ratio, noise, bio_var, seed,
                steady_state, oscillating, mdot, mnet, mpet, mmsmt, mat,
                settings_json, sim_payload_json, inferred_payload_json, history_entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, uid, created_at, nz, nt, av_k, ratio, noise, bio_var, seed,
                steady_state, oscillating, mdot, mnet, mpet, mmsmt, mat,
                settings_json, sim_json, inf_json, hist_json,
            ),
        )

        if inferred_payload:
            tmp_statmat = inferred_payload.get("tmp_statmat", {})
            metrics = inferred_payload.get("metrics", {})
            dots = inferred_payload.get("dots", {})
            for alg_id, stat in tmp_statmat.items():
                # Matches upload_alg in runsim.php: spec = TMP_STATMAT[1][3], sens = TMP_STATMAT[1][4]
                spec = round(float(stat.get("specificity", 0.0)) / 100.0, 4)
                sens = round(float(stat.get("power", 0.0)) / 100.0, 4)
                tot_err = int(metrics.get(alg_id, {}).get("total_errors", 0))
                dmf = str(dots.get(alg_id, ""))
                dmo = json.dumps({
                    "tmp_statmat": stat,
                    "metrics": metrics.get(alg_id, {}),
                })
                cur.execute(
                    """
                    INSERT OR REPLACE INTO alg (simid, algid, uid, spec, sens, total_errors, dmf, dmo)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (run_id, alg_id, uid, spec, sens, tot_err, dmf, dmo),
                )

        conn.commit()
    finally:
        conn.close()


def load_all_history_from_db(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads all archived simulation history entries from the SQLite `nw` table (matching `showpre.php`)."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    entries: List[Dict[str, Any]] = []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT simid, history_entry_json, created_at
            FROM nw
            WHERE history_entry_json != '{}' AND history_entry_json != ''
            ORDER BY created_at DESC, rowid DESC
            """
        )
        rows = cur.fetchall()
        for idx, (_, hist_json, _) in enumerate(rows):
            try:
                entry = json.loads(hist_json)
                entry["id"] = idx + 1
                entries.append(entry)
            except Exception:
                continue
    finally:
        conn.close()
    return entries


def load_run_from_db(run_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Loads a specific simulation run's full `sim_payload`, `inferred_payload`, and `settings` from SQLite."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT settings_json, sim_payload_json, inferred_payload_json, history_entry_json
            FROM nw
            WHERE simid = ?
            LIMIT 1
            """,
            (run_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        settings = json.loads(row[0]) if row[0] else {}
        sim_payload = json.loads(row[1]) if row[1] else {}
        inferred_payload = json.loads(row[2]) if row[2] else {}
        history_entry = json.loads(row[3]) if row[3] else {}
        return {
            "settings": settings,
            "sim_payload": sim_payload,
            "inferred_payload": inferred_payload,
            "history_entry": history_entry,
        }
    finally:
        conn.close()


def get_db_stats(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Returns row counts and metadata for the `nw` and `alg` tables in `rgn_db.sqlite`."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nw")
        nw_count = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM alg")
        alg_count = int(cur.fetchone()[0])
        return {
            "db_path": path,
            "nw_rows": nw_count,
            "alg_rows": alg_count,
        }
    finally:
        conn.close()
