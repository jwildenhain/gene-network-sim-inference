"""
SQLite Database Persistence for GRN-Sim / phpIoRN.
Mirrors the original MySQL `rgn_db` schema (`dumpdb.sql`, `runsim.php`, `showpre.php`):
- Table `nw`: stores network parameters and generated graphs/matrices (`simid`, `uid`, `nz`, `nt`, `av_k`, `ratio`, `mdot`, `mnet`, `mpet`, `mmsmt`, `mat`, plus JSON payloads and steady-state / oscillation diagnostics).
- Table `alg`: stores reverse-engineering algorithm results (`simid`, `algid`, `uid`, `spec`, `sens`, `dmf`, `dmo`, plus full confusion matrix and 9-state statistics).
"""

import json
import os
import sqlite3
import tempfile
from typing import Dict, Any, List, Optional

from grn_sim.paths import archive_db_path


def get_db_path() -> str:
    """
    The active SQLite archive path (``$RGN_DB_PATH`` or ``<user data>/rgn_db.sqlite``, see ``grn_sim.paths``), with
    its folder created. When that location cannot be written (a read-only install, a folder owned by another user)
    a per-machine temp file is used instead.
    """
    db_path = str(archive_db_path())
    parent = os.path.dirname(os.path.abspath(db_path))
    try:
        os.makedirs(parent, exist_ok=True)
        writable = os.access(parent, os.W_OK) and (not os.path.exists(db_path) or os.access(db_path, os.W_OK))
    except OSError:
        writable = False
    if not writable:
        return os.path.join(tempfile.gettempdir(), "grn_sim_archive.sqlite")
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
                response_type TEXT NOT NULL DEFAULT 'responsive',
                response_stats_json TEXT NOT NULL DEFAULT '{}',
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

        # Check and migrate columns in nw table
        cur.execute("PRAGMA table_info(nw)")
        cols = [c[1] for c in cur.fetchall()]
        if "response_type" not in cols:
            cur.execute("ALTER TABLE nw ADD COLUMN response_type TEXT NOT NULL DEFAULT 'responsive'")
        if "response_stats_json" not in cols:
            cur.execute("ALTER TABLE nw ADD COLUMN response_stats_json TEXT NOT NULL DEFAULT '{}'")
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

        # Extract or classify perturbation response dynamics
        resp_dyn = sim_payload.get("response_dynamics") or (history_entry or {}).get("response_dynamics")
        if not resp_dyn:
            from grn_sim.simulator.response_dynamics import classify_perturbation_response
            import numpy as np
            X_mat = np.array(sim_payload.get("X", []))
            resp_dyn = classify_perturbation_response(X_mat, ss_diag)
        response_type = str(resp_dyn.get("response_type", "responsive"))
        response_stats_json = json.dumps(resp_dyn)

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
                steady_state, oscillating, response_type, response_stats_json,
                mdot, mnet, mpet, mmsmt, mat,
                settings_json, sim_payload_json, inferred_payload_json, history_entry_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, uid, created_at, nz, nt, av_k, ratio, noise, bio_var, seed,
                steady_state, oscillating, response_type, response_stats_json,
                mdot, mnet, mpet, mmsmt, mat,
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


def _compute_entry_f1(entry: Dict[str, Any]) -> float:
    """Computes F1 score for re-engineering accuracy from power (recall) and PPV (precision)."""
    best_alg = entry.get("best_alg_key") or (entry.get("best_algorithm", "vote").lower())
    m = entry.get("metrics_summary", {}).get(best_alg, {})
    power = float(entry.get("best_power_pct", m.get("power", 0))) / 100.0
    ppv = float(m.get("ppv", 0)) / 100.0
    if power + ppv > 0:
        return round(2.0 * (power * ppv) / (power + ppv), 4)
    tot_err = float(entry.get("best_total_errors", 99))
    return round(1.0 / (1.0 + tot_err), 4)


def load_all_history_from_db(db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads all archived simulation history entries from the SQLite `nw` table (matching `showpre.php`)."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    entries: List[Dict[str, Any]] = []
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT simid, history_entry_json, created_at, response_type, response_stats_json
            FROM nw
            WHERE history_entry_json != '{}' AND history_entry_json != ''
            ORDER BY rowid DESC
            """
        )
        rows = cur.fetchall()
        for idx, (_, hist_json, _, r_type, r_stats_json) in enumerate(rows):
            try:
                entry = json.loads(hist_json)
                entry["id"] = idx + 1
                if "response_type" not in entry or not entry["response_type"]:
                    entry["response_type"] = r_type or "responsive"
                if r_stats_json and r_stats_json != "{}":
                    entry["response_dynamics"] = json.loads(r_stats_json)
                # `is None`, not falsiness: a stored F1 of 0.0 (nothing recovered) must stay 0.0
                if entry.get("f1_score") is None:
                    entry["f1_score"] = _compute_entry_f1(entry)
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
            SELECT settings_json, sim_payload_json, inferred_payload_json, history_entry_json, response_type, response_stats_json
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
        response_type = row[4]
        response_dynamics = json.loads(row[5]) if row[5] else None
        return {
            "settings": settings,
            "sim_payload": sim_payload,
            "inferred_payload": inferred_payload,
            "history_entry": history_entry,
            "response_type": response_type,
            "response_dynamics": response_dynamics,
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


def get_summary_statistics_by_type(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes aggregated summary statistics across all runs in `nw` and `alg`,
    grouped by perturbation response category:
    1. 'responsive': Clear / several perturbations visible (steady state reached, max_amp >= 0.05).
    2. 'flat': Flat levels for each gene with no visible perturbation (steady state, max_amp < 0.05).
    3. 'oscillating': Limit cycle or non-convergent dynamical trajectories (steady state not reached).
    """
    import numpy as np

    path = init_db(db_path)
    conn = sqlite3.connect(path)
    categories = ["responsive", "flat", "oscillating"]
    result: Dict[str, Any] = {
        "total_simulations": 0,
        "categories": {
            cat: {
                "category": cat,
                "label": (
                    "Responsive (Clear Shifts)"
                    if cat == "responsive"
                    else ("Flat (No Visible Perturbation)" if cat == "flat" else "Oscillating (Limit Cycle)")
                ),
                "badge_class": "badge-emerald" if cat == "responsive" else ("badge-secondary" if cat == "flat" else "badge-amber"),
                "badge_icon": "⚡" if cat == "responsive" else ("⚪" if cat == "flat" else "〰"),
                "count": 0,
                "pct_of_total": 0.0,
                "avg_num_genes": 0.0,
                "avg_k_av": 0.0,
                "avg_noise": 0.0,
                "avg_max_amplitude": 0.0,
                "avg_mean_amplitude": 0.0,
                "avg_variance": 0.0,
                "avg_responsive_genes": 0.0,
                "avg_responsive_perts": 0.0,
                "algorithms": {},
                "top_algorithm": "N/A",
                "top_algorithm_errors": 0,
            }
            for cat in categories
        },
    }

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT simid, nz, nt, av_k, noise, steady_state, oscillating,
                   response_type, response_stats_json, sim_payload_json
            FROM nw
            """
        )
        nw_rows = cur.fetchall()
        result["total_simulations"] = len(nw_rows)
        if not nw_rows:
            return result

        cur.execute(
            """
            SELECT simid, algid, spec, sens, total_errors, dmo
            FROM alg
            """
        )
        alg_rows = cur.fetchall()
        alg_by_simid: Dict[str, List[Dict[str, Any]]] = {}
        for s_id, a_id, spec, sens, tot_err, dmo in alg_rows:
            dmo_dict = json.loads(dmo) if dmo else {}
            m = dmo_dict.get("metrics", {})
            alg_by_simid.setdefault(s_id, []).append({
                "algid": a_id,
                "spec": spec,
                "sens": sens,
                "total_errors": tot_err,
                "precision": m.get("precision", spec),
                "recall": m.get("recall", sens),
                "f1_score": m.get("f1_score", 0.0),
                "signed_ppv": m.get("signed_ppv", 0.0),
            })

        cat_runs: Dict[str, List[Dict[str, Any]]] = {c: [] for c in categories}
        for simid, nz, _nt, av_k, noise, ss, osc, r_type, r_stats_json, _sim_j in nw_rows:
            c = r_type if r_type in categories else ("oscillating" if osc or not ss else "responsive")
            stats = json.loads(r_stats_json) if r_stats_json and r_stats_json != "{}" else {}
            cat_runs[c].append({
                "simid": simid,
                "nz": nz,
                "av_k": av_k,
                "noise": noise,
                "max_amp": stats.get("max_amplitude", 0.0),
                "mean_amp": stats.get("mean_amplitude", 0.0),
                "variance": stats.get("variance", 0.0),
                "resp_genes": stats.get("num_responsive_genes", 0),
                "resp_perts": stats.get("num_responsive_perts", 0),
                "algs": alg_by_simid.get(simid, []),
            })

        total = len(nw_rows)
        for cat, runs in cat_runs.items():
            c_dict = result["categories"][cat]
            n_c = len(runs)
            c_dict["count"] = n_c
            c_dict["pct_of_total"] = round((n_c / total) * 100.0, 1) if total > 0 else 0.0
            if n_c > 0:
                c_dict["avg_num_genes"] = round(float(np.mean([r["nz"] for r in runs])), 1)
                c_dict["avg_k_av"] = round(float(np.mean([r["av_k"] for r in runs])), 2)
                c_dict["avg_noise"] = round(float(np.mean([r["noise"] for r in runs])), 3)
                c_dict["avg_max_amplitude"] = round(float(np.mean([r["max_amp"] for r in runs])), 4)
                c_dict["avg_mean_amplitude"] = round(float(np.mean([r["mean_amp"] for r in runs])), 4)
                c_dict["avg_variance"] = round(float(np.mean([r["variance"] for r in runs])), 6)
                c_dict["avg_responsive_genes"] = round(float(np.mean([r["resp_genes"] for r in runs])), 1)
                c_dict["avg_responsive_perts"] = round(float(np.mean([r["resp_perts"] for r in runs])), 1)

                all_algs: Dict[str, List[Dict[str, Any]]] = {}
                for r in runs:
                    for a in r["algs"]:
                        all_algs.setdefault(a["algid"], []).append(a)

                best_alg = "N/A"
                lowest_err = float("inf")
                for a_id, a_list in all_algs.items():
                    avg_err = round(float(np.mean([x["total_errors"] for x in a_list])), 1)
                    avg_f1 = round(float(np.mean([x["f1_score"] for x in a_list])), 3)
                    avg_spec = round(float(np.mean([x["spec"] for x in a_list])) * 100.0, 1)
                    avg_sens = round(float(np.mean([x["sens"] for x in a_list])) * 100.0, 1)
                    avg_prec = round(float(np.mean([x["precision"] for x in a_list])) * 100.0, 1)
                    avg_sppv = round(float(np.mean([x["signed_ppv"] for x in a_list])) * 100.0, 1)
                    c_dict["algorithms"][a_id] = {
                        "algid": a_id,
                        "runs_count": len(a_list),
                        "avg_total_errors": avg_err,
                        "avg_f1": avg_f1,
                        "avg_spec_pct": avg_spec,
                        "avg_sens_pct": avg_sens,
                        "avg_precision_pct": avg_prec,
                        "avg_signed_ppv_pct": avg_sppv,
                    }
                    if avg_err < lowest_err:
                        lowest_err = avg_err
                        best_alg = a_id

                c_dict["top_algorithm"] = best_alg.upper()
                c_dict["top_algorithm_errors"] = lowest_err if best_alg != "N/A" else 0  # never Infinity: it is not valid JSON
    finally:
        conn.close()

    return result
