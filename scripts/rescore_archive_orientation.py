"""
Rescores archived runs whose predictions were scored in the wrong orientation.

Inference algorithms differ in the orientation of their predicted matrix (see grn_sim.inference.orientation):
before the fix every prediction was compared with the run's ``true_matrix`` as returned, so for example GENIE3,
CausalGRN and DeepRIG were scored transposed on Hill-simulated runs, and the Gardner-style algorithms on the DREAM4,
Norman and Replogle benchmark sets. This script brings the stored predictions into the orientation of the run's
``true_matrix`` and recomputes everything derived from them: the metrics, the 9-state statistics, confusion matrices,
comparison graphs and difference graphs of the inferred payload, the history entry (best algorithm) and the ``alg``
table rows. The simulation itself (sim_payload) is not touched.

AUPR needs the continuous estimate, which runs do not store; it is cleared (shown as "--") for transposed algorithms.

Dry run by default: prints what would change. With --apply the archive is first copied to --backup-dir.
Runs already rescored (or scored by the fixed studio) carry ``pred_orientation`` and are skipped.

    PYTHONPATH=src python scripts/rescore_archive_orientation.py --db data/rgn_db.sqlite [--apply] [--report out.json]
"""

import argparse
import datetime
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.inference.orientation import ADDITIVE_GARDNER, ADDITIVE_PHYSICAL, REGULATOR_INPUT_ALGORITHMS, orient_prediction
from grn_sim.visualization.graph_export import (
    compute_detailed_confusion_matrix,
    compute_new_stats_9state,
    export_9state_graphviz_dot,
    export_comparison_graph_json,
    render_graphviz_svg,
)
from grn_sim.web.server import _drawing_matrix, _inference_data_model, _run_truth_target_regulator

DEFAULT_BACKUP_DIR = Path.home() / "grn-sim-archive-backups"

# Generated as B X = +U until these sets took DREAM4's B X = -U; every run this script rescores predates that.
LEGACY_PHYSICAL_SIGN_DATASETS = frozenset({"norman_2019_tf16", "replogle_2022_p53"})


def _run_context(settings: Dict[str, Any], history: Dict[str, Any]) -> Dict[str, Any]:
    """The fields of an archived run that decide the orientation of its truth and its predictions."""
    return {
        "mode": settings.get("mode") or history.get("mode") or "insilico",
        "settings": settings,
        "dataset_source": settings.get("dataset_source"),
        "loading_mode": settings.get("loading_mode"),
    }


def _fit_mode(run: Dict[str, Any], alg: str) -> str:
    """Perturbation model the studio fitted ``alg`` with (the regulator-input rule of _execute_inference)."""
    settings = run.get("settings") or {}
    mode = str(run.get("mode") or "")
    engine = settings.get("kinetics_engine", "hill_nonlinear")
    use_reg = mode != "sos" and not mode.startswith("benchmark_") and engine in ("hill_nonlinear", "perturb_seq_counts")
    return "regulator" if use_reg and alg in REGULATOR_INPUT_ALGORITHMS else "additive"


def _studio_metrics(old: Dict[str, Any], m: Any, stats_9: Dict[str, Any]) -> Dict[str, Any]:
    new = dict(old)
    new.update({
        "coverage_pct": round(m.coverage_pct, 1),
        "false_positives_pct": round(m.false_positives_pct, 1),
        "inferred_connections_pct": round(m.inferred_connections_pct, 1),
        "fp_per_inferred_pct": round(m.fp_per_inferred_pct, 1),
        "inferred_sign_pct": round(m.inferred_sign_pct, 1),
        "total_errors": m.total_errors,
        "precision": round(m.precision, 3),
        "recall": round(m.recall, 3),
        "f1_score": round(m.f1_score, 3),
        "signed_ppv": round(m.signed_ppv, 3),
        "signed_true_positives": m.signed_true_positives,
        "true_positives": m.true_positives,
        "false_positives": m.false_positives,
        "true_negatives": m.true_negatives,
        "false_negatives": m.false_negatives,
        "sign_errors": m.sign_errors,
        "state_counts": stats_9["state_counts"],
    })
    if "aupr" in new:
        new["aupr"] = None
    return new


def _sweep_entries(old_metrics: Dict[str, Any], old_stat: Dict[str, Any], m: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Metrics and tmp_statmat in the parameter sweep's format (runtime fields kept)."""
    f1 = float(round(m.f1_score, 4))
    metrics = dict(old_metrics)
    metrics.update({
        "total_errors": int(m.total_errors),
        "true_positives": int(m.true_positives),
        "false_positives": int(m.false_positives),
        "true_negatives": int(m.true_negatives),
        "false_negatives": int(m.false_negatives),
        "f1": f1,
    })
    stat = dict(old_stat)
    stat.update({
        "power": float(round(m.inferred_connections_pct, 2)),
        "specificity": float(round(m.specificity * 100.0, 2)),
        "ppv": float(round(m.precision * 100.0, 2)),
        "signed_ppv": float(round(m.signed_ppv * 100.0, 2)),
        "fdr": float(round(m.fp_per_inferred_pct, 2)),
        "fpr": float(round(m.fpr * 100.0, 2)),
        "fnr": float(round(m.fnr * 100.0, 2)),
        "total_errors": int(m.total_errors),
        "f1": f1,
    })
    return metrics, stat


def _power(stat: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[float]:
    if "power" in stat:
        return float(stat["power"])
    if "inferred_connections_pct" in metrics:
        return float(metrics["inferred_connections_pct"])
    return None


def rescore_run(
    simid: str,
    settings: Dict[str, Any],
    sim_payload: Dict[str, Any],
    inferred: Dict[str, Any],
    history: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Rescores one run in place (``inferred`` and ``history`` are modified). Returns a summary of the changes, or
    None when the run has nothing to rescore.
    """
    preds = inferred.get("pred_matrices") or {}
    if not preds or inferred.get("pred_orientation"):
        return None
    true_matrix = np.array(sim_payload.get("true_matrix") or [], dtype=int)
    names = sim_payload.get("gene_names") or [f"G{i + 1}" for i in range(true_matrix.shape[0] if true_matrix.ndim == 2 else 0)]
    if true_matrix.ndim != 2 or not true_matrix.size:
        return None

    run = _run_context(settings, history)
    truth_ctx = dict(sim_payload)
    truth_ctx.setdefault("mode", run["mode"])
    truth_ctx.setdefault("settings", settings)
    truth_tr = _run_truth_target_regulator(truth_ctx)
    data_model = _inference_data_model(run)
    dataset = str(run.get("dataset_source") or str(run["mode"]).replace("benchmark_", ""))
    if data_model == ADDITIVE_GARDNER and dataset in LEGACY_PHYSICAL_SIGN_DATASETS:
        data_model = ADDITIVE_PHYSICAL
    sweep_format = simid.startswith("sweep-")

    metrics_all = inferred.setdefault("metrics", {})
    statmat_all = inferred.setdefault("tmp_statmat", {})
    changes: Dict[str, Dict[str, Any]] = {}
    for alg, raw in preds.items():
        raw_arr = np.array(raw, dtype=int)
        if raw_arr.shape != true_matrix.shape:
            continue
        pred = orient_prediction(alg, raw_arr, data_model, truth_tr, _fit_mode(run, alg))
        if np.array_equal(pred, raw_arr):
            continue
        old_metrics = metrics_all.get(alg, {})
        old_stat = statmat_all.get(alg, {})
        before = {"power": _power(old_stat, old_metrics), "total_errors": old_metrics.get("total_errors", old_stat.get("total_errors"))}

        m = evaluate_inference(true_matrix, pred)
        preds[alg] = pred.tolist()
        if sweep_format:
            metrics_all[alg], statmat_all[alg] = _sweep_entries(old_metrics, old_stat, m)
        else:
            stats_9 = compute_new_stats_9state(true_matrix, pred)
            metrics_all[alg] = _studio_metrics(old_metrics, m, stats_9)
            statmat_all[alg] = stats_9["tmp_statmat"]
            if "confusion_matrix" in inferred:
                inferred["confusion_matrix"][alg] = compute_detailed_confusion_matrix(true_matrix, pred)
            if "comparison_graphs" in inferred:
                inferred["comparison_graphs"][alg] = export_comparison_graph_json(
                    _drawing_matrix(true_matrix, truth_tr), _drawing_matrix(pred, truth_tr), names
                )
            if alg in (inferred.get("svgs_dot") or {}) or alg in (inferred.get("dots") or {}):
                draw_stats = _drawing_matrix(np.array(stats_9["a_stats"], dtype=int), truth_tr)
                dot_h = export_9state_graphviz_dot(draw_stats, names, graph_name=f"Diff_{alg}_dot", layout="dot")
                dot_n = export_9state_graphviz_dot(draw_stats, names, graph_name=f"Diff_{alg}_neato", layout="neato")
                inferred.setdefault("dots", {})[alg] = dot_h
                inferred.setdefault("svgs_dot", {})[alg] = render_graphviz_svg(dot_h, engine="dot", matrix=draw_stats, gene_names=names, is_9state=True)
                inferred.setdefault("svgs_neato", {})[alg] = render_graphviz_svg(dot_n, engine="neato", matrix=draw_stats, gene_names=names, is_9state=True)
                if "svgs_dot_grid" in inferred:
                    inferred["svgs_dot_grid"][alg] = render_graphviz_svg(dot_h, engine="dot", matrix=draw_stats, gene_names=names, is_9state=True, layout="grid")
        after = {"power": _power(statmat_all[alg], metrics_all[alg]), "total_errors": metrics_all[alg].get("total_errors")}
        changes[alg] = {"before": before, "after": after}

    if not changes:
        return None
    primary = inferred.get("primary_algorithm")
    if primary in (inferred.get("comparison_graphs") or {}):
        inferred["comparison_graph"] = inferred["comparison_graphs"][primary]
        inferred["dot"] = (inferred.get("dots") or {}).get(primary, inferred.get("dot", ""))
    inferred["pred_orientation"] = "true_matrix"
    inferred["data_model"] = data_model
    inferred["orientation_rescored"] = {
        "at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transposed_algorithms": sorted(changes),
    }

    best_before = history.get("best_alg_key")
    if history and statmat_all:
        algs = [a for a in statmat_all if a in metrics_all]
        if sweep_format:
            best = max(algs, key=lambda a: (metrics_all[a].get("f1", 0.0), -metrics_all[a].get("total_errors", 0)))
            history["best_f1"] = metrics_all[best].get("f1")
        else:
            best = min(algs, key=lambda a: metrics_all[a].get("total_errors", 0))
        history["best_algorithm"] = best.upper()
        history["best_alg_key"] = best
        history["best_power_pct"] = statmat_all[best].get("power", 0.0)
        history["best_specificity_pct"] = statmat_all[best].get("specificity", 0.0)
        history["best_total_errors"] = metrics_all[best].get("total_errors", 0)
        history["metrics_summary"] = statmat_all
    return {
        "simid": simid,
        "data_model": data_model,
        "truth_target_regulator": truth_tr,
        "changes": changes,
        "best_before": best_before,
        "best_after": history.get("best_alg_key"),
    }


def backup(db: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{db.stem}.{stamp}.before-orientation-rescore{db.suffix}"
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, type=Path, help="archive to rescore (rgn_db.sqlite)")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: dry run)")
    ap.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    ap.add_argument("--report", type=Path, help="write the per-run changes as JSON")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"no archive at {args.db}", file=sys.stderr)
        return 1
    if args.apply:
        print(f"backup: {backup(args.db, args.backup_dir)}")

    conn = sqlite3.connect(str(args.db) if args.apply else f"file:{args.db}?mode=ro", uri=not args.apply)
    summaries: List[Dict[str, Any]] = []
    try:
        rows = conn.execute(
            "SELECT simid, uid, settings_json, sim_payload_json, inferred_payload_json, history_entry_json FROM nw"
        ).fetchall()
        for simid, uid, settings_json, sim_json, inferred_json, history_json in rows:
            settings = json.loads(settings_json or "{}")
            sim_payload = json.loads(sim_json or "{}")
            inferred = json.loads(inferred_json or "{}")
            history = json.loads(history_json or "{}")
            summary = rescore_run(simid, settings, sim_payload, inferred, history)
            if summary is None:
                continue
            summaries.append(summary)
            if not args.apply:
                continue
            conn.execute(
                "UPDATE nw SET inferred_payload_json = ?, history_entry_json = ? WHERE simid = ? AND uid = ?",
                (json.dumps(inferred), json.dumps(history), simid, uid),
            )
            metrics, statmat, dots = inferred.get("metrics", {}), inferred.get("tmp_statmat", {}), inferred.get("dots", {})
            for alg in summary["changes"]:
                stat = statmat.get(alg, {})
                conn.execute(
                    "UPDATE alg SET spec = ?, sens = ?, total_errors = ?, dmf = ?, dmo = ? WHERE simid = ? AND algid = ? AND uid = ?",
                    (
                        round(float(stat.get("specificity", 0.0)) / 100.0, 4),
                        round(float(stat.get("power", 0.0)) / 100.0, 4),
                        int(metrics.get(alg, {}).get("total_errors", 0)),
                        str(dots.get(alg, "")),
                        json.dumps({"tmp_statmat": stat, "metrics": metrics.get(alg, {})}),
                        simid, alg, uid,
                    ),
                )
        if args.apply:
            conn.commit()
    finally:
        conn.close()

    # Summary: per data model and algorithm, mean power / total errors before and after
    agg: Dict[Tuple[str, str], List[List[float]]] = defaultdict(list)
    for s in summaries:
        for alg, c in s["changes"].items():
            b, a = c["before"], c["after"]
            if None not in (b["power"], a["power"], b["total_errors"], a["total_errors"]):
                agg[(s["data_model"], alg)].append([b["power"], a["power"], b["total_errors"], a["total_errors"]])
    print(f"{'apply' if args.apply else 'dry run'}: {len(summaries)} runs with transposed predictions")
    print(f"{'data model':18s} {'algorithm':10s} {'runs':>5s} {'power% before':>14s} {'after':>7s} {'errors before':>14s} {'after':>7s}")
    for (dm, alg), v in sorted(agg.items()):
        m = np.mean(v, axis=0)
        print(f"{dm:18s} {alg:10s} {len(v):5d} {m[0]:14.1f} {m[1]:7.1f} {m[2]:14.1f} {m[3]:7.1f}")
    best_changed = [s for s in summaries if s["best_before"] != s["best_after"]]
    print(f"best algorithm changed in {len(best_changed)} runs")
    if args.report:
        args.report.write_text(json.dumps(summaries, indent=1))
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
