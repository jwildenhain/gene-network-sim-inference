"""
Runs the new algorithms (and S2G / NIR / RidgeCV / GENIE3 for reference) on every benchmark dataset of the studio,
scores them in T[regulator, target] orientation, and times every fit.

    python scripts/benchmark_all_datasets.py [--out docs/all_datasets_benchmark.json]

Datasets without a ground truth (the 105-gene Norman response matrix) are run for time and for agreement with the other
methods only (rank correlation of the top edges), not scored.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any, Dict, List

import numpy as np

from grn_sim.benchmarks.benchmark_datasets import (benchmark_data_model, list_benchmark_datasets, load_benchmark_dataset,
                                                   regulator_target_matrix)
from grn_sim.evaluation.metrics import compute_aupr, evaluate_inference
from grn_sim.inference import orientation, registry

NEW = ["inspre", "mean_difference", "pearson", "gene_features", "grnboost"]
REFERENCE = ["s2g", "nir", "ridge_cv", "genie3", "by_chance"]


def run_dataset(did: str) -> List[Dict[str, Any]]:
    data = load_benchmark_dataset(did)
    X, U = np.asarray(data["X"], dtype=float), np.asarray(data["U"], dtype=float)
    n = X.shape[0]
    truth = data.get("true_matrix")
    has_truth = truth is not None and np.any(np.asarray(truth) != 0) and data.get("true_matrix") is not None
    T = regulator_target_matrix(did, np.asarray(truth)) if has_truth else None
    k = max(1, int(round(np.count_nonzero(T) / n))) if has_truth else 3
    model = benchmark_data_model(did)
    rows: List[Dict[str, Any]] = []
    scores: Dict[str, np.ndarray] = {}
    for key in NEW + REFERENCE:
        spec = registry.get(key)
        algo = registry.build(key, registry.BuildContext(k_est=k, use_reg=False))
        sign = -1.0 if spec.negate_u_in_additive else 1.0
        t = time.perf_counter()
        try:
            res = algo.fit(X, sign * U)
        except Exception as exc:                                      # report, do not stop the run
            rows.append({"dataset": did, "method": key, "error": repr(exc)[:160]})
            continue
        dt = time.perf_counter() - t
        A = orientation.orient_prediction(key, res.A_est, model, target_regulator=False)
        scores[key] = np.abs(A)
        row: Dict[str, Any] = {"dataset": did, "method": key, "genes": n, "conditions": int(X.shape[1]),
                               "seconds": round(dt, 4), "edges_predicted": int(res.total_predicted_edges)}
        if has_truth:
            adj = orientation.orient_prediction(key, res.adjacency_pred, model, target_regulator=False)
            m = evaluate_inference(T, adj, ignore_diagonal=True)
            row.update(aupr=round(compute_aupr(T, A, ignore_diagonal=True), 4), f1=round(m.f1_score, 4),
                       precision=round(m.precision, 4), recall=round(m.recall, 4), true_edges=int(np.count_nonzero(T)),
                       random_aupr=round(float(np.count_nonzero(T)) / (n * (n - 1)), 4))
        rows.append(row)
    if not has_truth:
        ref = scores.get("s2g")
        top = max(1, 3 * n)
        for r in rows:
            S = scores.get(r["method"])
            if S is None or ref is None:
                continue
            a = set(np.argsort(-S.ravel())[:top])
            b = set(np.argsort(-ref.ravel())[:top])
            r["top_edge_overlap_with_s2g"] = round(len(a & b) / top, 3)
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    for d in list_benchmark_datasets():
        try:
            rows += run_dataset(d["id"])
        except Exception as exc:
            rows.append({"dataset": d["id"], "error": repr(exc)[:200]})
    for did in dict.fromkeys(r["dataset"] for r in rows):
        sub = [r for r in rows if r["dataset"] == did]
        print(f"\n== {did}")
        for r in sub:
            if "error" in r:
                print(f"  {r.get('method', ''):16} ERROR {r['error']}")
            elif "aupr" in r:
                print(f"  {r['method']:16} AUPR {r['aupr']:.3f} (random {r['random_aupr']:.3f})  F1 {r['f1']:.3f}  "
                      f"edges {r['edges_predicted']:4d}  {r['seconds']:.3f}s")
            else:
                print(f"  {r['method']:16} edges {r['edges_predicted']:4d}  overlap(top 3n, s2g) "
                      f"{r.get('top_edge_overlap_with_s2g', '')}  {r['seconds']:.3f}s")
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
