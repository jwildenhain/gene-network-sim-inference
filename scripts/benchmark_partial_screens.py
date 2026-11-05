"""
Every registered algorithm on partial Perturb-seq screens (m perturbed genes out of n measured), scored per block.

    python scripts/benchmark_partial_screens.py --genes 500 --perturbed 30 100 --seeds 3 --cells 100

Per seed: a ``replogle_like`` network, a knockdown screen of ``m`` randomly chosen genes (LNA cells, measurement
artefacts), a calibrated ACE estimate, and the repository's (X, U) pair (X is n x m, U carries each perturbed gene's own
response). Each algorithm is run as it is ("native") and, if it estimates a steady-state Jacobian, through
:class:`grn_sim.inference.partial.PartialAdapter` ("adapted": fit on the perturbed block, extended to the unperturbed
genes). A native fit that fails (shape error, time budget) is reported as such. Scores are oriented to
S[regulator, target] and evaluated per block (see ``grn_sim.evaluation.partial_screen``).
"""

from __future__ import annotations

import argparse
import collections
import json
import signal
import time
import warnings
from typing import Any, Dict, List

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation.partial_screen import evaluate_partial
from grn_sim.inference import orientation, registry
from grn_sim.evaluation.partial_screen import perturbed_genes
from grn_sim.inference.bicycle_lite import BicycleLite
from grn_sim.inference.dotears import DOTEARSLite
from grn_sim.inference.partial import PartialAdapter, extend_node_graph
from grn_sim.perturbseq.pipeline import build_network, run_screen

K = 3
# algorithms whose A_est is a steady-state Jacobian (A X = -U): the adapter applies to them
JACOBIAN_LIKE = ["s2g", "g2s", "vote", "score_vote", "nir", "fast", "ridge_cv", "dmdgrn", "cellbox", "netdeconv"]
NEW_PARTIAL = {"inspre"}          # fits partial screens by itself
SKIP: set = set()


class Budget(Exception):
    pass


def _alarm(seconds: int) -> None:
    signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(Budget()))
    signal.alarm(int(seconds))


def orient(key: str, A: np.ndarray) -> np.ndarray:
    return np.abs(orientation.orient_prediction(key, A, orientation.ADDITIVE_GARDNER, target_regulator=False))


def run_one(key: str, mode: str, X: np.ndarray, U: np.ndarray, budget: int) -> Dict[str, Any]:
    algo = registry.build(key, registry.BuildContext(k_est=K, use_reg=False))
    if mode == "adapted":
        algo = PartialAdapter(algo, key)
    t = time.perf_counter()
    _alarm(budget)
    try:
        res = algo.fit(X, U)
    except Budget:
        return {"status": f"over {budget}s"}
    except Exception as exc:
        return {"status": "error: " + repr(exc)[:70]}
    finally:
        signal.alarm(0)
    S = np.abs(res.A_est) if mode == "adapted" or key in NEW_PARTIAL else orient(key, res.A_est)
    return {"status": "ok", "seconds": round(time.perf_counter() - t, 2), "S": S}


def block_consensus(mats: Dict[str, np.ndarray], P: np.ndarray) -> np.ndarray:
    """Rank-average of S matrices, ranked within each block (P->P and P->unperturbed) so the blocks' scales do not mix."""
    n = next(iter(mats.values())).shape[0]
    inP = np.zeros(n, dtype=bool)
    inP[P] = True
    out = np.zeros((n, n))
    for tgt in (inP, ~inP):
        mask = inP[:, None] & tgt[None, :] & ~np.eye(n, dtype=bool)
        acc = np.zeros((n, n))
        for S in mats.values():
            R = np.zeros((n, n))
            vals = np.abs(S)[mask]
            from scipy.stats import rankdata
            R[mask] = (rankdata(vals) - 1.0) / max(len(vals) - 1, 1)
            acc += R
        out[mask] = acc[mask] / len(mats)
    return out


def run_cell_level(name: str, ace: Any, run: Any, budget: int) -> Dict[str, Any]:
    """bicycle-lite and DOTEARS-lite need the cell-level ACE estimate (and the counts for DOTEARS); both fit the perturbed block
    only and are extended to the unperturbed genes with ``extend_node_graph``."""
    t = time.perf_counter()
    _alarm(budget)
    try:
        if name == "bicycle_lite":
            res = BicycleLite(K, lam=30, cov_weight=0.05).fit_ace(ace, alpha=None)
        else:
            res = DOTEARSLite(K, lam=0.003).fit_adata(run.adata, ace, alpha=None)
        S = np.abs(extend_node_graph(res.A_est, ace))
    except Budget:
        return {"status": f"over {budget}s"}
    except Exception as exc:
        return {"status": "error: " + repr(exc)[:70]}
    finally:
        signal.alarm(0)
    return {"status": "ok", "seconds": round(time.perf_counter() - t, 2), "S": S}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=500)
    ap.add_argument("--perturbed", type=int, nargs="+", default=[30, 100])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--selection", choices=["random", "effective"], default="random",
                    help="how the perturbed genes are chosen (the published tables used random)")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--budget", type=int, default=240, help="seconds per fit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    timed_out: set = set()                                   # (method, mode, m): not retried on later seeds
    for m in a.perturbed:
        for seed in range(1, a.seeds + 1):
            t = time.time()
            run = run_screen(build_network("replogle_like", a.genes, seed=seed), mode="knockdown",
                             cells_per_condition=a.cells, n_perturbed=m, perturbed_selection=a.selection, seed=seed)
            ace = sb.screen_inputs(run, "cells", seed, min_cells=max(10, a.cells // 5))
            X, U = ace.to_xu()
            print(f"m={m} seed={seed}: screen {time.time() - t:.0f}s, X {X.shape}, effective nodes "
                  f"{int((np.abs(ace.own_z) >= 2).sum())}", flush=True)
            kept: Dict[Any, np.ndarray] = {}
            P = perturbed_genes(run.screen)
            for key in registry.keys():
                if key in SKIP:
                    continue
                modes = ["native"] + (["adapted"] if key in JACOBIAN_LIKE else [])
                for mode in modes:
                    if (key, mode, m) in timed_out:
                        r = {"status": f"over {a.budget}s (not retried)"}
                    else:
                        r = run_one(key, mode, X, U, a.budget)
                    if r["status"].startswith("over"):
                        timed_out.add((key, mode, m))
                    row = {"perturbed": m, "seed": seed, "method": key, "mode": mode, "status": r["status"]}
                    if r["status"] == "ok":
                        row["seconds"] = r["seconds"]
                        row.update(evaluate_partial(run.screen, r["S"]))
                    if r["status"] == "ok":
                        kept[(key, mode)] = r["S"]
                    rows.append(row)
                    print(f"   {key:16} {mode:8} {r['status'][:40]:40} "
                          + (f"A {row['aupr_A']:.3f} B {row['aupr_B']:.3f} {r['seconds']:.1f}s" if r["status"] == "ok" else ""),
                          flush=True)
            for name in ("bicycle_lite", "dotears_lite"):
                r = {"status": f"over {a.budget}s (not retried)"} if (name, "native", m) in timed_out else run_cell_level(name, ace, run, a.budget)
                if r["status"].startswith("over"):
                    timed_out.add((name, "native", m))
                row = {"perturbed": m, "seed": seed, "method": name, "mode": "native", "status": r["status"]}
                if r["status"] == "ok":
                    kept[(name, "native")] = r["S"]
                    row["seconds"] = r["seconds"]
                    row.update(evaluate_partial(run.screen, r["S"]))
                rows.append(row)
                print(f"   {name:16} native   {r['status'][:40]:40} "
                      + (f"A {row['aupr_A']:.3f} B {row['aupr_B']:.3f} {r['seconds']:.1f}s" if r["status"] == "ok" else ""), flush=True)
            members = {k[0]: v for k, v in kept.items() if k[0] in ("inspre", "mean_difference", "bicycle_lite", "dotears_lite")}
            if len(members) >= 3:
                S = block_consensus(members, P)
                row = {"perturbed": m, "seed": seed, "method": "consensus(" + "+".join(sorted(members)) + ")", "mode": "native",
                       "status": "ok", "seconds": 0.0}
                row.update(evaluate_partial(run.screen, S))
                rows.append(row)
                print(f"   consensus        native   ok                                       A {row['aupr_A']:.3f} B {row['aupr_B']:.3f}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    for m in a.perturbed:
        agg: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            if r["perturbed"] == m:
                agg[(r["method"], r["mode"])].append(r)
        ok = {k: [r for r in v if r["status"] == "ok"] for k, v in agg.items()}
        base = [r for v in ok.values() for r in v][:1]
        print(f"\n### {a.genes} genes, {m} perturbed, {a.seeds} seeds"
              + (f"  (random: A {base[0]['base_A']:.3f}, B {base[0]['base_B']:.4f})" if base else ""))
        print("| method | mode | AUPR A (P->P) | AUPR B (P->unperturbed) | balanced | scores B? | s | status |")
        print("|---|---|---|---|---|---|---|---|")
        order = sorted(agg, key=lambda k: -np.mean([r["aupr_balanced"] for r in ok[k]]) if ok[k] else 1)
        for k in order:
            v, allr = ok[k], agg[k]
            if not v:
                print(f"| {k[0]} | {k[1]} | - | - | - | - | - | {allr[0]['status']} |")
                continue
            f = lambda key: f"{np.mean([r[key] for r in v]):.3f} ± {np.std([r[key] for r in v]):.3f}"
            print(f"| {k[0]} | {k[1]} | {f('aupr_A')} | {f('aupr_B')} | {f('aupr_balanced')} | "
                  f"{'yes' if np.mean([r['scored_B'] for r in v]) > 0.5 else 'no'} | {np.mean([r['seconds'] for r in v]):.1f} | "
                  f"{len(v)}/{len(allr)} ok |")


if __name__ == "__main__":
    main()
