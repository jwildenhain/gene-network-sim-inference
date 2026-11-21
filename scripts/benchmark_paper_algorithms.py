"""
The experimental paper algorithms (``grn_sim.inference.experimental``) against the registered methods on full simulated screens.

    python scripts/benchmark_paper_algorithms.py --genes 50 100 --seeds 3 --out docs/results/paper_algorithms.json
    python scripts/benchmark_paper_algorithms.py --generators linear_sem nb_copula nonlinear_scm --genes 100

Per (generator, genes, seed): one screen (100 cells per perturbation, calibrated ACE estimate), every method fitted on the same
(X, U), scores oriented to S[regulator, target], direct-edge AUPR over the perturbed rows. DCD-FG-lite fits the cells.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
import warnings
from typing import Any, Dict, List

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.inference import orientation, registry
from grn_sim.inference.experimental import DCDFGLite
from grn_sim.inference.llc import LLCMean
from grn_sim.perturbseq.pipeline import build_network, run_screen

BASELINES = ["mean_difference", "inspre", "nir", "ridge_cv", "s2g", "notears", "cellbox"]
ABS_DIRECT = {"inspre", "mean_difference"}      # already S[regulator, target]


def make_screen(generator: str, n: int, seed: int, cells: int):
    if generator == "mechanistic":
        run = run_screen(build_network("replogle_like", n, seed=seed), mode="knockdown", cells_per_condition=cells, seed=seed)
        return run, sb.screen_inputs(run, "cells", seed)
    run = vb.simulate_screen(vb.Setting(generator=generator, n_genes=n, cells=cells), seed)
    return run, sb.screen_inputs(run, "cells", seed, normalisation=vb.NORMALISATION)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--generators", nargs="+", default=["mechanistic"])
    ap.add_argument("--genes", type=int, nargs="+", default=[50, 100])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--skip-dcdfg-above", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    for gen in a.generators:
        for n in a.genes:
            for seed in range(1, a.seeds + 1):
                run, ace = make_screen(gen, n, seed, a.cells)
                X, U = ace.to_xu()
                jobs = {k: (lambda k=k: registry.build(k, registry.BuildContext(k_est=3, use_reg=False)).fit(X, U)) for k in BASELINES}
                jobs["llc_mean"] = lambda: LLCMean().fit(X, U)
                if n <= a.skip_dcdfg_above:
                    jobs["dcdfg_lite"] = lambda: DCDFGLite().fit_adata(run.adata, list(run.screen.gene_names))
                for key, fn in jobs.items():
                    t = time.perf_counter()
                    try:
                        res = fn()
                    except Exception as exc:
                        rows.append({"generator": gen, "genes": n, "seed": seed, "method": key, "status": "error: " + repr(exc)[:60]})
                        continue
                    A = np.abs(res.A_est)
                    S = A if key in ABS_DIRECT or key in ("llc_mean", "dcdfg_lite") else np.abs(
                        orientation.orient_prediction(key, A, orientation.ADDITIVE_GARDNER, target_regulator=False))
                    sc = sb.evaluate_scores(run.screen, S)
                    rows.append({"generator": gen, "genes": n, "seed": seed, "method": key, "status": "ok",
                                 "aupr": sc["aupr_direct"], "base": sc["base_direct"], "seconds": round(time.perf_counter() - t, 2)})
                    print(gen, n, seed, key, round(sc["aupr_direct"], 3), rows[-1]["seconds"], flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    agg: Dict[Any, List[float]] = collections.defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            agg[(r["generator"], r["genes"], r["method"])].append(r["aupr"])
    print("\n| generator | genes | method | AUPR |\n|---|---|---|---|")
    for k, v in agg.items():
        print(f"| {k[0]} | {k[1]} | {k[2]} | {np.mean(v):.3f} ± {np.std(v):.3f} |")


if __name__ == "__main__":
    main()
