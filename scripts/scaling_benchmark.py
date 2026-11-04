"""
Run time and memory of the new methods against the number of genes.

    python scripts/scaling_benchmark.py --genes 100 --cells 20 [--truth-only] [--budget 600]

One process per gene count (prints a JSON line; ``ru_maxrss`` is the peak of that process). ``--truth-only`` skips the
cell simulation and times the (X, U) methods on the noise-free log2 fold changes of the screen (their cost depends on
the matrix size only). Cell-level methods (ACE estimation, bicycle-lite, DOTEARS-lite) get ``--budget`` seconds each.
"""

from __future__ import annotations

import argparse
import json
import resource
import signal
import time
import warnings

import numpy as np

K = 3


class Budget(Exception):
    pass


def timed(name, fn, out, budget):
    def handler(signum, frame):
        raise Budget()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(budget))
    t = time.perf_counter()
    try:
        res = fn()
        out[name] = {"seconds": round(time.perf_counter() - t, 2),
                     "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}
        return res
    except Budget:
        out[name] = {"seconds": None, "note": f"stopped after {budget}s"}
    except MemoryError:
        out[name] = {"seconds": None, "note": "out of memory"}
    finally:
        signal.alarm(0)
    return None


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, required=True)
    ap.add_argument("--cells", type=int, default=20)
    ap.add_argument("--truth-only", action="store_true")
    ap.add_argument("--budget", type=int, default=600)
    a = ap.parse_args()
    from grn_sim.inference.inspre import InspreAlgorithm
    from grn_sim.inference.interventional_baselines import (GeneFeatureBaseline, GRNBoostLite, MeanDifference,
                                                            PearsonCorrelation)
    from grn_sim.inference.interventional_consensus import consensus
    from grn_sim.perturbseq.ace import estimate_ace, truth_ace
    from grn_sim.perturbseq.pipeline import build_network, run_screen

    out = {"genes": a.genes, "cells_per_condition": None if a.truth_only else a.cells}
    t = time.perf_counter()
    if a.truth_only:
        from grn_sim.perturbseq.conditions import simulate_conditions
        net = build_network("replogle_like", a.genes, seed=1)
        screen = simulate_conditions(net, mode="knockdown", seed=1)
        out["screen_seconds"] = round(time.perf_counter() - t, 1)
        ace = truth_ace(screen)
        adata = None
    else:
        run = run_screen(build_network("replogle_like", a.genes, seed=1), mode="knockdown",
                         cells_per_condition=a.cells, seed=1)
        out["screen_seconds"] = round(time.perf_counter() - t, 1)
        out["cells"] = int(run.adata.n_obs)
        ace = timed("ace_estimation", lambda: estimate_ace(run.adata, gene_names=list(run.screen.gene_names),
                                                           use_called_guide=True), out, a.budget)
        adata = run.adata
        if ace is None:
            print(json.dumps(out))
            return
    out["nodes"] = int(len(ace.nodes))
    out["effective_nodes"] = int((np.abs(ace.own_z) >= 2).sum())      # nodes with a detectable knockdown: the size of the fits
    X, U = ace.to_xu()
    S = {}
    for name, fn in (("inspre", lambda: InspreAlgorithm(K).fit(X, U)),
                     ("mean_difference", lambda: MeanDifference(K).fit(X, U)),
                     ("pearson", lambda: PearsonCorrelation(K).fit(X, U)),
                     ("gene_features", lambda: GeneFeatureBaseline(K).fit(X, U)),
                     ("grnboost_lite", lambda: GRNBoostLite(K).fit(X, U))):
        r = timed(name, fn, out, a.budget)
        if r is not None:
            S[name] = r.A_est
    if not a.truth_only:
        from grn_sim.inference.bicycle_lite import BicycleLite
        from grn_sim.inference.dotears import DOTEARSLite
        d = ace.denoised(0.05)
        r = timed("inspre_fit_ace", lambda: InspreAlgorithm(K).fit_ace(d), out, a.budget)
        r = timed("bicycle_lite(40 iter)", lambda: BicycleLite(K, lam=30, cov_weight=0.05, max_iter=40).fit_ace(d),
                  out, a.budget)
        r = timed("dotears_lite(2 rounds)", lambda: DOTEARSLite(K, max_outer=2).fit_adata(adata, d), out, a.budget)
    if len(S) >= 2:
        timed("consensus", lambda: consensus({k: np.abs(v) for k, v in S.items() if k != "grnboost_lite"}), out, a.budget)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
