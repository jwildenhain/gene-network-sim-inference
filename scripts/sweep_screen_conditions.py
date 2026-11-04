"""
Sweeps of the screen conditions (cells per perturbation, guide efficiency) and ablations of the optional switches.

    python scripts/sweep_screen_conditions.py --sweep [--genes 100 --seeds 3]
    python scripts/sweep_screen_conditions.py --ablate

``--sweep``: direct-edge AUPR of the fast methods on calibrated, denoised ACE input for 10..100 cells per perturbation and
guide efficiencies 0.5..0.92. ``--ablate``: the optional settings of the new methods on the same screens: KS instead of
Welch test, per-entry inspre weights, unrestricted candidate sets for bicycle-lite and DOTEARS-lite.
"""
import argparse
import collections
import json
import time
import warnings

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.inference.bicycle_lite import BicycleLite
from grn_sim.inference.dotears import DOTEARSLite
from grn_sim.inference.inspre import InspreAlgorithm
from grn_sim.inference.interventional_baselines import MeanDifference
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.perturbseq.ace import estimate_ace
from grn_sim.perturbseq.pipeline import build_network, run_screen

K = 3


def fast_methods(ace):
    X, U = ace.to_xu()
    leg = lambda algo, tr=True: (lambda: np.abs(algo.fit(X, U).A_est).T if tr else np.abs(algo.fit(X, U).A_est))
    return {
        "inspre": lambda: np.abs(InspreAlgorithm(K).fit_ace(ace, alpha=0.05).A_est),
        "mean_difference": leg(MeanDifference(K), False),
        "s2g": leg(SimpleToGeneral()),
        "nir": leg(NIRAlgorithm(k_interactions=K)),
        "ridge_cv(self)": leg(KaggleMultiOutputRidge(k_interactions=K)),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = collections.defaultdict(list)
    if a.sweep:
        configs = [("cells", c, 0.92) for c in (10, 20, 40, 100)] + [("efficiency", 40, e) for e in (0.5, 0.7)]
        for label, cells, eff in configs:
            for s in range(1, a.seeds + 1):
                run = run_screen(build_network("replogle_like", a.genes, seed=s), mode="knockdown", cells_per_condition=cells,
                                 guide_efficiency_mean=eff, seed=s)
                ace = sb.screen_inputs(run, "cells", s, min_cells=min(20, max(5, cells // 3)))
                for name, f in fast_methods(ace).items():
                    res[(cells, eff, name)].append(sb.evaluate_scores(run.screen, f())["aupr_direct"])
        print("| cells | guide eff. | " + " | ".join(fast_methods(ace)) + " |")
        print("|---|---|" + "---|" * len(fast_methods(ace)))
        for cells, eff in dict.fromkeys((c, e) for c, e, _ in res):
            print(f"| {cells} | {eff} | " + " | ".join(f"{np.mean(res[(cells, eff, n)]):.3f}" for n in fast_methods(ace)) + " |")
    if a.ablate:
        for s in range(1, a.seeds + 1):
            run = run_screen(build_network("replogle_like", a.genes, seed=s), mode="knockdown", cells_per_condition=40, seed=s)
            names = list(run.screen.gene_names)
            for test in ("welch", "ks"):
                ace = estimate_ace(run.adata, gene_names=names, use_called_guide=True, test=test, seed=s)
                d = ace.denoised(0.05)
                variants = {
                    f"inspre | test={test}": lambda: InspreAlgorithm(K).fit_ace(ace, alpha=0.05),
                }
                if test == "welch":
                    variants.update({
                        "inspre | entry weights": lambda: InspreAlgorithm(K).fit_ace(ace, alpha=0.05, entry_weights=True),
                        "bicycle_lite | 12 candidates": lambda: BicycleLite(K, lam=30, cov_weight=0.05).fit_ace(d),
                        "bicycle_lite | all pairs": lambda: BicycleLite(K, lam=30, cov_weight=0.05, candidates_per_node=None).fit_ace(d),
                        "dotears_lite | 12 candidates": lambda: DOTEARSLite(K, lam=0.003).fit_adata(run.adata, d, alpha=None),
                        "dotears_lite | all pairs": lambda: DOTEARSLite(K, lam=0.003, candidates_per_node=None).fit_adata(run.adata, d, alpha=None),
                    })
                for name, f in variants.items():
                    t = time.time()
                    S = np.abs(f().A_est)
                    res[name].append((sb.evaluate_scores(run.screen, S)["aupr_direct"], time.time() - t))
        print("| variant | direct AUPR | seconds |\n|---|---|---|")
        for name, v in res.items():
            print(f"| {name} | {np.mean([x[0] for x in v]):.3f} ± {np.std([x[0] for x in v]):.3f} | {np.mean([x[1] for x in v]):.1f} |")
    if a.out:
        json.dump({str(k): v for k, v in res.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
