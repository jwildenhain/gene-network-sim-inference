"""
Benchmark of network-inference methods on simulated Perturb-seq screens, against direct and total-effect truth.

    python scripts/benchmark_screen_methods.py --genes 100 --cells 40 --seeds 5 --out docs/screen_method_benchmark.json
    python scripts/benchmark_screen_methods.py --genes 100 --cells 40 --seeds 5 --prior      # prior-quality experiment

Per seed a ``replogle_like`` network is built, a knockdown screen simulated (LNA cells, measurement artefacts), and every
method is given the same calibrated ACE estimate of the cells, once as it is and once denoised (effects not significant
at BH 5% set to zero). ``truth`` rows use the noise-free log2 fold changes (the best a method can do on this network).
The older Gardner-family algorithms are fitted as ``A X = -U`` with U carrying each node's own response on the
diagonal; their output is A[target, regulator] and is transposed for scoring (see grn_sim.inference.orientation).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import Any, Callable, Dict, List

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation.prior_experiment import boost_scores, make_prior, prior_penalty, summarise
from grn_sim.inference.bicycle_lite import BicycleLite
from grn_sim.inference.dotears import DOTEARSLite
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.inspre import InspreAlgorithm
from grn_sim.inference.interventional_baselines import (GeneFeatureBaseline, GRNBoostLite, MeanDifference,
                                                        PearsonCorrelation)
from grn_sim.inference.interventional_consensus import consensus
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.perturbseq.pipeline import build_network, run_screen

K = 3


def legacy(factory: Callable[[], Any], target_regulator: bool = True):
    return lambda ace, run: sb.score_with_algorithm(factory(), ace, "target_regulator" if target_regulator else "regulator_target")


def build_methods(skip: List[str]) -> Dict[str, Callable[[Any, Any], np.ndarray]]:
    """name -> f(ace, run) -> S[regulator, target]. ``run`` gives the h5ad for the methods that need cells."""
    m: Dict[str, Callable[[Any, Any], np.ndarray]] = {
        "inspre": lambda a, r: np.abs(InspreAlgorithm(K).fit_ace(a, alpha=None).A_est),
        "mean_difference": legacy(lambda: MeanDifference(K), False),
        "pearson": legacy(lambda: PearsonCorrelation(K), False),
        "gene_features": legacy(lambda: GeneFeatureBaseline(K), False),
        "grnboost_lite": legacy(lambda: GRNBoostLite(K)),
        "s2g": legacy(SimpleToGeneral),
        "nir": legacy(lambda: NIRAlgorithm(k_interactions=K)),
        "ridge_cv (default, no self)": legacy(lambda: KaggleMultiOutputRidge(k_interactions=K)),
        "ridge_cv (self allowed)": legacy(lambda: KaggleMultiOutputRidge(k_interactions=K, allow_self_regulation=True)),
        "g2s (loo)": legacy(lambda: GeneralToSpecific(drop_rule="loo", max_interactions=2 * K)),
    }
    if "bicycle" not in skip:
        m["bicycle_lite"] = lambda a, r: np.abs(BicycleLite(K, lam=30, cov_weight=0.05).fit_ace(a, alpha=None).A_est)
    if "dotears" not in skip:
        m["dotears_lite"] = lambda a, r: np.abs(DOTEARSLite(K, lam=0.003).fit_adata(r.adata, a, alpha=None).A_est)
    return m


def run_once(n: int, cells: int, seed: int, skip: List[str]) -> List[Dict[str, Any]]:
    run = run_screen(build_network("replogle_like", n, seed=seed), mode="knockdown", cells_per_condition=cells, seed=seed)
    rows: List[Dict[str, Any]] = []
    methods = build_methods(skip)
    inputs = {"cells": sb.screen_inputs(run, "cells", seed)}
    inputs["cells+denoised"] = inputs["cells"].denoised(0.05)
    inputs["truth"] = sb.screen_inputs(run, "truth", seed)
    scores: Dict[str, Dict[str, np.ndarray]] = {k: {} for k in inputs}
    for kind, ace in inputs.items():
        for name, f in methods.items():
            if kind == "truth" and name in ("dotears_lite", "bicycle_lite"):
                continue
            t = time.time()
            try:
                S = f(ace, run)
            except Exception as exc:                                   # a failing method must not stop the benchmark
                rows.append({"seed": seed, "input": kind, "method": name, "error": repr(exc)[:200]})
                continue
            scores[kind][name] = S
            row = {"seed": seed, "input": kind, "method": name, "seconds": round(time.time() - t, 2)}
            row.update(sb.evaluate_scores(run.screen, S))
            rows.append(row)
        members = {k: v for k, v in scores[kind].items() if k in ("inspre", "mean_difference", "bicycle_lite", "dotears_lite")}
        if len(members) >= 3:
            _, _, node_rows = sb.truth_matrices(run.screen)
            S = consensus(members, rows=node_rows)
            row = {"seed": seed, "input": kind, "method": "consensus(" + "+".join(sorted(members)) + ")", "seconds": 0.0}
            row.update(sb.evaluate_scores(run.screen, S))
            rows.append(row)
    return rows


def run_prior(n: int, cells: int, seed: int) -> List[Dict[str, Any]]:
    run = run_screen(build_network("replogle_like", n, seed=seed), mode="knockdown", cells_per_condition=cells, seed=seed)
    ace = sb.screen_inputs(run, "cells", seed).denoised(0.05)
    direct, _, node_rows = sb.truth_matrices(run.screen)
    rng = np.random.default_rng(seed)
    S_ins0 = np.abs(InspreAlgorithm(K).fit_ace(ace, alpha=None).A_est)
    S_md0 = np.abs(MeanDifference(K).fit(*ace.to_xu()).A_est)
    out = []
    for sens in (0.2, 0.5, 0.8):
        for prec in (0.0, 0.2, 0.5, 0.9):
            prior = make_prior(direct, node_rows, sens, prec, rng)
            S_ins = np.abs(InspreAlgorithm(K).fit_ace(ace, alpha=None, edge_penalty=prior_penalty(prior)).A_est)
            S_md = boost_scores(S_md0, prior)
            for name, S0, S1 in (("inspre", S_ins0, S_ins), ("mean_difference", S_md0, S_md)):
                row = {"seed": seed, "method": name, "target_sensitivity": sens, "target_precision": prec}
                row.update(summarise(direct, node_rows, prior, S0, S1))
                out.append(row)
    return out


def table(rows: List[Dict[str, Any]], keys=("aupr_direct", "p_at_k_direct", "aupr_total", "p_at_k_total", "seconds")) -> str:
    import collections
    agg: Dict[Any, Dict[str, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if "error" in r:
            continue
        for k in keys:
            agg[(r["input"], r["method"])][k].append(r[k])
    lines = []
    for inp in ("truth", "cells", "cells+denoised"):
        sub = [(m, v) for (i, m), v in agg.items() if i == inp]
        sub.sort(key=lambda t: -np.mean(t[1]["aupr_direct"]))
        lines.append(f"\n### input: {inp}\n")
        lines.append("| method | direct AUPR | direct P@k | total AUPR | total P@k | s |")
        lines.append("|---|---|---|---|---|---|")
        for m, v in sub:
            f = lambda k: f"{np.mean(v[k]):.3f} ± {np.std(v[k]):.3f}"
            lines.append(f"| {m} | {f('aupr_direct')} | {f('p_at_k_direct')} | {f('aupr_total')} | {f('p_at_k_total')} | {np.mean(v['seconds']):.1f} |")
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip", nargs="*", default=[], help="bicycle dotears")
    ap.add_argument("--prior", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows: List[Dict[str, Any]] = []
    for s in range(1, a.seeds + 1):
        t = time.time()
        rows += run_prior(a.genes, a.cells, s) if a.prior else run_once(a.genes, a.cells, s, a.skip)
        print(f"seed {s} done in {time.time() - t:.0f}s", flush=True)
    if a.prior:
        import collections
        agg: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            agg[(r["method"], r["target_sensitivity"], r["target_precision"])].append(r)
        print("\n| method | sens | prec | no prior | prior alone | with prior | novel no prior | novel with prior |")
        print("|---|---|---|---|---|---|---|---|")
        for (m, se, pr), v in sorted(agg.items()):
            f = lambda k: f"{np.mean([x[k] for x in v]):.3f}"
            print(f"| {m} | {se} | {pr} | {f('no_prior')} | {f('prior_alone')} | {f('with_prior')} | {f('novel_no_prior')} | {f('novel_with_prior')} |")
    else:
        print(table(rows))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"genes": a.genes, "cells": a.cells, "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
