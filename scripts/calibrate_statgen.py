"""
Calibrates the statistical generators (grn_sim.statgen) towards genome-scale Perturb-seq, like
scripts/calibrate_perturbseq.py does for the mechanistic presets.

    python scripts/calibrate_statgen.py --generator linear_sem [--genes 100 --seeds 2] [--out results.json]

For every (effect_scale, noise) of the grid a knockdown screen (40 cells per perturbation, measurement artefacts) is
simulated on the modular replogle_like graph, and the observed statistics are compared with the realism targets
(grn_sim.perturbseq.realism.TARGETS; ``fraction_effects_indirect`` is left out: it is a genome-scale property no 100-gene
network can meet). Prints the grid sorted by the mean realism score (0 = every statistic inside its range).
"""
import argparse
import json
import warnings

import numpy as np

from grn_sim.statgen import ModelParams, preset_graph, run_stat_screen

GRIDS = {
    "linear_sem": ([0.15, 0.25, 0.35, 0.5, 0.7], [0.1, 0.2, 0.35, 0.6]),
    "nonlinear_scm": ([0.25, 0.4, 0.6, 0.9, 1.3], [0.1, 0.2, 0.35, 0.6]),
    "nb_copula": ([0.15, 0.25, 0.35, 0.5, 0.7], [0.15, 0.3, 0.5, 0.8]),
    "boolean": ([0.3, 0.5, 0.8, 1.2, 2.0], [0.2, 0.5, 1.0]),
}
SKIP = {"fraction_effects_indirect"}


def score(run) -> tuple:
    rep = run.realism()
    rows = [r for r in rep["report"] if r["statistic"] not in SKIP and r.get("value") is not None]
    out = 0.0
    for r in rows:
        lo, hi, v = r["target_low"], r["target_high"], r["value"]
        width = max(hi - lo, 1e-9)
        out += max(0.0, (lo - v) / width, (v - hi) / width)
    return out, {r["statistic"]: round(float(r["value"]), 3) for r in rows}


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=sorted(GRIDS))
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scales, noises = GRIDS[a.generator]
    results = []
    for es in scales:
        for nz in noises:
            vals, stats = [], []
            for seed in range(1, a.seeds + 1):
                A = preset_graph("replogle_like", a.genes, seed=seed)
                run = run_stat_screen(A, a.generator, ModelParams(effect_scale=es, noise=nz), cells_per_condition=40, seed=seed)
                s, st = score(run)
                vals.append(s)
                stats.append(st)
            mean_stats = {k: round(float(np.mean([s[k] for s in stats])), 3) for k in stats[0]}
            results.append({"effect_scale": es, "noise": nz, "score": round(float(np.mean(vals)), 3), "stats": mean_stats})
            print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: r["score"])
    print("BEST", json.dumps(results[:3]))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
