"""
Calibration sweep of the Perturb-seq generator against genome-scale Perturb-seq statistics.

For every combination of network-regime and cell parameters, simulates a knockdown screen (stage 1), cells (stage 2)
and their measurement (stage 3), computes ground-truth and observed realism statistics (stage 4) and scores them
against grn_sim.perturbseq.realism.TARGETS (0 = every statistic inside its target range).

Example (500 genes, a small grid):
    PYTHONPATH=src python scripts/calibrate_perturbseq.py --genes 500 --k-av 2.5 3.0 --guide-efficiency 0.92 \
        --out calibration_500.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace

from grn_sim.perturbseq import MeasurementConfig, population_to_anndata, simulate_cells, simulate_conditions
from grn_sim.perturbseq.calibration import PRESETS
from grn_sim.perturbseq.realism import format_report, observed_statistics, realism_report, realism_score, truth_statistics


def run_one(n: int, preset_kwargs: dict, guide_efficiency: float, cells: int, controls: int, seed: int) -> dict:
    preset = replace(PRESETS["replogle_like"], **preset_kwargs)
    t0 = time.time()
    net = preset.build(n, seed=seed)
    screen = simulate_conditions(net, mode="knockdown", seed=seed)
    pop = simulate_cells(net, screen, cells_per_condition=cells, n_control_cells=controls,
                         config=preset.cell_config(guide_efficiency_mean=guide_efficiency), seed=seed + 1)
    adata = population_to_anndata(pop, screen, seed=seed + 2, measurement=MeasurementConfig())
    t1 = time.time()
    stats = {**truth_statistics(screen), **observed_statistics(adata, seed=seed),
             "lna_clipped_values": float(pop.meta.get("lna_clipped_values", 0))}
    rows = realism_report(stats)
    return {
        "params": {**preset_kwargs, "guide_efficiency_mean": guide_efficiency, "genes": n, "seed": seed},
        "score": realism_score(rows),
        "report": rows,
        "statistics": stats,
        "seconds": {"simulate": round(t1 - t0, 1), "statistics": round(time.time() - t1, 1)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genes", type=int, default=500)
    ap.add_argument("--k-av", type=float, nargs="+", default=[2.5])
    ap.add_argument("--ratio", type=float, nargs="+", default=[0.8])
    ap.add_argument("--p-within", type=float, nargs="+", default=[0.85])
    ap.add_argument("--attachment-offset", type=float, nargs="+", default=[0.5])
    ap.add_argument("--spread", type=float, nargs="+", default=[0.5])
    ap.add_argument("--hill", nargs="+", default=["1,3"], help="Hill coefficient ranges as low,high")
    ap.add_argument("--guide-efficiency", type=float, nargs="+", default=[0.92])
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--controls", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="write all results as JSON")
    args = ap.parse_args()

    results = []
    hills = [tuple(float(v) for v in h.split(",")) for h in args.hill]
    grid = itertools.product(args.k_av, args.ratio, args.p_within, args.attachment_offset, args.spread, hills,
                             args.guide_efficiency)
    for k_av, ratio, p_within, offset, spread, hill, eff in grid:
        kwargs = {"k_av": k_av, "ratio": ratio, "p_within": p_within, "attachment_offset": offset, "spread": spread,
                  "hill_range": hill}
        res = run_one(args.genes, kwargs, eff, args.cells, args.controls, args.seed)
        results.append(res)
        print(f"\n=== {res['params']}  score {res['score']:.2f}  ({res['seconds']})", flush=True)
        print(format_report(res["report"]), flush=True)
        print(f"LNA values clipped: {int(res['statistics']['lna_clipped_values'])}", flush=True)
    best = min(results, key=lambda r: r["score"])
    print(f"\nbest: {best['params']}  score {best['score']:.2f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
