"""
Checks the calibrated generators on seeds not used for calibration: realism statistics and how hard the data are (AUPR of
mean difference and S2G) at each noise level, next to the mechanistic generator.

    python scripts/verify_statgen_noise.py [--genes 100 --seeds 3 --generators linear_sem,nb_copula] [--out file.json]
"""
import argparse
import json
import warnings

import numpy as np

from grn_sim.evaluation import vcc_benchmark as vb
from grn_sim.evaluation import screen_benchmark as sb

KEEP = ("on_target_knockdown", "fraction_with_phenotype", "affected_pair_density", "top5pct_perturbation_effect_share",
        "control_false_positive_rate")


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--generators", default=",".join(vb.GENERATORS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib = {x.name: x for x in vb.algorithm_library()}
    out = []
    for gen in a.generators.split(","):
        for noise in vb.NOISE_LEVELS:
            stats, md, s2g, score = [], [], [], []
            for seed in range(501, 501 + a.seeds):
                run = vb.simulate_screen(vb.Setting(generator=gen, noise=noise, n_genes=a.genes), seed)
                rep = run.realism()
                stats.append({k: rep["statistics"].get(k) for k in KEEP})
                rows = [r for r in rep["report"] if r["statistic"] != "fraction_effects_indirect"]
                score.append(sum(max(0.0, (r["target_low"] - r["value"]) / (r["target_high"] - r["target_low"]),
                                     (r["value"] - r["target_high"]) / (r["target_high"] - r["target_low"]))
                                 for r in rows if r.get("value") is not None))
                ace = sb.screen_inputs(run, "cells", seed)
                X, U = ace.to_xu()
                rng = np.random.default_rng(seed)
                md.append(sb.evaluate_scores(run.screen, lib["mean_difference"].score(ace, X, U, rng))["aupr_direct"])
                s2g.append(sb.evaluate_scores(run.screen, lib["s2g"].score(ace, X, U, rng))["aupr_direct"])
            row = {"generator": gen, "noise": noise, "realism_score": round(float(np.mean(score)), 2),
                   "aupr_mean_difference": round(float(np.mean(md)), 3), "aupr_s2g": round(float(np.mean(s2g)), 3),
                   **{k: round(float(np.mean([s[k] for s in stats if s[k] is not None])), 3) for k in KEEP}}
            out.append(row)
            print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
