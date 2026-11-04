"""Leave-perturbation-out prediction of the inspre structure against structure-free baselines (100-gene screens)."""
import argparse
import warnings

import numpy as np

from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation.heldout_prediction import heldout_prediction
from grn_sim.perturbseq.pipeline import build_network, run_screen


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=100)
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for s in range(1, a.seeds + 1):
        run = run_screen(build_network("replogle_like", a.genes, seed=s), mode="knockdown", cells_per_condition=a.cells, seed=s)
        ace = sb.screen_inputs(run, "cells", s)
        truth = sb.screen_inputs(run, "truth", s)
        # truth lfc rows in the node order of ``ace``
        idx = {g: i for i, g in enumerate(truth.nodes)}
        tl = np.vstack([truth.lfc[idx[g]] for g in ace.nodes])
        r = heldout_prediction(ace, tl, seed=s)
        rows.append(r)
        print(s, {k: round(v, 3) for k, v in r.items()}, flush=True)
    print("mean", {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]})


if __name__ == "__main__":
    main()
