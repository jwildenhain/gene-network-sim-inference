"""
Can missing perturbations be predicted from other contexts, so that algorithms see a complete screen?

    python scripts/impute_partial_screen.py --genes 150 --perturbed 30 --seeds 3 --rewire 0 0.3

Setup (as the VCC cross-context task): three *source* contexts in which every gene was perturbed (the same network with
different basal rates, noisy effect estimates) and a *target* context in which only ``m`` genes were perturbed; with
``--rewire`` a fraction of the target's edges differs from the sources. The target's missing perturbations are filled in
with the mean source response of that gene (the AtlasShift / atlas-mean transfer), giving a complete n x n screen that any
algorithm can be fitted to. Compared per block (A: P->P, B: P->Q, C: Q->P, D: Q->Q; P = perturbed in the target):

* ``partial``  the algorithm on the target's perturbed columns only (adapted where it needs to be);
* ``imputed``  the algorithm on observed + imputed columns (all genes "perturbed");
* ``sources``  the algorithm on the mean source atlas alone (structure transferred, target data unused);
* ``oracle``   the algorithm on a full target screen (what imputation tries to approximate).
"""
import argparse
import collections
import copy
import warnings
from typing import Any, Dict

import numpy as np

from grn_sim.benchmarks.vcc_benchmark import _rewire_edges
from grn_sim.evaluation import screen_benchmark as sb
from grn_sim.evaluation.partial_screen import evaluate_four_blocks, perturbed_genes
from grn_sim.inference import orientation, registry
from grn_sim.inference.partial import PartialAdapter
from grn_sim.perturbseq.calibration import PRESETS
from grn_sim.perturbseq.conditions import simulate_conditions
from grn_sim.perturbseq.pipeline import run_screen

K = 3
METHODS = ["inspre", "mean_difference", "nir", "s2g", "ridge_cv", "cellbox"]


def fit(key: str, X: np.ndarray, U: np.ndarray, partial: bool) -> np.ndarray:
    algo = registry.build(key, registry.BuildContext(k_est=K, use_reg=False))
    if partial and key not in ("inspre", "mean_difference", "nir", "cellbox"):
        S = np.abs(PartialAdapter(algo, key).fit(X, U).A_est)
        return S
    A = np.abs(algo.fit(X, U).A_est)
    return np.abs(orientation.orient_prediction(key, A, orientation.ADDITIVE_GARDNER, target_regulator=False))


def source_atlas(net: Any, noise: float, seed: int) -> tuple:
    sc = simulate_conditions(net, mode="knockdown", seed=seed)
    singles = sc.single_index
    L = np.stack([sc.total_effects[:, singles[g]] for g in range(sc.n_genes)], axis=1)      # column g: response to perturbing g
    L = L + np.random.default_rng(seed + 5).normal(scale=noise, size=L.shape)
    return L


def source_atlas_cells(net: Any, cells: int, seed: int) -> tuple:
    """Source context measured like the target: a full single-cell screen and its ACE estimate (same noise level as the target)."""
    run = run_screen(net, mode="knockdown", cells_per_condition=cells, seed=seed)
    ace = sb.screen_inputs(run, "cells", seed, min_cells=max(10, cells // 3))
    X, U = ace.to_xu()
    full = np.zeros((net.n, net.n))
    for c, g in enumerate(ace.nodes):
        full[:, g] = X[:, c]
    return full


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=150)
    ap.add_argument("--perturbed", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--selection", choices=["random", "effective"], default="random",
                    help="how the perturbed genes are chosen (the published tables used random)")
    ap.add_argument("--cells", type=int, default=40)
    ap.add_argument("--rewire", type=float, nargs="+", default=[0.0, 0.3])
    ap.add_argument("--source-noise", type=float, default=0.05)
    ap.add_argument("--sources", choices=["cells", "truth"], default="cells",
                    help="cells: sources are single-cell screens with the target's cells per perturbation (same noise); "
                         "truth: noise-free effects plus --source-noise (less noisy than the target: flatters the transfer)")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    a = ap.parse_args()
    res: Dict[Any, list] = collections.defaultdict(list)
    for rew in a.rewire:
        for seed in range(1, a.seeds + 1):
            base = PRESETS["replogle_like"].build(a.genes, seed=seed)
            rng = np.random.default_rng(seed + 11)
            sources = []
            for s in range(3):
                net_s = copy.deepcopy(base)
                net_s.params.v = np.clip(net_s.params.v * (1.0 + 0.15 * np.sin(s + 1 + np.arange(a.genes))), 0.2, 4.0)
                sources.append(source_atlas_cells(net_s, a.cells, seed * 10 + s) if a.sources == "cells"
                               else source_atlas(net_s, a.source_noise, seed * 10 + s))
            target = copy.deepcopy(base)
            if rew > 0:
                _rewire_edges(target, rew, rng, shift=1)
            run = run_screen(target, mode="knockdown", cells_per_condition=a.cells, n_perturbed=a.perturbed, perturbed_selection=a.selection, seed=seed)
            full = run_screen(target, mode="knockdown", cells_per_condition=a.cells, seed=seed)
            adj = run.screen.adjacency
            P = perturbed_genes(run.screen)
            ace = sb.screen_inputs(run, "cells", seed, min_cells=max(10, a.cells // 3))
            X, U = ace.to_xu()
            n = a.genes
            atlas = np.mean(sources, axis=0)                               # (n, n) mean source responses, column g = perturbing g
            X_imp = atlas.copy()
            U_imp = np.zeros((n, n))
            own_src = np.diag(atlas).copy()
            U_imp[np.arange(n), np.arange(n)] = own_src
            for c, g in enumerate(ace.nodes):                              # observed target columns replace the transferred ones
                X_imp[:, g] = X[:, c]
                U_imp[g, g] = U[g, c]
            ace_full = sb.screen_inputs(full, "cells", seed, min_cells=max(10, a.cells // 3))
            Xf, Uf = ace_full.to_xu()
            U_src = np.zeros((n, n))
            U_src[np.arange(n), np.arange(n)] = own_src
            variants = {"partial": (X, U, True), "imputed": (X_imp, U_imp, False), "sources": (atlas, U_src, False),
                        "oracle": (Xf, Uf, False)}
            for key in a.methods:
                for vname, (Xv, Uv, part) in variants.items():
                    try:
                        S = fit(key, Xv, Uv, part)
                    except Exception as exc:
                        res[(rew, key, vname)].append({"error": repr(exc)[:60]})
                        continue
                    res[(rew, key, vname)].append(evaluate_four_blocks(adj, P, S))
            print(f"rewire {rew} seed {seed} done", flush=True)
    for rew in a.rewire:
        print(f"\n### {a.genes} genes, {a.perturbed} perturbed in the target, {len(a.methods)} methods, rewire {rew}, {a.seeds} seeds")
        sample = next(v[0] for k, v in res.items() if k[0] == rew and "aupr_A" in v[0])
        print(f"(random AUPR: A {sample['base_A']:.3f}  B {sample['base_B']:.3f}  C {sample['base_C']:.3f}  D {sample['base_D']:.3f})")
        print("| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |\n|---|---|---|---|---|---|---|")
        for key in a.methods:
            for vname in ("partial", "imputed", "sources", "oracle"):
                v = [r for r in res[(rew, key, vname)] if "aupr_A" in r]
                if not v:
                    print(f"| {key} | {vname} | " + (res[(rew, key, vname)][0].get("error", "-")) + " | | | | |")
                    continue
                f = lambda k: f"{np.mean([r[k] for r in v]):.2f}"
                print(f"| {key} | {vname} | {f('aupr_A')} | {f('aupr_B')} | {f('aupr_C')} | {f('aupr_D')} | {f('aupr_all')} |")


if __name__ == "__main__":
    main()
