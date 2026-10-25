#!/usr/bin/env python3
"""
Paired multi-network study of held-out perturbation prediction (the research behind the Virtual Cell Challenge tab).

For every random network the same experiment is scored with several predictors and the *paired* difference to a
no-structure null is reported, because single-network comparisons are dominated by noise (the null alone ranges from
about 0 to 0.4 between networks). Predictors:

  null        mean response profile of the seen experiments (what you get without any regulatory knowledge)
  self_only   knockdown data only: -efficiency at the perturbed gene and 0 elsewhere, taken from the design
  oracle      the TRUE regulators (signs included) with Hill kinetics refit on the seen experiments: upper bound
  hill_search model-matched forward selection of regulators (hill_response.hill_forward_selection)
  <algorithm> that algorithm's top regulators per target, Hill kinetics refit (hill_response.infer_signed_structure)

Every fit sees only the seen experiments; held-out columns are used for scoring only. Scores are Pearson r between the
predicted and the true response profile, averaged over the held-out perturbations, once over all genes and once over the
off-target genes only (the perturbed gene itself removed, which matters for knockdown data).

    python scripts/holdout_study.py                                       # activation data, 16 genes, 24 networks
    python scripts/holdout_study.py --mode knockdown                      # CRISPRi-style knockdown data
    python scripts/holdout_study.py --noise 0 --networks 12 --json out.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grn_sim.benchmarks import vcc_benchmark as vcc  # noqa: E402
from grn_sim.benchmarks.hill_response import (  # noqa: E402
    HillResponseModel,
    hill_forward_selection,
    infer_signed_structure,
    mean_profile_baseline,
    pearson,
)
from grn_sim.core.network import GeneNetwork  # noqa: E402
from grn_sim.core.topology import TopologyType  # noqa: E402
from grn_sim.simulator.experiment import PerturbationSimulator  # noqa: E402

DEFAULT_ALGORITHMS = ("g2s", "genie3", "nir", "vote")


def make_experiment(mode: str, genes: int, seed: int, noise: float, efficiency: float = vcc.KNOCKDOWN_EFFICIENCY):
    """The benchmark's target context: a scale-free network with 15% of its edges moved."""
    net = GeneNetwork.create(n=genes, k_av=1.8, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)
    vcc._rewire_edges(net, 0.15, np.random.default_rng(seed + 999), shift=1)
    sim = PerturbationSimulator(1.0, noise, perturbation_mode=mode, knockdown_efficiency=efficiency)
    return sim.run(net, seed=seed + 999)


def run_study(
    mode: str = "input",
    genes: int = 16,
    held_out: int = 4,
    noise: float = 0.01,
    networks: int = 24,
    algorithms=DEFAULT_ALGORITHMS,
    progress: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """Returns {"all": {predictor: [r per network]}, "off_target": {...}, "structure": {"precision": [...], "recall": [...]}}.

    ``structure`` is the edge precision / recall of the hill_search regulators against the true network."""
    warnings.filterwarnings("ignore")
    names = ["null"] + (["self_only"] if mode == "knockdown" else []) + ["oracle", "hill_search"] + list(algorithms)
    out = {"all": {k: [] for k in names}, "off_target": {k: [] for k in names}, "structure": {"precision": [], "recall": []}}
    held = list(range(genes - held_out, genes))
    seen = [j for j in range(genes) if j not in held]

    for seed in range(1, networks + 1):
        exp = make_experiment(mode, genes, seed, noise)
        x_abs = (exp.X + 1.0) * exp.reference_steady_state[:, None]
        x_seen, u_seen, x_ctrl = x_abs[:, seen], exp.U[:, seen], exp.reference_steady_state

        def record(name: str, R: np.ndarray) -> None:
            out["all"][name].append(float(np.mean([pearson(R[:, j], exp.X[:, j]) for j in held])))
            out["off_target"][name].append(float(np.mean([
                pearson(np.delete(R[:, j], j), np.delete(exp.X[:, j], j)) for j in held
            ])))

        def hill(signed: np.ndarray) -> np.ndarray:
            return HillResponseModel(signed, x_seen, u_seen, x_ctrl, mode=mode).response_matrix(exp.U)

        record("null", mean_profile_baseline(exp.X, seen))
        if mode == "knockdown":
            record("self_only", np.diag(np.diag(exp.U)))
        record("oracle", hill(exp.true_matrix.astype(int)))
        searched = hill_forward_selection(x_seen, u_seen, x_ctrl, mode=mode)
        record("hill_search", hill(searched))
        hits = int(((searched != 0) & (exp.true_matrix != 0)).sum())
        out["structure"]["precision"].append(hits / max(1, int((searched != 0).sum())))
        out["structure"]["recall"].append(hits / max(1, int((exp.true_matrix != 0).sum())))

        X_train, U_train = exp.X.copy(), exp.U.copy()
        X_train[:, held] = 0.0
        U_train[:, held] = 0.0
        for name in algorithms:
            res = vcc._build_runners(2)[name].fit(X_train, vcc.FIT_INPUT_SIGN.get(name, 1.0) * U_train)
            signed, _ = infer_signed_structure(res.A_est, res.adjacency_pred, X_train, U_train, q=vcc.HILL_STRUCTURE_Q)
            record(name, hill(signed))
        if progress:
            print(f"  network {seed}/{networks}", file=sys.stderr, flush=True)
    return out


def summarise(scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Per predictor: mean r, paired difference to the null with its standard error, and the number of wins."""
    null = np.asarray(scores["null"])
    summary = {}
    for name, values in scores.items():
        v = np.asarray(values)
        d = v - null
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        summary[name] = {"mean": float(v.mean()), "minus_null": float(d.mean()), "se": se, "wins": int((d > 0).sum()), "n": len(d)}
    return summary


def format_report(result: Dict[str, Dict[str, List[float]]], title: str) -> str:
    lines = [title]
    if "structure" in result:
        lines.append(f"-- hill_search regulators vs the true network: precision {np.mean(result['structure']['precision']):.2f}, "
                     f"recall {np.mean(result['structure']['recall']):.2f}")
    for key, heading in (("all", "ALL GENES"), ("off_target", "OFF-TARGET GENES ONLY (perturbed gene left out)")):
        lines.append(f"-- {heading}: held-out Pearson r, paired with the null")
        for name, s in summarise(result[key]).items():
            lines.append(f"  {name:12s} mean {s['mean']:+.3f}   minus null {s['minus_null']:+.3f} +/- {s['se']:.3f}   wins {s['wins']}/{s['n']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the top of this file for details.")
    parser.add_argument("--mode", choices=vcc.PERTURBATION_MODES, default="input", help="perturbation generator (default: input)")
    parser.add_argument("--genes", type=int, default=16)
    parser.add_argument("--held-out", type=int, default=4, help="perturbations withheld from every fit")
    parser.add_argument("--noise", type=float, default=0.01, help="relative measurement noise")
    parser.add_argument("--networks", type=int, default=24, help="number of random networks (paired design)")
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS), help="comma separated algorithm keys")
    parser.add_argument("--json", help="also write the per-network scores to this file")
    args = parser.parse_args(argv)

    algorithms = [a for a in args.algorithms.split(",") if a]
    unknown = [a for a in algorithms if a not in vcc._build_runners(2) or a == "atlasshift"]
    if unknown:
        parser.error(f"unknown algorithm(s): {', '.join(unknown)}")
    if not 1 <= args.held_out < args.genes - 1:
        parser.error("--held-out must be between 1 and genes - 2")

    started = time.time()
    result = run_study(args.mode, args.genes, args.held_out, args.noise, args.networks, algorithms, progress=sys.stderr.isatty())
    title = (f"mode={args.mode} genes={args.genes} held-out={args.held_out} noise={args.noise} networks={args.networks} "
             f"({time.time() - started:.0f}s)")
    print(format_report(result, title))
    if args.json:
        Path(args.json).write_text(json.dumps({"settings": vars(args), "scores": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
