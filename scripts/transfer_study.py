#!/usr/bin/env python3
"""
Paired multi-network study of the cross-context transfer predictors (grn_sim.benchmarks.cross_context).

Each random network gives three source atlases and a rewired target context (the Virtual Cell Challenge tab's setting).
The predictors see the sources and the *seen* target perturbations only; the held-out target perturbations are used for
scoring. Scores are Pearson r between the predicted and the true response profile, averaged over the held-out perturbations
(and, leave-one-out, over the seen ones), once over all genes and once over the off-target genes only (the perturbed gene
itself removed, which matters for knockdown data). Every predictor is compared with the plain source-atlas average on the
*same* network (paired difference, standard error, number of wins), because single networks are dominated by noise.

    python scripts/transfer_study.py                              # activation data, 16 genes, 24 networks
    python scripts/transfer_study.py --mode knockdown --rewire 0.3
    python scripts/transfer_study.py --seed-offset 100 --networks 12 --json tuning.json
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grn_sim.benchmarks import vcc_benchmark as vcc  # noqa: E402
from grn_sim.benchmarks.cross_context import TRANSFER_PREDICTORS, TransferData, predict_matrix  # noqa: E402
from grn_sim.benchmarks.hill_response import HillResponseModel, mean_profile_baseline, pearson  # noqa: E402
from grn_sim.simulator.experiment import PerturbationSimulator  # noqa: E402

BASELINE = "atlas_mean"


def _correlations(R: np.ndarray, X: np.ndarray, columns: Sequence[int]) -> Tuple[float, float]:
    """Mean Pearson r over ``columns``: all genes, and without the perturbed gene itself."""
    every = np.mean([pearson(R[:, j], X[:, j]) for j in columns])
    off_target = np.mean([pearson(np.delete(R[:, j], j), np.delete(X[:, j], j)) for j in columns])
    return float(every), float(off_target)


def evaluate_network(mode: str, genes: int, seed: int, rewire: float, held_out: int) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """{predictor: {"held_out": (r, off-target r), "seen": (r, off-target r)}} for one simulated network."""
    sim = PerturbationSimulator(1.0, 0.01, perturbation_mode=mode, knockdown_efficiency=vcc.KNOCKDOWN_EFFICIENCY)
    sources, target = vcc._simulate_contexts(genes, 1.8, seed, rewire, sim)
    held = list(range(genes - held_out, genes))
    seen = [j for j in range(genes) if j not in held]
    data = TransferData(
        sources=tuple(exp.X for _, exp in sources), target=target.X, design=target.U,
        control=target.reference_steady_state, seen=tuple(seen),
    )

    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for predictor in TRANSFER_PREDICTORS:
        R = predict_matrix(predictor, data)
        out[predictor.key] = {"held_out": _correlations(R, target.X, held), "seen": _correlations(R, target.X, seen)}

    # no cross-context information at all: the mean response of the seen target columns (leave-one-out for the seen ones)
    null = mean_profile_baseline(target.X, seen)
    out["null"] = {"held_out": _correlations(null, target.X, held), "seen": _correlations(null, target.X, seen)}

    # ceiling: the target's true regulators with Hill kinetics refit on the seen target columns (held-out only)
    x_abs = (target.X + 1.0) * target.reference_steady_state[:, None]
    oracle = HillResponseModel(
        target.true_matrix.astype(int), x_abs[:, seen], target.U[:, seen], target.reference_steady_state, mode=mode
    ).response_matrix(target.U)
    out["oracle_hill"] = {"held_out": _correlations(oracle, target.X, held)}
    return out


def run_study(
    mode: str = "input", genes: int = 16, held_out: int = 4, rewire: float = 0.15, networks: int = 24,
    seed_offset: int = 0, progress: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """{"held_out": {predictor: {"all": [...], "off_target": [...]}}, "seen": {...}}, one value per network."""
    warnings.filterwarnings("ignore")
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = {"held_out": {}, "seen": {}}
    for k in range(1, networks + 1):
        for name, parts in evaluate_network(mode, genes, seed_offset + k, rewire, held_out).items():
            for split, (every, off_target) in parts.items():
                cell = result[split].setdefault(name, {"all": [], "off_target": []})
                cell["all"].append(every)
                cell["off_target"].append(off_target)
        if progress:
            print(f"  network {k}/{networks}", file=sys.stderr, flush=True)
    return result


def summarise(values: Dict[str, List[float]], baseline: str = BASELINE) -> Dict[str, Dict[str, float]]:
    """Per predictor: mean r, paired difference to ``baseline`` with its standard error, and the number of wins."""
    base = np.asarray(values[baseline])
    summary = {}
    for name, v in values.items():
        arr = np.asarray(v)
        d = arr - base
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        summary[name] = {"mean": float(arr.mean()), "minus_baseline": float(d.mean()), "se": se, "wins": int((d > 0).sum()), "n": len(d)}
    return summary


def format_report(result: Dict[str, Dict[str, Dict[str, List[float]]]], title: str) -> str:
    lines = [title, f"(paired differences are to '{BASELINE}', the plain average of the source atlases)"]
    for split, heading in (("held_out", "HELD-OUT target perturbations"), ("seen", "SEEN target perturbations (leave-one-out)")):
        for key, view in (("all", "all genes"), ("off_target", "off-target genes only")):
            values = {name: parts[key] for name, parts in result[split].items()}
            lines.append(f"-- {heading}, {view}: Pearson r")
            for name, s in summarise(values).items():
                lines.append(f"  {name:17s} mean {s['mean']:+.3f}   minus {BASELINE} {s['minus_baseline']:+.3f} +/- {s['se']:.3f}   wins {s['wins']}/{s['n']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the top of this file for details.")
    parser.add_argument("--mode", choices=vcc.PERTURBATION_MODES, default="input")
    parser.add_argument("--genes", type=int, default=16)
    parser.add_argument("--held-out", type=int, default=4, help="target perturbations withheld from every fit")
    parser.add_argument("--rewire", type=float, default=0.15, help="fraction of edges moved in the target context")
    parser.add_argument("--networks", type=int, default=24, help="number of random networks (paired design)")
    parser.add_argument("--seed-offset", type=int, default=0, help="networks use seeds offset+1 ... offset+networks")
    parser.add_argument("--json", help="also write the per-network scores to this file")
    args = parser.parse_args(argv)
    if not 1 <= args.held_out < args.genes - 1:
        parser.error("--held-out must be between 1 and genes - 2")
    if not 0.0 <= args.rewire <= 1.0:
        parser.error("--rewire must be in [0, 1]")

    started = time.time()
    result = run_study(args.mode, args.genes, args.held_out, args.rewire, args.networks, args.seed_offset, progress=sys.stderr.isatty())
    print(format_report(result, f"mode={args.mode} genes={args.genes} held-out={args.held_out} rewire={args.rewire} "
                                f"networks={args.networks} seeds={args.seed_offset + 1}-{args.seed_offset + args.networks} "
                                f"({time.time() - started:.0f}s)"))
    if args.json:
        Path(args.json).write_text(json.dumps({"settings": vars(args), "scores": result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
