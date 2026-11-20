"""
Render docs/results/partial_screen_grid.md from the two partial-screen grids in docs/results/partial_grid/.

    python scripts/render_partial_grid.py            # write the page
    python scripts/render_partial_grid.py --print    # print the generated tables only

Grid 1 (``grid_<n>.json``, scripts/benchmark_partial_screens.py): every registered algorithm, balanced AUPR over the P->P (A) and
P->unperturbed (B) blocks. Grid 2 (``impute_<n>_<pct>pct.md``, scripts/impute_partial_screen.py): six methods on the partial
data, the partial data with the missing perturbations filled in from three source contexts, the sources alone, and the full
oracle screen, scored on four blocks (A P->P, B P->Q, C Q->P, D Q->Q; P = perturbed in the target, Q = not).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "docs" / "results" / "partial_grid"
SIZES = (100, 250)
PCTS = (6, 20, 50, 75)


def grid1() -> Dict[int, Dict[int, Dict[Any, Dict[str, float]]]]:
    out: Dict[int, Dict[int, Dict[Any, Dict[str, float]]]] = {}
    for n in SIZES:
        rows = json.load(open(GRID / f"grid_{n}.json"))
        ms = sorted({r["perturbed"] for r in rows})
        out[n] = {}
        for pct, m in zip(PCTS, ms):
            g: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
            for r in rows:
                if r["perturbed"] == m and r["status"] == "ok":
                    g[(r["method"], r["mode"])].append(r)
            out[n][pct] = {k: {"m": m, "n": len(v), **{x: float(np.mean([r[x] for r in v])) for x in ("aupr_A", "aupr_B", "aupr_balanced")},
                               "sd": float(np.std([r["aupr_balanced"] for r in v]))} for k, v in g.items()}
    return out


def parse_impute(path: Path) -> Dict[float, Dict[Any, List[float]]]:
    res: Dict[float, Dict[Any, List[float]]] = {}
    rew = None
    for line in path.read_text().splitlines():
        m = re.match(r"### .* rewire ([0-9.]+),", line)
        if m:
            rew = float(m.group(1))
            res[rew] = {}
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        if rew is not None and len(c) == 7 and c[1] in ("partial", "imputed", "sources", "oracle"):
            try:
                res[rew][(c[0], c[1])] = [float(x) for x in c[2:]]
            except ValueError:
                pass
    return res


def grid2() -> Dict[int, Dict[int, Dict[float, Dict[Any, List[float]]]]]:
    return {n: {p: parse_impute(GRID / f"impute_{n}_{p}pct.md") for p in PCTS} for n in SIZES}


def tables() -> str:
    g1, g2 = grid1(), grid2()
    o: List[str] = []
    o.append("## Grid 1: every algorithm on a partial screen\n")
    for n in SIZES:
        o.append(f"### {n} genes: best five by balanced AUPR (mean of A and B; mean over 5 seeds, ± sd of the balanced score)\n")
        o.append("| perturbed | rank | method (mode) | A P->P | B P->unperturbed | balanced |\n|---|---|---|---|---|---|")
        for pct in PCTS:
            d = g1[n][pct]
            for i, (k, v) in enumerate(sorted(d.items(), key=lambda kv: -kv[1]["aupr_balanced"])[:5], 1):
                o.append(f"| {pct}% ({v['m']}) | {i} | {k[0]} ({k[1]}) | {v['aupr_A']:.2f} | {v['aupr_B']:.2f} | {v['aupr_balanced']:.2f} ± {v['sd']:.2f} |")
        o.append("")
        o.append(f"### {n} genes: balanced AUPR of every method by fraction perturbed\n")
        o.append("| method (mode) | " + " | ".join(f"{p}%" for p in PCTS) + " |\n|---|" + "---|" * len(PCTS))
        keys = sorted({k for p in PCTS for k in g1[n][p]}, key=lambda k: -np.mean([g1[n][p][k]["aupr_balanced"] for p in PCTS if k in g1[n][p]]))
        for k in keys:
            o.append(f"| {k[0]} ({k[1]}) | " + " | ".join(f"{g1[n][p][k]['aupr_balanced']:.2f}" if k in g1[n][p] else "-" for p in PCTS) + " |")
        o.append("")
    o.append("## Grid 2: filling in the missing perturbations from three source contexts\n")
    o.append("Mean over the six methods (inspre, mean_difference, nir, s2g, ridge_cv, cellbox) and 3 seeds. "
             "'observed' = mean of blocks A and B (regulator was perturbed in the target); 'hidden' = mean of C and D (regulator was never "
             "perturbed in the target); 'all' = all pairs.\n")
    for n in SIZES:
        for rew in (0.0, 0.3):
            o.append(f"### {n} genes, {int(rew * 100)}% of the target's edges rewired relative to the sources\n")
            o.append("| perturbed | data | observed (A,B) | hidden (C,D) | all pairs |\n|---|---|---|---|---|")
            for pct in PCTS:
                d = g2[n][pct].get(rew, {})
                for mode in ("partial", "imputed", "sources", "oracle"):
                    v = [x for (meth, md), x in d.items() if md == mode]
                    if v:
                        a = np.mean(v, axis=0)
                        o.append(f"| {pct}% | {mode} | {(a[0] + a[1]) / 2:.2f} | {(a[2] + a[3]) / 2:.2f} | {a[4]:.2f} |")
            o.append("")
    o.append("### Imputed minus partial, all-pairs AUPR, by method (mean over fractions)\n")
    o.append("| genes | rewired | " + " | ".join(["inspre", "mean_difference", "nir", "s2g", "ridge_cv", "cellbox"]) + " |\n|---|---|---|---|---|---|---|---|")
    for n in SIZES:
        for rew in (0.0, 0.3):
            cells = []
            for meth in ["inspre", "mean_difference", "nir", "s2g", "ridge_cv", "cellbox"]:
                dif = [g2[n][p][rew][(meth, "imputed")][4] - g2[n][p][rew][(meth, "partial")][4] for p in PCTS
                       if (meth, "imputed") in g2[n][p].get(rew, {})]
                cells.append(f"{np.mean(dif):+.2f}")
            o.append(f"| {n} | {int(rew * 100)}% | " + " | ".join(cells) + " |")
    o.append("")
    return "\n".join(o)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    t = tables()
    if a.print:
        print(t)
        return
    head = (ROOT / "docs" / "results" / "partial_screen_grid_header.md").read_text()
    (ROOT / "docs" / "results" / "partial_screen_grid.md").write_text(head.rstrip() + "\n\n" + t)
    print("wrote docs/results/partial_screen_grid.md")


if __name__ == "__main__":
    main()
