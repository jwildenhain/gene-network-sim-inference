"""
Findings of the Virtual Cell screen benchmark, computed from the stored simulations (never typed by hand).

Every comparison is *matched*: the settings differ in one factor only and the simulations are paired by (data generator,
seed), so a difference is the effect of that factor on the same networks. Each difference comes with a two-sided paired
Wilcoxon p-value; "no clear difference" is stated when p >= 0.05. ``render`` writes the markdown of
``docs/results/vcc_screen_findings.md``; ``tests/test_vcc_findings.py`` checks the file still matches the data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from grn_sim.evaluation import vcc_benchmark as vb

BASE = {"pathway": "replogle_like", "n_genes": 100, "mode": "knockdown", "pairs": 0, "selection": "full", "noise": "medium"}
ALPHA = 0.05


def _index(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple, Dict[str, float]]:
    """(setting key, seed) -> {algorithm: AUPR, "__base__": the chance AUPR (share of true edges) of that simulation}."""
    out: Dict[Tuple, Dict[str, float]] = {}
    for r in rows:
        if r.get("status", "ok") != "ok" or r.get("aupr_direct") is None:
            continue
        d = out.setdefault((vb.Setting.from_row(r), int(r["seed"])), {})
        d[r["algorithm"]] = float(r["aupr_direct"])
        if r.get("base_direct") is not None:
            d["__base__"] = float(r["base_direct"])
    return out


def _p_two_sided(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def matched(rows: Sequence[Dict[str, Any]], factor: str, levels: Sequence[Any], base: Dict[str, Any],
            algorithms: Sequence[str]) -> Dict[str, Any]:
    """Mean AUPR per level and the paired difference of each level against the first, per algorithm."""
    idx = _index(rows)
    want = {k: v for k, v in base.items() if k != factor}
    by_level: Dict[Any, Dict[Tuple[str, int], Dict[str, float]]] = {lv: {} for lv in levels}
    for (s, seed), algs in idx.items():
        d = s.to_dict()
        if any(str(d[k]) != str(v) for k, v in want.items()):
            continue
        for lv in levels:
            if str(d[factor]) == str(lv):
                # pair by (generator, seed); across generators by seed alone (the same graph for every generator)
                by_level[lv][(seed,) if factor == "generator" else (s.generator, seed)] = algs
    ref = levels[0]
    chance = {str(lv): (float(np.mean([v["__base__"] for v in by_level[lv].values() if "__base__" in v]))
                        if any("__base__" in v for v in by_level[lv].values()) else None) for lv in levels}
    out: Dict[str, Any] = {"factor": factor, "levels": list(levels), "base": want, "chance": chance, "algorithms": {}}
    for a in algorithms:
        row: Dict[str, Any] = {}
        for lv in levels:
            keys = sorted(k for k in by_level[lv] if a in by_level[lv][k])
            row[str(lv)] = {"mean": float(np.mean([by_level[lv][k][a] for k in keys])) if keys else None, "n": len(keys)}
            if lv != ref:
                common = [k for k in keys if a in by_level[ref].get(k, {})]
                diff = np.array([by_level[lv][k][a] - by_level[ref][k][a] for k in common])
                row[str(lv)].update(diff=float(diff.mean()) if diff.size else None, p=_p_two_sided(diff), n_pairs=len(common))
        out["algorithms"][a] = row
    return out


def compute(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    overall = vb.aggregate(rows, n_boot=400)
    algs = [a["algorithm"] for a in overall["algorithms"] if a["verdict"] not in ("reference", "control")]
    gens = [g for g in vb.GENERATORS if any(r.get("generator") == g for r in rows)]
    by_gen = {g: {a["algorithm"]: a for a in vb.aggregate(rows, {**{k: str(v) for k, v in BASE.items()}, "generator": g},
                                                          n_boot=0)["algorithms"]} for g in gens}
    mech_vs = matched(rows, "generator", ["mechanistic"] + [g for g in gens if g != "mechanistic"], BASE, algs)
    return {
        "n_simulations": overall["n_simulations"], "n_settings": overall["n_settings"],
        "overall": overall["algorithms"],
        "dropped": dict(vb.DROPPED),
        "by_generator": {g: {a: {"mean": v["mean"], "verdict": v["verdict"], "n": v["n"]} for a, v in d.items()}
                         for g, d in by_gen.items()},
        "generator_contrast": mech_vs,
        "noise": matched(rows, "noise", ["low", "medium", "high"], BASE, algs),
        "mode": matched(rows, "mode", ["knockdown", "knockout", "input"], BASE, algs),
        "size": matched(rows, "n_genes", [100, 50, 200], BASE, algs),
        "doubles_100": matched(rows, "pairs", [0, 50], BASE, algs),
        "doubles_200": matched(rows, "pairs", [0, 100], {**BASE, "n_genes": 200}, algs),
        "selection": matched(rows, "selection", ["full", "effective", "random"], BASE, algs),
        "selection_pairs": matched(rows, "pairs", [0, 50], {**BASE, "selection": "effective"}, algs),
        "pathway": matched(rows, "pathway", list(vb.PATHWAYS), BASE, algs),
    }


# -- rendering -------------------------------------------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    return "–" if v is None else f"{v:.3f}"


def _contrast_table(m: Dict[str, Any], labels: Dict[str, str]) -> List[str]:
    levels = [str(x) for x in m["levels"]]
    ref = levels[0]
    head = f"| algorithm | {labels.get(ref, ref)} |" + "".join(f" {labels.get(lv, lv)} | Δ vs {labels.get(ref, ref)} (p) |" for lv in levels[1:])
    lines = [head, "|---|---|" + "---|---|" * (len(levels) - 1)]
    for a, row in m["algorithms"].items():
        cells = [f"| {a} | {_fmt(row[ref]['mean'])} |"]
        for lv in levels[1:]:
            c = row[lv]
            if c.get("diff") is None:
                cells.append(" – | – |")
            else:
                mark = "" if c["p"] < ALPHA else " ns"
                cells.append(f" {_fmt(c['mean'])} | {c['diff']:+.3f} (p={c['p']:.2g}{mark}) |")
        lines.append("".join(cells))
    ch = m.get("chance", {})
    lines.append("| chance (share of true edges) | " + _fmt(ch.get(ref)) + " |" +
                 "".join(f" {_fmt(ch.get(lv))} | |" for lv in levels[1:]))
    n = {lv: max((row[lv].get("n_pairs") or row[lv]["n"]) for row in m["algorithms"].values()) for lv in levels}
    lines.append("")
    lines.append("Paired simulations per comparison: " + ", ".join(f"{labels.get(lv, lv)}: {n[lv]}" for lv in levels[1:]) +
                 f". Fixed: {', '.join(f'{k}={v}' for k, v in m['base'].items())}. ns = p >= {ALPHA} (no clear difference).")
    return lines


def _significant(m: Dict[str, Any], level: str, sign: int) -> List[str]:
    out = []
    for a, row in m["algorithms"].items():
        c = row[level]
        if c.get("diff") is not None and c["p"] < ALPHA and np.sign(c["diff"]) == sign:
            out.append(f"{a} ({c['diff']:+.3f})")
    return out


def render(f: Dict[str, Any]) -> str:
    over = f["overall"]
    real = [a for a in over if a["verdict"] not in ("reference", "control")]
    good = [a["algorithm"] for a in real if a["verdict"] == "better"]
    marginal = [a["algorithm"] for a in real if a["verdict"] == "marginal"]
    bad = [a["algorithm"] for a in real if a["verdict"] in ("not_better", "worse")]
    top = real[0]
    tied = [a["algorithm"] for a in real[1:] if a["rank_low"] <= top["rank_high"]]
    L = ["# Virtual Cell screen benchmark: findings", "",
         "Generated by `scripts/run_vcc_benchmark.py --report` from the stored simulations "
         f"({f['n_simulations']} simulations, {f['n_settings']} settings); every number below is computed from them and "
         "`tests/test_vcc_findings.py` fails if this file and the data disagree. Score: direct-edge AUPR over the perturbed genes. "
         "Comparisons are matched (one factor changed, simulations paired by data generator and seed) with two-sided paired "
         "Wilcoxon p-values (across data generators: paired by seed, the same graph for every generator).", "",
         "## Which algorithms work", "",
         f"* Beat both the random ranking and the hubness-only control over the whole corpus: {len(good)} of {len(real)} "
         f"({', '.join(good)}).",
         f"* Only at the hubness level (marginal): {', '.join(marginal) if marginal else 'none'}.",
         f"* Not better than random: {', '.join(bad) if bad else 'none'}.",
         "* Dropped before the benchmark (they fail the admission test on noise-free data, tests/test_vcc_algorithm_admission.py): "
         + "; ".join(f"{k}: {v}" for k, v in f["dropped"].items()) + ".",
         f"* Rank 1: {top['algorithm']} (AUPR {top['mean']:.3f}, 95% CI {top['ci_low']:.3f}-{top['ci_high']:.3f}, rank interval "
         f"{top['rank_low']}-{top['rank_high']})" + (f"; not separable from it: {', '.join(tied)}." if tied else "; separable from all others."),
         "", "| rank (95%) | algorithm | AUPR | 95% CI | verdict |", "|---|---|---|---|---|"]
    for a in over:
        L.append(f"| {a['rank']} ({a['rank_low']}-{a['rank_high']}) | {a['algorithm']} | {a['mean']:.3f} | "
                 f"{a['ci_low']:.3f}-{a['ci_high']:.3f} | {a['verdict']} |")
    gl = {g: vb.GENERATORS[g] for g in vb.GENERATORS}
    L += ["", "## Mechanistic versus statistical data", "",
          "Baseline setting (modular graph, 100 genes, CRISPRi, medium noise, full screen), the same graphs and seeds for every "
          "generator; Δ = statistical minus mechanistic.", ""]
    L += _contrast_table(f["generator_contrast"], gl)
    for name, _factor, labels, title in (("noise", "noise", {}, "Noise level"),
                                     ("mode", "mode", vb.MODES, "Perturbation type"),
                                     ("size", "n_genes", {}, "Network size (genes)"),
                                     ("doubles_100", "pairs", {"0": "no doubles", "50": "50 doubles"}, "Double knockouts, 100 genes"),
                                     ("doubles_200", "pairs", {"0": "no doubles", "100": "100 doubles"}, "Double knockouts, 200 genes"),
                                     ("selection", "selection", vb.SELECTIONS, "Partial screens (50% of genes perturbed)"),
                                     ("selection_pairs", "pairs", {"0": "no doubles", "50": "50 doubles"},
                                      "Double knockouts in a partial screen (effective genes)"),
                                     ("pathway", "pathway", vb.PATHWAYS, "Pathway architecture")):
        m = f[name]
        L += ["", f"## {title}", ""]
        if name == "selection":
            L += ["AUPR is computed over the rows of the perturbed genes, so its chance level (last row) changes with the "
                  "selection: effective genes have more true targets. Compare differences against the change in chance.", ""]
        if name in ("doubles_100", "doubles_200", "selection_pairs"):
            L += ["About 30% of double-perturbed cells lose one of their two guides in guide calling and are then analysed as "
                  "single perturbations of the other gene (a real artefact of combinatorial screens), which dilutes the "
                  "single-perturbation effects every method uses; tests/test_perturbation_dependencies.py shows that with the "
                  "true labels the double knockouts do not change the single-perturbation data.", ""]
        L += _contrast_table(m, {str(k): v for k, v in labels.items()})
        for lv in [str(x) for x in m["levels"][1:]]:
            up, down = _significant(m, lv, 1), _significant(m, lv, -1)
            name_lv = labels.get(lv, lv) if isinstance(labels, dict) else lv
            L.append("")
            L.append(f"{name_lv}: significantly higher for {', '.join(up) if up else 'none'}; "
                     f"significantly lower for {', '.join(down) if down else 'none'}.")
    return "\n".join(L) + "\n"
