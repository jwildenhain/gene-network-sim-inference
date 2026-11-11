"""
``docs/results/statgen_calibration.md``, rendered from the calibrated parameters (``grn_sim.statgen.calibration``) and the
verification results (``docs/results/statgen_noise_verification.json``, written by ``scripts/verify_statgen_noise.py``).
Only the method text is fixed; every number comes from code or data, and ``tests/test_statgen_report.py`` checks the file.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from grn_sim.perturbseq.realism import TARGETS
from grn_sim.statgen.calibration import CALIBRATED

COLUMNS = ("on_target_knockdown", "fraction_with_phenotype", "affected_pair_density", "top5pct_perturbation_effect_share",
           "control_false_positive_rate")
ORDER = ("mechanistic", "linear_sem", "nonlinear_scm", "nb_copula", "boolean")
UNITS = {"linear_sem": "log-sd of the propagated noise", "nonlinear_scm": "log-sd", "nb_copula": "sqrt of the NB dispersion",
         "boolean": "flip probability x 10"}

METHOD = """\
The Virtual Cell tab generates experimental data from statistical models (`grn_sim.statgen`) instead of the panel-1 Hill
kinetics; mechanistic data from the panel-1 engine stay available as one more generator.

## Method

1. **Grid search** (`scripts/calibrate_statgen.py`): for every statistical generator, effect scale x noise on 100-gene modular
   graphs, CRISPRi screens with 40 cells per perturbation and the measurement layer, scored by how far the observed screen
   statistics lie outside the genome-scale Perturb-seq ranges of `grn_sim.perturbseq.realism.TARGETS` (0 = all inside; the
   genome-scale share of indirect effects is left out: no 100-gene network can reach it).
2. **Choice**: where several grid points met the targets, the middle of that region, not the edge of the grid.
3. **Noise levels**: low / medium / high = 0.5 / 1 / 2 x the calibrated cell noise (effect sizes unchanged), so the levels
   differ in signal-to-noise only. For the mechanistic generator the same factors scale the extrinsic noise of its cell model.
4. **Verification** (`scripts/verify_statgen_noise.py`) on seeds never used for calibration (table below).
5. **Perturbation dependencies** are tested separately (`tests/test_perturbation_dependencies.py`): a perturbation moves
   exactly the genes downstream of it, double perturbations interact only downstream of both partners, and the cell-level
   analysis finds dependencies on at most 5% of independent gene pairs.

Model changes made during calibration: the Boolean model became a stochastic logistic rule over weighted inputs (hard threshold
rules gave far too many affected pairs and a weak knockdown) with common random numbers across conditions (so genes that do not
depend on a perturbation show exactly no effect); edge weights got a heavier tail (weight spread, the log-sd of |edge weight|, tuned per
generator in a second round together with effect scale and noise); the nonlinear model's noise was lowered; the mechanistic screens use the preset's 0.92 guide efficiency; effects are estimated with median-ratio size factors
(total-count normalisation made 9-14% of independent gene pairs look dependent).
"""


def _target_line() -> str:
    parts = []
    for k in COLUMNS:
        t = TARGETS[k]
        parts.append(f"{k.replace('_', ' ')} {t.low:g}-{t.high:g}")
    return "Targets: " + "; ".join(parts) + "."


def render(results: Sequence[Dict[str, Any]]) -> str:
    rows = sorted(results, key=lambda r: (ORDER.index(r["generator"]), ["low", "medium", "high"].index(r["noise"])))
    L: List[str] = ["# Statistical data generators: calibration and noise levels", "", METHOD, "## Calibrated parameters", "",
                    "| generator | effect scale | noise (medium) | weight spread | unit of noise |", "|---|---|---|---|---|"]
    for g in ORDER[1:]:
        p = CALIBRATED[g]
        L.append(f"| {g} | {p.effect_scale:g} | {p.noise:g} | {p.weight_spread:g} | {UNITS[g]} |")
    L.append("| mechanistic | preset kinetics | 0.15 / 0.10 | preset | extrinsic log-sd per gene / shared (cell model) |")
    seeds = sorted({int(s) for r in rows for s in r.get("seeds", [])})
    L += ["", f"## Verification on unseen seeds ({', '.join(map(str, seeds)) if seeds else 'see the JSON'}; 100 genes, CRISPRi)", "",
          _target_line() + " Realism score: summed distance outside the ranges (0 = all inside). AUPR: direct edges, chance about "
          "0.024.", "",
          "| generator | noise | realism score | " + " | ".join(c.replace("_", " ") for c in COLUMNS) +
          " | AUPR mean diff. | AUPR S2G |", "|---|---|---|" + "---|" * len(COLUMNS) + "---|---|"]
    for r in rows:
        L.append(f"| {r['generator']} | {r['noise']} | {r['realism_score']} | " + " | ".join(str(r[c]) for c in COLUMNS) +
                 f" | {r['aupr_mean_difference']} | {r['aupr_s2g']} |")
    L += ["", "## Reading (computed from the table)", ""]
    for g in ORDER:
        sub = {r["noise"]: r for r in rows if r["generator"] == g}
        if not sub:
            continue
        med = sub.get("medium")
        outside = [c.replace("_", " ") for c in COLUMNS if med and not TARGETS[c].low <= med[c] <= TARGETS[c].high]
        trend = [sub[n]["aupr_mean_difference"] for n in ("low", "medium", "high") if n in sub]
        mono = all(a >= b for a, b in zip(trend, trend[1:]))
        L.append(f"* {g}: medium-noise realism score {med['realism_score'] if med else '–'}"
                 + (f"; outside the targets at medium noise: {', '.join(outside)}" if outside else "; every statistic inside the targets")
                 + f"; mean-difference AUPR low/medium/high = {' / '.join(f'{v:.3f}' for v in trend)}"
                 + (" (falls with noise)." if mono else " (not monotone in noise)."))
    return "\n".join(L) + "\n"
