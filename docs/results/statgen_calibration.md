# Statistical data generators: calibration and noise levels

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

## Calibrated parameters

| generator | effect scale | noise (medium) | weight spread | unit of noise |
|---|---|---|---|---|
| linear_sem | 0.25 | 0.2 | 1.7 | log-sd of the propagated noise |
| nonlinear_scm | 0.6 | 0.03 | 1.4 | log-sd |
| nb_copula | 0.25 | 0.2 | 1.7 | sqrt of the NB dispersion |
| boolean | 1.2 | 1 | 1.7 | flip probability x 10 |
| mechanistic | preset kinetics | 0.15 / 0.10 | preset | extrinsic log-sd per gene / shared (cell model) |

## Verification on unseen seeds (501, 502, 503; 100 genes, CRISPRi)

Targets: on target knockdown 0.8-0.95; fraction with phenotype 0.25-0.45; affected pair density 0.015-0.05; top5pct perturbation effect share 0.25-0.7; control false positive rate 0-0.05. Realism score: summed distance outside the ranges (0 = all inside). AUPR: direct edges, chance about 0.024.

| generator | noise | realism score | on target knockdown | fraction with phenotype | affected pair density | top5pct perturbation effect share | control false positive rate | AUPR mean diff. | AUPR S2G |
|---|---|---|---|---|---|---|---|---|---|
| mechanistic | low | 3.74 | 0.801 | 0.533 | 0.1 | 0.213 | 0.0 | 0.508 | 0.659 |
| mechanistic | medium | 3.55 | 0.795 | 0.467 | 0.1 | 0.213 | 0.0 | 0.461 | 0.64 |
| mechanistic | high | 3.43 | 0.802 | 0.237 | 0.1 | 0.213 | 0.0 | 0.311 | 0.562 |
| linear_sem | low | 0.68 | 0.797 | 0.407 | 0.023 | 0.261 | 0.0 | 0.408 | 0.402 |
| linear_sem | medium | 0.54 | 0.801 | 0.337 | 0.023 | 0.261 | 0.0 | 0.361 | 0.373 |
| linear_sem | high | 0.59 | 0.805 | 0.263 | 0.023 | 0.261 | 0.0 | 0.275 | 0.324 |
| nonlinear_scm | low | 0.41 | 0.796 | 0.223 | 0.022 | 0.333 | 0.0 | 0.313 | 0.306 |
| nonlinear_scm | medium | 0.37 | 0.798 | 0.197 | 0.022 | 0.333 | 0.0 | 0.321 | 0.316 |
| nonlinear_scm | high | 0.36 | 0.795 | 0.223 | 0.022 | 0.333 | 0.0 | 0.314 | 0.317 |
| nb_copula | low | 0.54 | 0.799 | 0.397 | 0.023 | 0.261 | 0.0 | 0.41 | 0.391 |
| nb_copula | medium | 0.55 | 0.802 | 0.37 | 0.023 | 0.261 | 0.0 | 0.388 | 0.377 |
| nb_copula | high | 0.57 | 0.8 | 0.27 | 0.023 | 0.261 | 0.0 | 0.308 | 0.316 |
| boolean | low | 0.0 | 0.853 | 0.253 | 0.029 | 0.289 | 0.0 | 0.275 | 0.307 |
| boolean | medium | 0.1 | 0.851 | 0.233 | 0.024 | 0.277 | 0.0 | 0.277 | 0.288 |
| boolean | high | 0.37 | 0.852 | 0.18 | 0.017 | 0.248 | 0.0 | 0.251 | 0.245 |

## Reading (computed from the table)

* mechanistic: medium-noise realism score 3.55; outside the targets at medium noise: on target knockdown, fraction with phenotype, affected pair density, top5pct perturbation effect share; mean-difference AUPR low/medium/high = 0.508 / 0.461 / 0.311 (falls with noise).
* linear_sem: medium-noise realism score 0.54; every statistic inside the targets; mean-difference AUPR low/medium/high = 0.408 / 0.361 / 0.275 (falls with noise).
* nonlinear_scm: medium-noise realism score 0.37; outside the targets at medium noise: on target knockdown, fraction with phenotype; mean-difference AUPR low/medium/high = 0.313 / 0.321 / 0.314 (not monotone in noise).
* nb_copula: medium-noise realism score 0.55; every statistic inside the targets; mean-difference AUPR low/medium/high = 0.410 / 0.388 / 0.308 (falls with noise).
* boolean: medium-noise realism score 0.1; outside the targets at medium noise: fraction with phenotype; mean-difference AUPR low/medium/high = 0.275 / 0.277 / 0.251 (not monotone in noise).
