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
depend on a perturbation show exactly no effect); edge weights got a heavier tail (log-sd 1.1); the nonlinear model's noise was
lowered; the mechanistic screens use the preset's 0.92 guide efficiency; effects are estimated with median-ratio size factors
(total-count normalisation made 9-14% of independent gene pairs look dependent).

## Calibrated parameters

| generator | effect scale | noise (medium) | unit |
|---|---|---|---|
| linear_sem | 0.15 | 0.2 | log-sd of the propagated noise |
| nonlinear_scm | 0.6 | 0.07 | log-sd |
| nb_copula | 0.15 | 0.2 | sqrt of the NB dispersion |
| boolean | 1.2 | 1 | flip probability x 10 |
| mechanistic | preset kinetics | 0.15 / 0.10 | extrinsic log-sd per gene / shared (cell model) |

## Verification on unseen seeds (501, 502, 503; 100 genes, CRISPRi)

Targets: on target knockdown 0.8-0.95; fraction with phenotype 0.25-0.45; affected pair density 0.015-0.05; top5pct perturbation effect share 0.25-0.7; control false positive rate 0-0.05. Realism score: summed distance outside the ranges (0 = all inside). AUPR: direct edges, chance about 0.024.

| generator | noise | realism score | on target knockdown | fraction with phenotype | affected pair density | top5pct perturbation effect share | control false positive rate | AUPR mean diff. | AUPR S2G |
|---|---|---|---|---|---|---|---|---|---|
| mechanistic | low | 3.74 | 0.801 | 0.533 | 0.1 | 0.213 | 0.0 | 0.508 | 0.659 |
| mechanistic | medium | 3.55 | 0.795 | 0.467 | 0.1 | 0.213 | 0.0 | 0.461 | 0.64 |
| mechanistic | high | 3.43 | 0.802 | 0.237 | 0.1 | 0.213 | 0.0 | 0.311 | 0.562 |
| linear_sem | low | 0.36 | 0.807 | 0.343 | 0.022 | 0.233 | 0.0 | 0.427 | 0.399 |
| linear_sem | medium | 0.36 | 0.803 | 0.313 | 0.022 | 0.233 | 0.0 | 0.383 | 0.377 |
| linear_sem | high | 0.63 | 0.806 | 0.21 | 0.022 | 0.233 | 0.0 | 0.286 | 0.301 |
| nonlinear_scm | low | 0.4 | 0.796 | 0.25 | 0.021 | 0.317 | 0.0 | 0.301 | 0.281 |
| nonlinear_scm | medium | 0.49 | 0.802 | 0.19 | 0.021 | 0.317 | 0.0 | 0.292 | 0.284 |
| nonlinear_scm | high | 0.59 | 0.797 | 0.17 | 0.021 | 0.317 | 0.0 | 0.24 | 0.255 |
| nb_copula | low | 0.36 | 0.806 | 0.297 | 0.022 | 0.233 | 0.0 | 0.432 | 0.412 |
| nb_copula | medium | 1.72 | 0.805 | 0.253 | 0.022 | 0.233 | 0.083 | 0.392 | 0.372 |
| nb_copula | high | 0.58 | 0.805 | 0.21 | 0.022 | 0.233 | 0.0 | 0.293 | 0.299 |
| boolean | low | 2.15 | 0.851 | 0.373 | 0.051 | 0.254 | 0.083 | 0.341 | 0.421 |
| boolean | medium | 1.88 | 0.852 | 0.283 | 0.041 | 0.242 | 0.083 | 0.343 | 0.384 |
| boolean | high | 0.35 | 0.847 | 0.197 | 0.026 | 0.224 | 0.0 | 0.303 | 0.302 |

## Reading (computed from the table)

* mechanistic: medium-noise realism score 3.55; outside the targets at medium noise: on target knockdown, fraction with phenotype, affected pair density, top5pct perturbation effect share; mean-difference AUPR low/medium/high = 0.508 / 0.461 / 0.311 (falls with noise).
* linear_sem: medium-noise realism score 0.36; outside the targets at medium noise: top5pct perturbation effect share; mean-difference AUPR low/medium/high = 0.427 / 0.383 / 0.286 (falls with noise).
* nonlinear_scm: medium-noise realism score 0.49; outside the targets at medium noise: fraction with phenotype; mean-difference AUPR low/medium/high = 0.301 / 0.292 / 0.240 (falls with noise).
* nb_copula: medium-noise realism score 1.72; outside the targets at medium noise: top5pct perturbation effect share, control false positive rate; mean-difference AUPR low/medium/high = 0.432 / 0.392 / 0.293 (falls with noise).
* boolean: medium-noise realism score 1.88; outside the targets at medium noise: top5pct perturbation effect share, control false positive rate; mean-difference AUPR low/medium/high = 0.341 / 0.343 / 0.303 (not monotone in noise).
