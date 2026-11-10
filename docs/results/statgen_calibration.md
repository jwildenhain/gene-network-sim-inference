# Statistical data generators: calibration and noise levels

The Virtual Cell tab generates experimental data from statistical models (grn_sim.statgen) instead of the panel-1 Hill
kinetics; mechanistic data from the panel-1 engine stay available as one more generator. This page records how the noise
levels were set and checked.

## Method

1. **Grid search** (scripts/calibrate_statgen.py): for every generator, effect scale x noise on 100-gene modular graphs
   (the replogle_like graph of seeds 1-2), CRISPRi screens with 40 cells per perturbation and the measurement layer. Each
   grid point is scored by how far the observed screen statistics lie outside the genome-scale Perturb-seq ranges of
   grn_sim.perturbseq.realism.TARGETS (0 = all inside; the genome-scale share of indirect effects is left out).
2. **Choice**: where several grid points met the targets, the middle of that region rather than the edge of the grid.
3. **Noise levels**: low / medium / high = 0.5 / 1 / 2 x the calibrated cell noise (effects unchanged), so
   the levels differ in signal-to-noise only. For the mechanistic generator the same factors scale the extrinsic noise of the
   cell model (0.15 per gene, 0.10 shared at medium).
4. **Verification** (scripts/verify_statgen_noise.py) on seeds 501-503, never used for calibration, below.

Iterations that changed the models: the Boolean model was redesigned (hard threshold rules gave far too dense effects,
0.07-0.12 affected pairs, and a weak 0.64-0.75 knockdown: it is now a stochastic logistic rule over weighted inputs, and the
guide also scales the gene's own transcript); edge weights got a heavier tail (log-sd 0.8 -> 1.1) because the top-5% effect
share sat at 0.19-0.22 (target >= 0.25); the nonlinear model's noise was lowered (0.15 -> 0.07) because only 17% of
perturbations showed a phenotype; the mechanistic screens now use the preset's 0.92 guide efficiency (knockdown 0.69 -> 0.80).

## Calibrated parameters

| generator | effect scale | noise (medium) | unit |
|---|---|---|---|
| linear_sem | 0.15 | 0.20 | log-sd of the propagated noise |
| nonlinear_scm | 0.6 (capped by stability) | 0.07 | log-sd |
| nb_copula | 0.15 | 0.20 | sqrt of the NB dispersion |
| boolean | 1.2 (logistic gain / 4) | 1.0 | flip probability x 10 |
| mechanistic | preset kinetics | 0.15 / 0.10 | extrinsic log-sd per gene / shared |

## Verification on unseen seeds (100 genes, CRISPRi, 3 seeds per row)

Targets: knockdown 0.80-0.95, phenotype share 0.25-0.45, affected-pair density 0.015-0.05, top-5% effect share 0.25-0.70,
control false positives <= 0.05. AUPR = direct-edge AUPR of mean difference and S2G (chance about 0.024).

| generator | noise | realism score | knockdown | phenotype | pair density | top-5% share | control FP | AUPR mean diff. | AUPR S2G |
|---|---|---|---|---|---|---|---|---|---|
| mechanistic | low | 3.74 | 0.801 | 0.533 | 0.1 | 0.213 | 0.0 | 0.481 | 0.659 |
| mechanistic | medium | 3.55 | 0.795 | 0.467 | 0.1 | 0.213 | 0.0 | 0.431 | 0.628 |
| mechanistic | high | 3.43 | 0.802 | 0.237 | 0.1 | 0.213 | 0.0 | 0.302 | 0.548 |
| linear_sem | low | 0.36 | 0.807 | 0.343 | 0.022 | 0.233 | 0.0 | 0.311 | 0.32 |
| linear_sem | medium | 0.36 | 0.803 | 0.313 | 0.022 | 0.233 | 0.0 | 0.279 | 0.306 |
| linear_sem | high | 0.63 | 0.806 | 0.21 | 0.022 | 0.233 | 0.0 | 0.203 | 0.262 |
| nonlinear_scm | low | 0.4 | 0.796 | 0.25 | 0.021 | 0.317 | 0.0 | 0.291 | 0.278 |
| nonlinear_scm | medium | 0.49 | 0.802 | 0.19 | 0.021 | 0.317 | 0.0 | 0.28 | 0.278 |
| nonlinear_scm | high | 0.59 | 0.797 | 0.17 | 0.021 | 0.317 | 0.0 | 0.237 | 0.252 |
| nb_copula | low | 0.36 | 0.806 | 0.297 | 0.022 | 0.233 | 0.0 | 0.317 | 0.338 |
| nb_copula | medium | 1.72 | 0.805 | 0.253 | 0.022 | 0.233 | 0.083 | 0.288 | 0.305 |
| nb_copula | high | 0.58 | 0.805 | 0.21 | 0.022 | 0.233 | 0.0 | 0.211 | 0.258 |
| boolean | low | 0.79 | 0.853 | 0.353 | 0.053 | 0.257 | 0.0 | 0.363 | 0.415 |
| boolean | medium | 1.87 | 0.852 | 0.343 | 0.042 | 0.237 | 0.083 | 0.36 | 0.393 |
| boolean | high | 2.07 | 0.852 | 0.143 | 0.027 | 0.225 | 0.083 | 0.312 | 0.325 |

## Reading

* The statistical generators sit near the targets at medium noise (realism score 0.4-0.5, the misses are small: a top-5% share
  of 0.23 against 0.25, a phenotype share of 0.19 for the nonlinear model whose saturating links damp propagated effects).
* A control false-positive rate of 0.083 is one non-targeting guide group out of four in one of three seeds: with four control
  guides that statistic moves in steps of 25%.
* The mechanistic generator is further off (score about 3.5): its pair density of 0.10 is a property of the 500-gene-calibrated
  preset network at 100 genes (density and the indirect-effect share cannot both be met below genome scale), not of the noise.
* Noise levels change difficulty monotonically and keep it in a useful range: mean-difference AUPR falls from 0.29-0.48 (low)
  to 0.20-0.30 (high), far above chance and far below 1, for every generator.
