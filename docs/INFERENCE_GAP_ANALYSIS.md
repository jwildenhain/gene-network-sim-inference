# Inference gap analysis: G2S / NIR / RidgeCV versus inspre, and the new interventional methods

Branch `feat/inspre-and-interventional-baselines`. Numbers: 100-gene `replogle_like` networks, knockdown screens, 40 cells per
perturbation, LNA cells + measurement artefacts, 4 seeds (mean ± sd; the sd is large, treat differences below ~0.1 as ties).
Reproduce: `scripts/audit_inspre.py`, `scripts/benchmark_screen_methods.py [--prior]`. Raw rows: `docs/screen_*_100genes.json`.

## 1. Audit: what G2S, NIR and RidgeCV do relative to inspre

All of them solve `A X = -U` row by row, which for a square one-perturbation-per-gene screen is the inverse of the
response matrix (`A = -U X^-1`); inspre inverts the same matrix, normalises it so each gene's own effect is 1
(`G = I - V diag(1/diag V)`) and finds a *jointly* sparse inverse with one L1 penalty (ADMM).

| | inspre | NIR | S2G / G2S | RidgeCV |
|---|---|---|---|---|
| sparsity | joint L1 on all of V | exactly K regressors per row (greedy top-D search; BIC per row optional) | AIC forward / backward per row | L2, then top-K per row |
| own-gene term | in V, then normalised away | row i regresses on all other genes + its own perturbation | all genes incl. self as candidate | **excluded by default (`allow_self_regulation=False`)** |
| noise | per-node weights, optional edge penalties | none | none | CV alpha |
| density | free (matched to a target) | fixed K | AIC | fixed K |

Measured on a noise-free-to-noisy linear cyclic model (`scripts/audit_inspre.py`, AUPR of regulator->target):

| noise sd | inspre | S2G | RidgeCV (self allowed) | NIR K=3 | G2S (loo) | RidgeCV (default) |
|---|---|---|---|---|---|---|
| 0 | 1.00 | 0.84 | 0.86 | 0.71 | 0.43 | **0.07** |
| 0.02 | 0.89 | 0.87 | 0.79 | 0.65 | 0.43 | 0.07 |
| 0.05 | 0.69 | 0.64 | 0.53 | 0.58 | 0.40 | 0.07 |
| 0.10 | 0.32 | 0.28 | 0.13 | 0.28 | 0.21 | 0.08 |

Findings (verified by these runs):
1. **RidgeCV in additive (`A X = -U`) mode is essentially random (AUPR 0.07) with its default `allow_self_regulation=False`**: the
   perturbed gene's own term is the largest part of the equation and cannot be fitted, so the spike in `-U_i` leaks into other
   regressors. With `allow_self_regulation=True` it reaches 0.86 noise-free. The registry's Hill-mode ridge goes through
   `regulator_input_fit`, which has no self term by design, so the studio's Hill runs are not affected; additive/linear runs and
   anyone calling `KaggleMultiOutputRidge().fit(X, U)` directly are. Not changed here (the registry configuration is part of the
   archive's results); decide whether the additive default should be flipped.
2. When the data are exactly invertible, S2G and RidgeCV(self) are close to inspre; inspre's gain is **noise handling**, not a different
   estimand, and it needs no K. NIR with a fixed K loses recall on the extra (indirect) entries; G2S's AIC with `loo` is the weakest.
3. G2S/NIR/S2G cost grows steeply with genes; inspre needs one Cholesky: 500 genes in seconds.

## 2. Screen benchmark (Hill screens; direct truth = `screen.adjacency`, total truth = |log2FC| >= 0.1)

Input "cells" = calibrated ACE estimate of the simulated cells (`perturbseq/ace.py`); "cells+denoised" sets effects not
significant at BH 5% to zero; "truth" = noise-free log2FC. Legacy algorithms are fitted as `A X = -U` (U = each node's own response
on the diagonal) and transposed for scoring.

Direct-edge AUPR (total-effect AUPR in the JSON):

| method | truth | cells | cells+denoised |
|---|---|---|---|
| inspre | **0.77** | 0.24 | 0.31 |
| S2G | 0.76 | 0.35 | 0.24 |
| RidgeCV (self allowed) | 0.67 | 0.27 | 0.29 |
| NIR | 0.59 | 0.28 | 0.27 |
| mean difference | 0.45 | 0.25 | 0.24 |
| G2S (loo) | 0.26 | 0.08 | 0.15 |
| Pearson | 0.22 | 0.16 | 0.16 |
| GRNBoost-lite | 0.08 | 0.06 | 0.10 |
| gene-level features | 0.07 | 0.05 | 0.06 |
| RidgeCV (default) | 0.03 | 0.03 | 0.04 |
| **bicycle-lite** (gap 6) | - | 0.39 | 0.28 |
| **DOTEARS-lite** (gap 7) | - | 0.40 | 0.25 |
| **consensus** inspre+mean diff+bicycle+DOTEARS (gap 8) | - | **0.51** | 0.35 |

What this supports and does not:
- The ranking against **total** effects is dominated by mean difference (1.0 on truth, it is the definition) and then by methods
  that keep indirect effects; against **direct** edges it is the reverse. Reporting only one truth rewards the wrong thing.
- Gene-level features alone score 0.05-0.07 direct but 0.3-0.5 total: much of the "total" signal is hubness.
- With realistic noise every single method falls to 0.25-0.40 and the order changes with the input handling; the **consensus of the
  four interventional-aware methods (0.51) beat its best member (0.40)** in the raw-ACE setting, but not in the denoised setting
  (0.35, within noise of its best member 0.31), so the consensus claim is conditional.
- Denoising helps inspre (0.24 -> 0.31) but *narrowed* the candidate sets of bicycle-lite and DOTEARS-lite (their candidates come
  from inspre on the same input) and lowered them; their input-handling is therefore not comparable with inspre's in the denoised column.
- 4 seeds, one network size, one preset, one cell depth. No claim here about 500 genes: only inspre, mean difference, Pearson and
  gene features are fast enough there; bicycle-lite (~10 s) and DOTEARS-lite (~5 s) at 100 genes scale cubically in the genes.

## 3. What was built (gaps 1-4, 6, 7, 8, 9)

| gap | module | status |
|---|---|---|
| 1 baselines | `inference/interventional_baselines.py` (mean difference, Pearson, GRNBoost-lite, gene features) | done; GRNBoost-lite uses scikit-learn boosting, not the Dask GRNBoost2 |
| 2 evaluation | `evaluation/screen_benchmark.py`: direct and total AUPR + precision@k | done; held-out-perturbation prediction vs additive/linear baselines **not done** |
| 3 inspre | `inference/inspre.py`, registered as `inspre` | done: ADMM, per-node weights, per-edge penalty; no per-entry SE weights, lambda by target density |
| 4 calibrated ACE | `perturbseq/ace.py` | done as NT-calibrated Welch z + BH, **not SCEPTRE** (no NB regression / conditional resampling) |
| 6 bicycle-style | `inference/bicycle_lite.py` | done as sufficient-statistics OU/Lyapunov likelihood (gradients checked by finite differences); no latent layer, one shared covariance |
| 7 DOTEARS | `inference/dotears.py` | done (hard-intervention loss, per-gene variances, candidate-restricted); **CLIPR skipped: no code or reproducible description found** |
| 8 consensus | `inference/interventional_consensus.py` | done (rank-average) |
| 9 prior | `evaluation/prior_experiment.py` | done, below |

Gap 5 (sweeps over knockdown strength, guide efficiency, escapers, cells, dropout, cycles) was not asked for and not run.

## 4. Noisy-true-network prior (gap 9)

A prior keeps a fraction `sensitivity` of the true direct edges plus false pairs for a given `precision`. inspre: L1 penalty x0.2 on
prior pairs; mean difference: score x2 on prior pairs. AUPR on direct edges (mean of 4 seeds, inspre without prior 0.31, mean difference 0.24):

| sensitivity / precision | prior alone | inspre + prior | mean diff + prior | inspre, pairs *outside* the prior |
|---|---|---|---|---|
| 0.2 / 0.9 | 0.21 | 0.32 | 0.27 | 0.29 (0.29 without prior) |
| 0.5 / 0.5 | 0.27 | 0.34 | 0.30 | 0.22 (0.22) |
| 0.8 / 0.9 | **0.73** | 0.36 | 0.34 | 0.16 (0.15) |
| 0.8 / 0.2 | 0.18 | 0.38 | 0.31 | 0.14 (0.12) |
| any / 0.0 (all wrong) | 0.02 | 0.31 (no change) | 0.24 (no change) | unchanged |

- A wrong prior at this strength costs nothing; a right prior adds at most +0.07 (inspre) / +0.09 (mean difference).
- **The gain is the prior's own edges.** Scored only on pairs outside the prior, the prior changes almost nothing (<= +0.02): it does
  not make the data-driven part find new edges.
- A good prior (0.8 / 0.9) alone (0.73) beats every combination (0.36): the soft penalty/boost under-uses a precise prior. A
  product-of-evidence or a stronger penalty is the next experiment; the grid of strengths was not scanned.

## 5. Not verified / limits
- All results are from this simulator (Hill kinetics, LNA cells); published methods' real-data behaviour is not tested here.
- 4 seeds, sd 0.04-0.11 on most cells; the consensus gain and the S2G-vs-inspre ordering on noisy input are within 1-2 sd.
- The Dockerised studio (port 8085) runs an old image and was not touched; the new algorithms appear in the registry/API after a rebuild.
