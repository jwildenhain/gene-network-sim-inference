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
   anyone calling `KaggleMultiOutputRidge().fit(X, U)` directly are. **Changed 2026-11-04:** the additive default now includes the own-gene term (`allow_self_regulation=None` -> True in additive fits, False in
   the regulator-input fit; the diagonal of `A_est` stays 0, the own term is a nuisance regressor). Archived ridge_cv runs fitted in additive mode
   before this change were near random and are not rescored.
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

## 5. Caveats: what was resolved, made optional, or stays open (2026-11-04)

| earlier caveat | now | evidence |
|---|---|---|
| bicycle-lite / DOTEARS-lite restricted to inspre candidates | **optional**: `candidates_per_node=None` frees every pair | DOTEARS-lite 0.245 -> **0.409** (sd 0.03), bicycle-lite 0.273 -> 0.286; 1.5x slower (`docs/screen_ablation_100genes.md`) |
| inspre weights per node only | **optional**: `fit_ace(..., entry_weights=True)` (inverse-variance weight for every entry; ~p^4, use for a few hundred nodes) | 0.298 -> 0.310, within noise, 12x slower: left off |
| calibrated Welch test is not SCEPTRE | **optional KS test**: `estimate_ace(..., test="ks")`; full SCEPTRE (negative-binomial regression + conditional resampling) is **not** implemented | KS 0.249 vs Welch 0.298 direct AUPR in inspre (one preset, 3 seeds) |
| held-out-perturbation prediction vs additive/linear baselines missing | **done**: `evaluation/heldout_prediction.py`, `scripts/heldout_benchmark.py` | Hidden single perturbations are *not predictable from the structure*: r = 0.03 (inspre) vs 0.07 (mean profile) vs -0.06 (neighbour heuristic). Expected: the hidden node's outgoing effects are only visible when it is perturbed. On the simulator's double perturbations the additive prediction already has r = 0.98 with the truth, so a linear structure cannot beat additive there (consistent with Ahlmann-Eltze) |
| no sweeps over screen conditions | **done**: `scripts/sweep_screen_conditions.py` (cells per perturbation, guide efficiency) | below |
| bicycle-lite has no latent layer / one shared covariance; CLIPR skipped | **open**: a latent-variable bicycle is a research-sized change; no CLIPR code or reproducible description found | - |
| only 100 genes | **partly**: run time to 1000 genes measured (section 7); accuracy at 250-500 genes with cells **not** measured | - |
| studio ran an old image | **done**: image rebuilt and container replaced (section 8) | - |

Sweep (direct AUPR, 100 genes, 3 seeds, denoised ACE; `docs/screen_condition_sweep_100genes.md`):

| cells / perturbation | guide efficiency | inspre | mean diff | S2G | NIR | RidgeCV (self) |
|---|---|---|---|---|---|---|
| 10 | 0.92 | 0.10 | 0.16 | 0.16 | 0.17 | 0.07 |
| 20 | 0.92 | 0.18 | 0.22 | 0.29 | 0.24 | 0.17 |
| 40 | 0.92 | 0.33 | 0.22 | 0.37 | 0.30 | 0.35 |
| 100 | 0.92 | **0.50** | 0.32 | 0.48 | 0.44 | 0.49 |
| 40 | 0.70 | 0.22 | 0.17 | 0.24 | 0.23 | 0.24 |
| 40 | 0.50 | 0.14 | 0.16 | 0.10 | 0.13 | 0.08 |

Cells per perturbation and knockdown strength matter more than the choice of method: every method roughly doubles from 20 to 100 cells and halves
from efficiency 0.92 to 0.5; inspre and RidgeCV(self) gain most from more cells, S2G is best at low depth, and at weak knockdown (0.5) all are at 0.1-0.16.

## 6. All benchmark datasets (`scripts/benchmark_all_datasets.py`, `docs/all_datasets_benchmark.json`)

Five datasets have a ground truth; the 105-gene Norman matrix has none (timing and agreement only). Edges, AUPR in T[regulator, target].
**Only `ecoli_sos` and `norman_2019_real_16tf` are measured data (9 and 16 genes); `dream4_10gene`, `replogle_2022_p53` and `norman_2019_tf16` are
generated by inverting a linear model of the truth, so inspre/mean difference reaching 1.0 there is not evidence about real data.**

| dataset (genes) | random | inspre | mean diff | Pearson | GRNBoost-lite | S2G | NIR | RidgeCV | GENIE3 |
|---|---|---|---|---|---|---|---|---|---|
| ecoli_sos (9, real) | 0.57 | 0.63 | 0.61 | 0.65 | 0.63 | 0.63 | 0.66 | 0.59 | 0.66 |
| norman_2019_real_16tf (16, real) | 0.11 | 0.23 | 0.24 | 0.24 | 0.15 | 0.18 | 0.25 | 0.24 | 0.27 |
| dream4_10gene (10, linear-generated) | 0.12 | 1.00 | 1.00 | 0.53 | 0.15 | 1.00 | 0.22 | 1.00 | 0.26 |
| replogle_2022_p53 (12, linear-generated) | 0.13 | 1.00 | 0.97 | 0.58 | 0.18 | 0.75 | 0.37 | 0.99 | 0.36 |
| norman_2019_tf16 (16, linear-generated) | 0.10 | 1.00 | 0.73 | 0.51 | 0.16 | 0.67 | 0.67 | 1.00 | 0.34 |

On the two measured sets no method is clearly better than the others (SOS is near the 0.57 random level for all; Norman 16: 0.15-0.27 against 0.11
random, GENIE3 highest). The 105-gene matrix is ill-conditioned (condition number 5e4) and inspre's solution has spectral radius 1.8 (> 1, the stable-feedback
assumption is violated); the result is flagged `unstable` in the diagnostics.

## 7. Run time and memory (`scripts/scaling_benchmark.py`, one core per fit unless noted)

Seconds, knockdown screens with 60 cells per perturbation (all nodes with a detectable knockdown); `docs/scaling_benchmark_60cells.jsonl`. Screen simulation itself: 8 / 30 / 97 s at 100 / 250 / 500 genes.

| genes | ACE estimate | inspre | mean diff / Pearson / gene features | GRNBoost-lite | bicycle-lite (40 iterations) | DOTEARS-lite (2 rounds) | peak RSS |
|---|---|---|---|---|---|---|---|
| 100 | 0.04 | 0.12 | < 0.01 | 1.0 | 2.5 | 0.7 | 0.2 GB |
| 250 | 0.2 | 7.9 | 0.01 | 2.9 | 31 | 13 | 0.7 GB |
| 500 | 0.9 | 26 | 0.03 | 17 | 147 | 77 | 3.1 GB |
| 1000 (noise-free input, no cells) | - | 66 | 0.1-0.2 | 68 | - | - | 0.5 GB |

- **Efficient:** ACE estimation, mean difference, Pearson, gene features, consensus (seconds or less at 500 genes); inspre (26 s at 500, 66 s at 1000;
  the process pins BLAS to one thread, see section 9, so the dense solves are single-core); GRNBoost-lite (parallel over targets).
- **Not efficient at scale:** bicycle-lite and DOTEARS-lite. At 500 genes 40 L-BFGS iterations take 2.5 min and 2 DOTEARS rounds 77 s without converging;
  their cost grows ~p^3 per evaluation (Lyapunov solves, matrix exponential) and DOTEARS' per-node moments need p^3 memory. Practical to about 250 genes.
- The old methods for comparison (from the 500-gene runs recorded earlier): NIR ~200 s, RidgeCV ~210 s, G2S/Vote time out.

## 8. Docker
`grn-sim-phpiorn:latest` was rebuilt from this branch (the build runs the whole test suite: 683 passed, 5 skipped) and the container replaced; the previous image
is tagged `grn-sim-phpiorn:before-inspre-2026-11-04` and the previous container is kept, stopped, as `grn-sim-phpiorn-old-2026-11-04`. `/api/algorithms` lists 24
algorithms including `inspre`, `mean_difference`, `pearson`, `grnboost`, `gene_features`. `./data` (archive) is the same bind mount.

## 9. A bug found on the way: forking after a screen simulation could hang
The first rebuild stalled for 20+ minutes in the in-image tests. Cause (stack dump of the stuck worker): a forked G2S worker was blocked in
`blas_thread_init` (OpenBLAS mutex). `simulate_cells` raised the BLAS thread count to 8 (`CellConfig.blas_threads=8`), which creates OpenBLAS's thread pool
for good; the package pins BLAS to one thread precisely because process pools forked beside that pool can deadlock. Any server process that exported a screen and
then fitted G2S with a process pool could hang. Default is now `None` (keep the process limit); simulation time did not change (250 genes: 26.5 s vs 28.0 s with 8 threads).
Reproduction: 4 of 6 runs hung before the fix, 0 of 6 after; regression test `test_simulating_a_screen_leaves_blas_on_one_thread`.

## 10. Not verified / limits
- All results are from this simulator (Hill kinetics, LNA cells); published methods' real-data behaviour is not tested here.
- 4 seeds, sd 0.04-0.11 on most cells; the consensus gain and the S2G-vs-inspre ordering on noisy input are within 1-2 sd.
- The Dockerised studio (port 8085) runs an old image and was not touched; the new algorithms appear in the registry/API after a rebuild.
