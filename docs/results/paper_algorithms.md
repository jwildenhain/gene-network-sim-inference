# Paper algorithms tried (2026-11-21)

Implemented and benchmarked. **LLC mean-shift passed and is registered as `llc_mean` since 1.0.5** (`src/grn_sim/inference/llc.py`); DCD-FG-lite
stays experimental and unregistered (`src/grn_sim/inference/experimental/`: it fits cells, not an (X, U) pair, and it did not beat the
baselines); the SCEPTRE-style test is an option of `estimate_ace(..., test="sceptre")`. Benchmark `scripts/benchmark_paper_algorithms.py`;
raw rows `paper_algorithms.json`.

## Direct-edge AUPR on full screens (100 cells per perturbation, 3 seeds, mean ± sd)

| generator | genes | llc_mean | dcdfg_lite | cellbox | ridge_cv | s2g | inspre | mean_difference | nir | notears |
|---|---|---|---|---|---|---|---|---|---|---|
| mechanistic | 50 | 0.66 ± 0.01 | 0.21 ± 0.11 | 0.67 ± 0.03 | 0.63 ± 0.22 | 0.62 ± 0.12 | 0.62 ± 0.04 | 0.43 ± 0.06 | 0.53 ± 0.03 | 0.16 ± 0.02 |
| mechanistic | 100 | 0.68 ± 0.04 | 0.14 ± 0.02 | 0.61 ± 0.05 | 0.39 ± 0.17 | 0.50 ± 0.02 | 0.42 ± 0.13 | 0.29 ± 0.12 | 0.35 ± 0.05 | 0.09 ± 0.01 |
| linear_sem | 50 | 0.49 ± 0.02 | 0.25 ± 0.01 | 0.50 ± 0.03 | 0.52 ± 0.03 | 0.51 ± 0.02 | 0.51 ± 0.02 | 0.47 ± 0.01 | 0.49 ± 0.02 | 0.16 ± 0.01 |
| linear_sem | 100 | 0.45 ± 0.01 | 0.12 ± 0.01 | 0.43 ± 0.02 | 0.47 ± 0.02 | 0.45 ± 0.02 | 0.40 ± 0.02 | 0.44 ± 0.02 | 0.37 ± 0.02 | 0.07 ± 0.02 |
| nb_copula | 50 | 0.48 ± 0.01 | 0.25 ± 0.01 | 0.50 ± 0.02 | 0.52 ± 0.04 | 0.50 ± 0.02 | 0.50 ± 0.01 | 0.47 ± 0.02 | 0.47 ± 0.00 | 0.20 ± 0.03 |
| nb_copula | 100 | 0.46 ± 0.01 | 0.13 ± 0.01 | 0.44 ± 0.01 | 0.48 ± 0.02 | 0.46 ± 0.02 | 0.41 ± 0.01 | 0.44 ± 0.02 | 0.39 ± 0.01 | 0.07 ± 0.02 |
| nonlinear_scm | 50 | 0.42 ± 0.02 | 0.14 ± 0.03 | 0.43 ± 0.02 | 0.45 ± 0.02 | 0.43 ± 0.03 | 0.43 ± 0.02 | 0.41 ± 0.02 | 0.42 ± 0.03 | 0.13 ± 0.02 |
| nonlinear_scm | 100 | 0.35 ± 0.04 | 0.06 ± 0.01 | 0.34 ± 0.02 | 0.39 ± 0.04 | 0.36 ± 0.03 | 0.31 ± 0.02 | 0.35 ± 0.04 | 0.29 ± 0.03 | 0.05 ± 0.02 |

Chance level is about 0.02-0.05 at these sizes. Mean fit time: llc_mean 0.1 s, ridge_cv < 0.1 s, s2g 0.4 s, nir 0.4 s,
notears 0.8 s, **dcdfg_lite 31 s** (full-batch Adam, dense matrix exponential).

## Results

1. **LLC (mean-shift part, `llc_mean`) is in the top tier on calibrated screens and clearly best on mechanistic networks at 100 genes**
   (0.68 against cellbox 0.61, s2g 0.50, ridge_cv 0.39, nir 0.35, inspre 0.42). On the three statistical generators it ties the best
   methods (differences of 0.01-0.05, inside the seed spread). It is a per-target ridge regression on the shifts of the other
   genes, i.e. the same equations NIR and RidgeCV solve; the gap to ridge_cv on mechanistic networks comes from the regularisation
   and the use of raw (not own-normalised) shifts, which I have not isolated. The covariance constraints of the full LLC were not
   implemented (they need many cells per experiment). Three more checks were made when it was registered (1.0.5):
   - **Partial screens (native, 100 genes, 3 seeds, `partial_grid/llc_fast_grnboost_100genes.md`):** the best method at every fraction,
     balanced AUPR 0.46 / 0.56 / 0.62 / 0.67 at 6 / 20 / 50 / 75 % of the genes perturbed (NIR native 0.20 / 0.40 / 0.42 / 0.39, inspre
     0.16 / 0.43 / 0.31 / 0.39).
   - **Measured data (`all_datasets_benchmark.json`):** E. coli SOS 0.58 (random 0.57), Norman 16 TFs 0.26 (random 0.11): in the pack with
     inspre 0.65 / 0.23, NIR 0.66 / 0.25 and GENIE3 0.66 / 0.27. Two measured sets of 9 and 16 genes cannot rank methods.
   - **The studio's own Hill (X, U) data is where it is mid-field:** regulator-input fits on `replogle_like` modular networks give F1 0.26 / 0.23
     at 50 / 100 genes (AUPR 0.29 / 0.20) against NIR 0.43 / 0.37 (0.35 / 0.25) and inspre 0.39 / 0.35 (0.42 / 0.28); on the 100-gene audit
     network it ranks 21st of 25 by F1 (`audit_100_250.md`). The calibrated-screen lead therefore does not carry over to every data model,
     and the ridge strength (0.01 to 1) does not change that. It is registered as one more method, not as a new best.
2. **DCD-FG-lite does not beat anything**: 0.06-0.25, between NOTEARS (0.05-0.20) and the rest, and about 300 times slower than
   LLC. The low-rank factorisation and interventional likelihood did not help on these simulated screens; the paper's negative-binomial
   model and low-rank acyclicity trick were not implemented, so this is a test of the linear-Gaussian simplification only.
3. **SCEPTRE-style conditional resampling** (`test="sceptre"`) matches the calibrated Welch test within noise: inspre on
   denoised effects 0.45 / 0.60 against 0.47 / 0.62 at 100 / 250 genes, mean-difference 0.29 / 0.44 against 0.29 / 0.43, ranking by |z|
   0.36 / 0.57 against 0.35 / 0.55 (3 seeds). A first version resampled among the control cells only and collapsed to chance because
   the screen has barely more control than perturbed cells; resampling over perturbed and control cells together fixed it. Full SCEPTRE
   (negative-binomial regression, skew-normal null, more covariates) was not implemented. The option is available through
   `screen_inputs(..., test="sceptre")` and `estimate_ace(test="sceptre")` only; the studio's pages do not expose it.

## Not implemented, and why

- **GIES, IGSP**: score- or CI-test-based permutation search; practical to about 100 genes and binary output (no AUPR), no existing
  implementation in the environment, and installing a third-party package was not part of the approved downloads.
- **DCDI**: the same family as the DOTEARS-lite and DCD-FG-lite versions already in the repository.
- **SparseRC**: needs a root-cause (sparse-source) assumption on a DAG and observational data that these screens do not provide.

## Limits

100 cells per perturbation, 50 and 100 genes (DCD-FG-lite not run above 150), 3 seeds, mechanistic and three statistical generators,
full (not partial) screens. Time was measured with the other jobs on the machine idle.
