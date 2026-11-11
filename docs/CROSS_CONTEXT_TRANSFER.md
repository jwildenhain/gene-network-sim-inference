# Cross-Context Transfer Predictors

Three lightweight predictors for the Virtual Cell Challenge tab (response method **Cross-context transfer**), a plain
source-atlas average as their reference, and a paired multi-network study of whether calibrating on a little target data
beats that average. Code: `src/grn_sim/benchmarks/cross_context.py`; study: `scripts/transfer_study.py`.

## Summary

1. **The plain average of the source atlases is already a strong predictor** in this simulator: held-out Pearson r about
   0.49 (activation data) and 0.90 (CRISPRi knockdown, all genes; 0.39 off-target) at 15% rewiring, against about 0 for the
   mean of the seen target columns.
2. **None of the three calibrated predictors beats it.** On 24 networks not used for tuning, every paired difference is
   within ±0.02 with a standard error of 0.01-0.03 (tables below). Gains of about +0.035 seen on the tuning networks did not
   replicate.
3. **Target-specific mechanism wins once the target is rewired enough.** At 30% rewiring the oracle (the target's true
   regulators with Hill kinetics refit on the seen target columns) is ahead of the source average by +0.17 ± 0.05 (activation,
   19/24 networks) and +0.15 ± 0.06 (knockdown, 17/24); no transfer predictor closes that gap.
4. **The "0" of the AtlasShift row is a convention, not a finding.** The benchmark's AtlasShift row predicts zeros for the
   held-out genes because they have "no atlas observation". The transfer rows keep the gene in the source atlases and hold it
   out of the *target context* only (the 2026 zero-shot setting): the same average then reaches r of about 0.5.

## The setting

Three source atlases (the same scale-free network with ±15% different basal rates: simulated source contexts A, B and C) and a
target context with a fraction of its edges moved (`_simulate_contexts` in `vcc_benchmark.py`). The predictors get the source
responses, the target's **seen** perturbations, the perturbation design for every column (which gene is hit, how hard) and the
target's unperturbed state; the target responses of the held-out columns are never read (tested by poisoning them with NaN).
Seen columns are predicted **leave-one-out**, so no column is ever predicted from its own target measurements (tested, and
shown to catch a leaky variant).

## The predictors

All are closed-form NumPy code. They are written from public *descriptions* of what worked in the challenge, simplified to
what a simulator without single cells can express. **No team's code was used or consulted**, and they are not reproductions of
the teams' methods.

| Key | Inspired by | What it does |
| :--- | :--- | :--- |
| `atlas_mean` | reference | The mean of the source responses. No calibration. |
| `atlas_calibrated` | AtlasShift's weighted ensemble of atlases with target-context calibration ([repository README](https://github.com/kaipengm2/Virtual-Cell-Challenge-2026)); its promoter-neighbour correction has no counterpart here and is omitted | Sources weighted by `1 / (error on the seen target columns)`, then a per-gene scale `(Σ M·T + λ) / (Σ M² + λ)`, i.e. ridge towards 1 (trust the sources unless the target says otherwise), clipped to [0, 3]. |
| `transpert_style` | the 2025 team Outlier's TransPert: summary-level data from several reference cell lines, differential-expression-style filtering, a globally optimised linear scale ([Arc wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up)) | The cross-source median, kept only where all sources agree in sign and exceed 0.5 × the consensus noise floor (the differential-expression gate), times one global scale fitted on the seen columns. |
| `pseudobulk_ridge` | pseudo-bulk regression with engineered biological features (2025 teams XLearning Lab and BioMap; a regression model at the top of the board, per Turbine's write-up, which I could not retrieve) | A pooled ridge regression of what the source mean gets wrong, on the off-target source mean, the on-target indicator, the on-target indicator times the design strength, and the off-target mean times the gene's basal expression level. |

Hyper-parameters (calibration ridge 1, gate noise scale 0.5, regression ridge 1) were chosen on tuning networks (seeds
101-112, both perturbation modes, rewiring 0.15 and 0.30); the study below uses seeds 1-24.

## Results

N = 16 genes, 4 of 16 target perturbations held out, 1% noise, 24 networks (seeds 1-24), Pearson r averaged over the held-out
perturbations. "Off-target" leaves the perturbed gene out, which matters for knockdown data (the perturbed gene is about 77%
of every profile). The paired difference is to the source-atlas average on the same network, ± standard error, with the
number of networks on which the predictor wins.

### Activation data (default perturbation)

| Predictor | r, all genes | r, off-target | off-target minus source average | wins |
| :--- | :---: | :---: | :---: | :---: |
| **Rewiring 15%** | | | | |
| Source-atlas average | +0.486 | +0.498 | — | — |
| AtlasShift, calibrated | +0.482 | +0.492 | -0.006 ± 0.011 | 15/24 |
| TransPert-style | +0.501 | +0.505 | +0.007 ± 0.016 | 12/24 |
| Pseudo-bulk regression | +0.465 | +0.491 | -0.007 ± 0.009 | 13/24 |
| mean of seen target columns (null) | +0.015 | +0.002 | -0.496 ± 0.074 | 2/24 |
| true structure + Hill (oracle) | +0.475 | +0.472 | -0.026 ± 0.047 | 13/24 |
| **Rewiring 30%** | | | | |
| Source-atlas average | +0.283 | +0.289 | — | — |
| AtlasShift, calibrated | +0.287 | +0.292 | +0.003 ± 0.013 | 14/24 |
| TransPert-style | +0.291 | +0.291 | +0.001 ± 0.016 | 11/24 |
| Pseudo-bulk regression | +0.282 | +0.294 | +0.005 ± 0.003 | 13/24 |
| mean of seen target columns (null) | +0.044 | +0.030 | -0.259 ± 0.071 | 6/24 |
| true structure + Hill (oracle) | +0.453 | +0.461 | **+0.171 ± 0.046** | 19/24 |

### CRISPRi knockdown data (`perturbation=knockdown`)

| Predictor | r, all genes | r, off-target | off-target minus source average | wins |
| :--- | :---: | :---: | :---: | :---: |
| **Rewiring 15%** | | | | |
| Source-atlas average | +0.897 | +0.389 | — | — |
| AtlasShift, calibrated | +0.897 | +0.392 | +0.002 ± 0.022 | 13/24 |
| TransPert-style | +0.902 | +0.405 | +0.016 ± 0.018 | 15/24 |
| Pseudo-bulk regression | +0.882 | +0.372 | -0.018 ± 0.020 | 12/24 |
| mean of seen target columns (null) | +0.006 | +0.047 | -0.342 ± 0.074 | 4/24 |
| true structure + Hill (oracle) | +0.905 | +0.330 | -0.060 ± 0.069 | 12/24 |
| **Rewiring 30%** | | | | |
| Source-atlas average | +0.790 | +0.229 | — | — |
| AtlasShift, calibrated | +0.778 | +0.212 | -0.018 ± 0.017 | 9/24 |
| TransPert-style | +0.794 | +0.248 | +0.019 ± 0.020 | 14/24 |
| Pseudo-bulk regression | +0.772 | +0.188 | -0.041 ± 0.027 | 5/24 |
| mean of seen target columns (null) | -0.011 | +0.117 | -0.112 ± 0.049 | 7/24 |
| true structure + Hill (oracle) | +0.893 | +0.380 | **+0.151 ± 0.056** | 17/24 |

## Reading the results

- **Why calibration has little to add.** The simulated contexts differ only by ±15% basal rates and moved edges, so the source
  responses already carry most of the target's response; per-gene scales and a global scale have little left to correct, and a
  pooled regression fitted on 12 seen columns mostly fits noise. Real cell lines differ far more, so this is a statement about
  this simulator, not about the challenge.
- **The pseudo-bulk regression is the least robust** (down to -0.04 off-target at 30% rewiring, knockdown). On the tuning
  networks the expression-level interaction was the only feature that changed off-target correlation, and it hurt in three of
  four conditions; a stronger ridge collapses the model back to the plain average. It is kept as a negative control.
- **Tuning gains did not replicate.** On the tuning networks, calibration and the gate gained about +0.035 for activation data;
  on the reported networks +0.007 at most. That is the reason the study uses separate seeds.
- **Rewiring is where mechanism wins.** Transfer needs the contexts to be similar. As the target moves away from the sources,
  predictors that know the target's own regulators (the oracle here; an inferred structure in practice, see
  `docs/HOLDOUT_RESEARCH.md` for how hard that is) pull ahead, and none of the calibration-only predictors helps.

## Limitations

One topology family (scale-free), N = 16, fixed efficiency 0.8 in knockdown mode, bulk steady states with 1% noise (no single
cells, dropout or count noise), three sources that are variants of one network, hyper-parameters tuned on a small set of
networks. The leaderboard methods this takes its names from use thousands of genes, protein embeddings, pretrained models and
single-cell statistics, none of which the simulator has; their real scores say nothing about these simplified versions.

## Reproduce and extend

```bash
python scripts/transfer_study.py                                  # activation data, 16 genes, 24 networks, rewiring 15%
python scripts/transfer_study.py --mode knockdown --rewire 0.30
python scripts/transfer_study.py --seed-offset 100 --networks 12  # the tuning networks
```
In the studio: tab 4, **Response Method: Cross-context transfer**. To add a predictor, write a
`fit(sources, target, design, control, train_columns) -> TransferModel` in `cross_context.py`, append a `TransferPredictor`
to `TRANSFER_PREDICTORS`, and add it to the parametrised contract tests in `tests/test_cross_context.py` (held-out columns
never read, seen columns leave-one-out, deterministic, finite on degenerate input).
