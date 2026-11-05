# GRNBoost-lite (boosted trees)

`grnboost` · Co-expression and statistical baselines · status: baseline / control

Gradient-boosted trees per target gene, regulators ranked by importance (the GRNBoost2 recipe, with scikit-learn).

## How it works

Not the Dask implementation of GRNBoost2. Many candidate genes and few conditions limit it.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on every data model.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `grnboost`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.06. Partial: 0.011 / 0.036 at 30 perturbed.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Moerman et al., Bioinformatics 2019
- Code: [`src/grn_sim/inference/interventional_baselines.py`](../../src/grn_sim/inference/interventional_baselines.py)
