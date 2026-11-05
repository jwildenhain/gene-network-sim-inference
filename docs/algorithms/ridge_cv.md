# RidgeCV (multi-output ridge)

`ridge_cv` · Regression on perturbation data (thesis family) · status: baseline / control

L2-regularised regression of every gene's perturbation on the responses, alpha chosen by cross-validation.

## How it works

Follows the Kaggle NeurIPS 2023 single-cell perturbation baseline. In additive fits the perturbed gene's own term is a regressor (without it the fit is near random; this was the default until 2026-11-04). The own-gene coefficient is not reported.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `ridge_cv`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.27 with the own-gene term, 0.03 without. Partial (adapted): 0.36 / 0.20 at 30 perturbed.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Kouagou (NeurIPS 2023 Open Problems, perturbation prediction)
- Code: [`src/grn_sim/inference/ridge_cv.py`](../../src/grn_sim/inference/ridge_cv.py)
