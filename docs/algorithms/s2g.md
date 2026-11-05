# Simple-to-General (S2G)

`s2g` · Regression on perturbation data (thesis family) · status: core (thesis)

Forward stepwise regression: for every gene, regulators are added one at a time until the description-length penalty stops improving.

## How it works

Solves A X = -U row by row. Starts from the column of X most correlated with the residual, swaps out negligible regressors and stops when N log(mean E^2) + k log N no longer drops. No fixed number of regulators.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `s2g`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens, 40 cells: direct AUPR 0.35, the best of the thesis family. Partial screens (adapted): 0.21 / 0.14 at 30 perturbed, 0.27 / 0.24 at 100. Needs the partial adapter.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Crampin / Judd & Mees; Wildenhain thesis ch. 3 (getvar.m, s2g.m)
- Code: [`src/grn_sim/inference/s2g.py`](../../src/grn_sim/inference/s2g.py)
