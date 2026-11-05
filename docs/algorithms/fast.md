# FAST (SVD inversion)

`fast` · Regression on perturbation data (thesis family) · status: core (thesis)

Inverts the response matrix with a truncated SVD and keeps the K largest coefficients per row.

## How it works

A0 = (X - U) V S^-1 U^T, then the top-K entries per row are refitted by least squares.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `fast`; listed by `/api/algorithms` and the studio.

## Measured here

Natively scores only the perturbed genes (0.30 / 0.006 at 30 perturbed, which is chance for unperturbed genes); adapted 0.07 / 0.05.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Farina et al. (fast.m, fast_modified.m)
- Code: [`src/grn_sim/inference/fast_svd.py`](../../src/grn_sim/inference/fast_svd.py)
