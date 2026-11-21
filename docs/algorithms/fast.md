# FAST (SVD inversion)

`fast` · Regression on perturbation data (thesis family) · status: core (thesis)

Inverts the response matrix with a truncated SVD and keeps the K largest coefficients per row.

## How it works

A0 = (X - U) V S^-1 U^T, then the top-K entries per row are refitted by least squares. A0 is I + A, so the MATLAB ranking spends one of the K slots of every row on the diagonal; the studio ranks by |A0 - I| and never selects the diagonal (`rank_without_identity`, off by default so the MATLAB-equivalence tests still hold).

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `fast`; listed by `/api/algorithms` and the studio.

## Measured here

Hill screens, 50 / 100 genes: F1 0.35 / 0.28 with the MATLAB ranking, 0.45 / 0.35 without the identity (NIR 0.45 / 0.37); on the linear engine it stays weak (0.16 / 0.05; NIR 0.27 / 0.15), see results/fast_diagnosis.md. Natively scores only the perturbed genes on partial screens (balanced 0.10-0.25, chance for unperturbed genes); adapted 0.06-0.22.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Farina et al. (fast.m, fast_modified.m)
- Code: [`src/grn_sim/inference/fast_svd.py`](../../src/grn_sim/inference/fast_svd.py)
