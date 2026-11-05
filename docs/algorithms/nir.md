# NIR (Network Identification by Multiple Regression)

`nir` · Regression on perturbation data (thesis family) · status: core (thesis)

Finds exactly K regulators per gene by a greedy forward search with the best few candidates kept at each step.

## How it works

Multiple linear regression on A X = -U with a top-D forward search, optional ridge and BIC model-order choice. For genes without a perturbation it regresses the gene's own response on the others, so it accepts partial screens as they are.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `nir`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.28. Partial (native): 0.27 / 0.29 at 30 perturbed, 0.37 / 0.33 at 100 (the best single method there).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Gardner et al., Science 2003 (nir_v3.m)
- Code: [`src/grn_sim/inference/nir.py`](../../src/grn_sim/inference/nir.py)
