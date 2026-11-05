# Pearson |r|

`pearson` · Co-expression and statistical baselines · status: baseline / control

Absolute correlation of two genes across the conditions.

## How it works

Symmetric, ignores which gene was perturbed; the cheapest control.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `pearson`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.16. Partial: 0.19 / 0.51 at 30 perturbed (high on edges to unperturbed genes).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- -
- Code: [`src/grn_sim/inference/interventional_baselines.py`](../../src/grn_sim/inference/interventional_baselines.py)
