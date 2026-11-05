# By-Chance (random)

`by_chance` · Regression on perturbation data (thesis family) · status: null model

Random network with the given connectivity and activator/inhibitor ratio: the null every other method must beat.

## How it works

Bernoulli draws; ignores the data.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: no orientation (random).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `by_chance`; listed by `/api/algorithms` and the studio.

## Measured here

AUPR equals the base rate: 0.011 / 0.006 at 30 perturbed (500 genes).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Wildenhain thesis (by_chance.m)
- Code: [`src/grn_sim/inference/by_chance.py`](../../src/grn_sim/inference/by_chance.py)
