# Voting Ensemble

`vote` · Regression on perturbation data (thesis family) · status: core (thesis)

Keeps an edge only when S2G and G2S (and NIR on Hill data) agree on its presence and sign.

## How it works

Intersection of the members' supports: few false positives, lower recall than either member.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `vote`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (adapted): 0.27 / 0.15 at 30 perturbed, 0.11 / 0.18 at 100; about as good as its members, not better.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Wildenhain thesis section 5.1.7
- Code: [`src/grn_sim/inference/ensemble.py`](../../src/grn_sim/inference/ensemble.py)
