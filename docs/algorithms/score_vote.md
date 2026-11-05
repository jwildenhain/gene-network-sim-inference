# Score Vote (evidence-weighted)

`score_vote` · Regression on perturbation data (thesis family) · status: extended

S2G, G2S and optionally NIR combined through a signed evidence score per edge instead of a presence/sign vote.

## How it works

Each member contributes the t-statistic of the edge's coefficient, mapped to a z-score; the z-scores are summed with a correction for the members' correlation. Keeps how strong each call is.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `score_vote`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (adapted): 0.30 / 0.15 at 30 perturbed, 0.18 / 0.18 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- This repository
- Code: [`src/grn_sim/inference/score_vote.py`](../../src/grn_sim/inference/score_vote.py)
