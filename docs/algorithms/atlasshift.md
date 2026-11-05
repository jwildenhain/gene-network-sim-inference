# AtlasShift (empirical cross-atlas transfer)

`atlasshift` · Response-matrix and transfer methods · status: extended

Predicts a perturbation's response from the same perturbation in other cell contexts (source atlases) by weighted averaging.

## How it works

Port of a Virtual Cell Challenge 2026 leaderboard entry (the docstring gives public rank #82, score 0.15456; not verified here). As a structure scorer it reads A[j, i] = response of i to perturbing j, fused over any registered source atlases.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `atlasshift`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native, no sources): 0.18 / 0.12 at 30 perturbed. Transfer across simulated contexts: see docs/CROSS_CONTEXT_TRANSFER.md.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- kaipengm2/Virtual-Cell-Challenge-2026
- Code: [`src/grn_sim/inference/atlasshift.py`](../../src/grn_sim/inference/atlasshift.py)
