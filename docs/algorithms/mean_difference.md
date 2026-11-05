# Mean difference

`mean_difference` · Response-matrix and transfer methods · status: baseline / control

Scores edge a -> j by the size of gene j's response when a is perturbed.

## How it works

The unnormalised effect matrix; a CausalBench challenge winner. Scores total, not direct, effects, so it is perfect against a total-effect truth by construction.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `mean_difference`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.25. Partial (native): 0.41 / 0.35 at 30 perturbed, 0.33 / 0.30 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- CausalBench (Chevalley et al.)
- Code: [`src/grn_sim/inference/interventional_baselines.py`](../../src/grn_sim/inference/interventional_baselines.py)
