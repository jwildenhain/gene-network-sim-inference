# Network Deconvolution

`netdeconv` · Response-matrix and transfer methods · status: extended

Removes indirect transitive effects from an observed dependency matrix: G_dir = G_obs (I + G_obs)^-1.

## How it works

Applies G_dir = G_obs (I + G_obs)^-1 to the observed dependency matrix.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `netdeconv`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (adapted): 0.14 / 0.007 at 30 perturbed, so it scores only edges among perturbed genes.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Feizi et al., Nat. Biotechnol. 2013
- Code: [`src/grn_sim/inference/network_deconv.py`](../../src/grn_sim/inference/network_deconv.py)
