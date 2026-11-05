# CPA-Lite (latent perturbation autoencoder)

`cpa_lite` · Neural and latent-space methods · status: extended

A small autoencoder whose latent space adds perturbation vectors; gene influence read from its Jacobian.

## How it works

Simplified from the compositional perturbation autoencoder; not the published model.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `cpa_lite`; listed by `/api/algorithms` and the studio.

## Measured here

At chance on structure recovery: 0.009 / 0.008 at 30 perturbed.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Lotfollahi et al., Mol. Syst. Biol. 2023 (simplified)
- Code: [`src/grn_sim/inference/cpa_lite.py`](../../src/grn_sim/inference/cpa_lite.py)
