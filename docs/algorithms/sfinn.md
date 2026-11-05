# SFINN-style scoring

`sfinn` · Neural and latent-space methods · status: extended

Perturbation sensitivity weighted by a shared-factor association between genes.

## How it works

A deterministic rule, not a neural network: |S| (1 + |G|) with S = -X pinv(U) and G the correlation after smoothing conditions over their nearest neighbours. On partial screens S is read from the perturbed genes' columns.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on every data model.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `sfinn`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native): 0.27 / 0.19 at 30 perturbed, 0.23 / 0.20 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- SFINN, Briefings in Bioinformatics 2024 (inspired by)
- Code: [`src/grn_sim/inference/sfinn.py`](../../src/grn_sim/inference/sfinn.py)
