# DeepRIG (graph autoencoder)

`deeprig` · Neural and latent-space methods · status: extended

Scores gene pairs with a graph autoencoder (GCN encoder, inner-product decoder) on a co-expression prior graph.

## How it works

Covariance-shrunk prior graph, low-rank bottleneck.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on every data model.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `deeprig`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native): 0.022 / 0.16 at 30 perturbed, 0.043 / 0.10 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- DeepRIG, PLoS Comput. Biol. 2024
- Code: [`src/grn_sim/inference/deeprig.py`](../../src/grn_sim/inference/deeprig.py)
