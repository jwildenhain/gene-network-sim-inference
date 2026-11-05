# CausalGRN

`causalgrn` · Causal and DAG methods · status: extended

Builds an undirected skeleton by partial correlation and directs edges with the interventions (do-calculus style).

## How it works

First-order partial correlations remove edges explained by a third gene; the perturbed gene gives the direction.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on every data model.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `causalgrn`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native): 0.18 / 0.46 at 30 perturbed, 0.18 / 0.45 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Causal discovery for perturbation screens (partial correlation + intervention orientation)
- Code: [`src/grn_sim/inference/causalgrn.py`](../../src/grn_sim/inference/causalgrn.py)
