# NOTEARS (continuous DAG)

`notears` · Causal and DAG methods · status: extended

Learns a directed acyclic graph by continuous optimisation with the constraint tr(exp(W o W)) - d = 0.

## How it works

min 1/2M ||(I - W) X + U||^2 + lambda |W|_1 with an augmented Lagrangian. Cannot represent feedback loops.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `notears`; listed by `/api/algorithms` and the studio.

## Measured here

Near chance: 0.03 / 0.10 at 30 perturbed, 0.013 / 0.007 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Zheng et al., NeurIPS 2018
- Code: [`src/grn_sim/inference/notears.py`](../../src/grn_sim/inference/notears.py)
