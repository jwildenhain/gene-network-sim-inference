# Gene-level features (hubness only)

`gene_features` · Co-expression and statistical baselines · status: baseline / control

Scores a -> j as (how much a moves anything) x (how much j moves under anything), ignoring the specific pair.

## How it works

A control that shows how much of a score is gene-level hubness and not pair-specific.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `gene_features`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.05 but 0.3-0.5 against total effects.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- -
- Code: [`src/grn_sim/inference/interventional_baselines.py`](../../src/grn_sim/inference/interventional_baselines.py)
