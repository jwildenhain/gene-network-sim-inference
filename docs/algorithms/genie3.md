# GENIE3 (random forest)

`genie3` · Co-expression and statistical baselines · status: extended

Predicts every gene from all others with tree ensembles and ranks regulators by feature importance.

## How it works

Extra-trees per target gene; the sign comes from the correlation. Winner of DREAM4/5.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `genie3`; listed by `/api/algorithms` and the studio.

## Measured here

Norman 16 TFs: AUPR 0.27 (random 0.11, the highest there). Partial (native): 0.007 / 0.24 at 30 perturbed.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Huynh-Thu et al., PLoS ONE 2010
- Code: [`src/grn_sim/inference/genie3.py`](../../src/grn_sim/inference/genie3.py)
