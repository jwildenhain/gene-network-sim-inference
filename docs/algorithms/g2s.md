# General-to-Specific (G2S)

`g2s` · Regression on perturbation data (thesis family) · status: core (thesis)

Backward elimination: starts from every gene as a regulator and drops the least useful one until AIC stops improving.

## How it works

Row-wise regression of A X = -U with the full basis, removing the column whose removal raises the squared error least. A fast exact engine (`_g2s_fast.py`) reproduces the MATLAB result and runs on a process pool for larger networks.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): needs `PartialAdapter` (`registry.build_partial`).
- Registry key: `g2s`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens, 40 cells: direct AUPR 0.08 with the leave-one-out shortcut (the weakest regression variant). Partial (adapted): 0.26 / 0.19 at 30 perturbed, 0.08 / 0.14 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Crampin et al.; Wildenhain thesis ch. 3 (gen2spec.m, g2s.m)
- Code: [`src/grn_sim/inference/g2s.py`](../../src/grn_sim/inference/g2s.py)
