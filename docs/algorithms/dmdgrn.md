# DMD-GRN (Koopman / dynamic mode decomposition)

`dmdgrn` · Response-matrix and transfer methods · status: extended

Fits a linear operator from perturbation to response with a truncated SVD (controlled DMD).

## How it works

Delta X ~ -A^-1 U; best-fit operator A = -U W S^-1 V^T, off-diagonal entries discretised.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `dmdgrn`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native): 0.02 / 0.43 at 30 perturbed, 0.13 / 0.49 at 100 (strong on edges to unperturbed genes, weak among perturbed ones).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Schmid 2010; Brunton, Kutz et al.
- Code: [`src/grn_sim/inference/dmdgrn.py`](../../src/grn_sim/inference/dmdgrn.py)
