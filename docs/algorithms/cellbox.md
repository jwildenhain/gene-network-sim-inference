# CellBox (neural ODE)

`cellbox` · Neural and latent-space methods · status: extended

Fits x_i = phi(sum_j W_ij x_j + u_i) / alpha_i to the steady states with an L1 penalty on W.

## How it works

Needs PyTorch (a SciPy fallback exists for up to 200 genes). Inputs are rescaled by 1/std(X): log fold changes of ~0.1 otherwise leave tanh in its linear range and the fit learns nothing (AUPR 0.03-0.06 before, 0.45-0.60 after on 100-gene screens). Optional GPU.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both).
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `cellbox`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.45-0.60. Partial (native): 0.52 / 0.25 at 30 perturbed, 0.15 / 0.03 at 100 (varies between seeds).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- CellBox (Sander lab), Cell Systems
- Code: [`src/grn_sim/inference/cellbox.py`](../../src/grn_sim/inference/cellbox.py)
