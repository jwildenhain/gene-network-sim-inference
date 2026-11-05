# bicycle-lite (steady-state OU likelihood)

`bicycle_lite` · Experimental (cell-level, not in the registry) · status: experimental (not in the registry)

Maximum-likelihood direct effects from per-node mean shifts and one pooled covariance of an Ornstein-Uhlenbeck steady state.

## How it works

Mean shifts -delta_a T[a, :] with T = (I - G)^-1, covariance from the Lyapunov equation, L1 penalty, L-BFGS-B from an inspre start. No latent layer. The covariance term only helps at a small weight on Hill-kinetics data. Needs `estimate_ace` output.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]` (direct effects among the perturbed genes, extended to the unperturbed ones).
- Not registered: called from Python on a calibrated effect estimate (`grn_sim.perturbseq.ace.estimate_ace`).

## Measured here

100-gene screens: direct AUPR 0.39. Partial: 0.34 / 0.12 at 30 perturbed. Practical to about 250 genes.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Rohbeck et al., CLeaR 2024 (bicycle; simplified)
- Code: [`src/grn_sim/inference/bicycle_lite.py`](../../src/grn_sim/inference/bicycle_lite.py)
