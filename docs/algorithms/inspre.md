# inspre (sparse inversion of the response matrix)

`inspre` · Response-matrix and transfer methods · status: extended

Turns the matrix of total knockdown effects into direct effects by finding a sparse inverse of it; handles feedback loops.

## How it works

R[a, j] = effect of perturbing a on gene j relative to its own effect. ADMM finds V with R V = I and L1 sparsity; direct effects G = I - V diag(1/diag V). Per-node weights, optional per-edge penalties (priors) and per-entry weights. On partial screens it fits the perturbed block and scores edges to unperturbed genes as V R. Assumes stable feedback (reports `unstable` otherwise).

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `inspre`; listed by `/api/algorithms` and the studio.

## Measured here

100-gene screens: direct AUPR 0.24 raw, 0.31 on denoised effects. Partial (native): 0.53 / 0.22 at 30 perturbed, 0.28 / 0.28 at 100. Real data: E. coli SOS 0.63 (random 0.57), Norman 16 TFs 0.23 (random 0.11).

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Brown, Morris, Lappalainen & Knowles (Nat. Commun. 2024)
- Code: [`src/grn_sim/inference/inspre.py`](../../src/grn_sim/inference/inspre.py)
