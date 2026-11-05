# Rank consensus

`interventional_consensus` · Experimental (cell-level, not in the registry) · status: experimental (not in the registry)

Average of the rank-normalised scores of several interventional-aware methods (inspre, mean difference, bicycle-lite, DOTEARS-lite).

## How it works

Borda count over the scorable pairs; worth keeping only if it beats its best member.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]` (direct effects among the perturbed genes, extended to the unperturbed ones).
- Not registered: called from Python on a calibrated effect estimate (`grn_sim.perturbseq.ace.estimate_ace`).

## Measured here

100-gene screens: direct AUPR 0.51 on raw effects (best member 0.40) but 0.35 on denoised ones (best member 0.31). Partial: 0.36 / 0.23 at 30 perturbed, 0.48 / 0.27 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- -
- Code: [`src/grn_sim/inference/interventional_consensus.py`](../../src/grn_sim/inference/interventional_consensus.py)
