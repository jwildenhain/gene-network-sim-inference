# DOTEARS-lite (interventional DAG)

`dotears_lite` · Experimental (cell-level, not in the registry) · status: experimental (not in the registry)

NOTEARS with the intervened gene's equation dropped in its own cells and per-gene noise variances.

## How it works

Linear SEM with the acyclicity constraint, free entries limited to inspre's strongest candidates (optionally all pairs).

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]` (direct effects among the perturbed genes, extended to the unperturbed ones).
- Not registered: called from Python on a calibrated effect estimate (`grn_sim.perturbseq.ace.estimate_ace`).

## Measured here

100-gene screens: direct AUPR 0.40 (0.41 with all pairs free). Partial: 0.32 / 0.13 at 30 perturbed.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- DOTEARS (NOTEARS with interventions); NOTEARS: Zheng et al., NeurIPS 2018
- Code: [`src/grn_sim/inference/dotears.py`](../../src/grn_sim/inference/dotears.py)
