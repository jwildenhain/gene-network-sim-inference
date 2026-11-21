# LLC (mean-shift equations)

`llc_mean` · Response-matrix and transfer methods · status: extended

Per-target regression of one gene's response on the responses of all other genes, over the experiments in which it was not perturbed.

## How it works

For a linear cyclic model and single-gene interventions the shift of gene j is the sum of B[i, j] times the shifts of the other genes, so each target is one ridge-regularised linear system; the confounder terms cancel in a shift. Top-K regulators per target. Hill (regulator-input) data uses the response of X + U as inspre does. Works on partial screens as they are. Only the mean-shift part of LLC: its covariance constraints need many cells per experiment.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `llc_mean`; listed by `/api/algorithms` and the studio.

## Measured here

Full screens, 100 genes, mechanistic network: direct AUPR 0.68 (cellbox 0.61, inspre 0.42, ridge_cv 0.39, nir 0.35); on three statistical generators it ties the best (0.35-0.49). Partial (native, 100 genes): balanced AUPR 0.46 / 0.56 / 0.62 / 0.67 at 6 / 20 / 50 / 75 perturbed (NIR 0.20 / 0.40 / 0.42 / 0.39). Real data: E. coli SOS 0.58 (random 0.57), Norman 16 TFs 0.26 (random 0.11), in the pack. results/paper_algorithms.md.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Hyttinen, Eberhardt & Hoyer, UAI 2012 (mean-shift subset)
- Code: [`src/grn_sim/inference/llc.py`](../../src/grn_sim/inference/llc.py)
