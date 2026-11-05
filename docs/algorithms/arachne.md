# ARACNE (mutual information + DPI)

`arachne` · Co-expression and statistical baselines · status: extended

Keeps edges with high mutual information and removes the weakest edge of every triangle (data processing inequality).

## How it works

Gaussian-copula mutual information with Ledoit-Wolf shrinkage; the direction is assigned with the perturbation data.

## Interface

- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.
- Output: `A_est[regulator, target]`.
- Partial screens (m perturbed genes of n measured): runs on a partial screen as it is.
- Registry key: `arachne`; listed by `/api/algorithms` and the studio.

## Measured here

Partial (native): 0.012 / 0.45 at 30 perturbed, 0.13 / 0.37 at 100.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Margolin et al., Nat. Protoc. 2006; Basso et al. 2005
- Code: [`src/grn_sim/inference/arachne.py`](../../src/grn_sim/inference/arachne.py)
