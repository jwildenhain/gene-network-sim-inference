# DCD-FG-lite (low-rank interventional DAG)

`dcdfg_lite` · Experimental (cell-level, not in the registry) · status: experimental (not in the registry)

Linear-Gaussian version of DCD-FG: a low-rank factor-graph weight matrix fitted by the interventional likelihood of single cells.

## How it works

x_j = sum_i W[i, j] x_i + noise with W = U V^T (zero diagonal); a cell whose gene g was perturbed contributes no term for g. NOTEARS acyclicity by augmented Lagrangian, L1 penalty, full-batch Adam. Fits cells, not an (X, U) pair, so it cannot be registered. Needs torch.

## Interface

- Input: `fit_adata(adata, gene_names)`: the cells themselves (the registry's `fit(X, U)` raises `NotImplementedError`).
- Output: `A_est[regulator, target]` (direct effects among the perturbed genes, extended to the unperturbed ones where the method does so).
- Not registered: called from Python on cells (`fit_adata`).

## Measured here

100-gene screens: direct AUPR 0.06-0.25, between NOTEARS and the other methods; about 31 s per fit at 100 genes (dense matrix exponential), so it was not run above 150. results/paper_algorithms.md.

Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).

## Reference and code

- Lopez, Hao, Regev et al., NeurIPS 2022 (simplified)
- Code: [`src/grn_sim/inference/experimental/dcdfg.py`](../../src/grn_sim/inference/experimental/dcdfg.py)
