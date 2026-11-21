"""
DCD-FG, linear-Gaussian version (Lopez, Hao, Regev et al., "Large-scale differentiable causal discovery of factor graphs",
NeurIPS 2022).

Model: ``x_j = sum_i W[i, j] x_i + sigma_j eps_j`` with the weight matrix factorised as ``W = U V^T`` (rank ``r``, zero
diagonal), fitted by gradient descent on the *interventional* negative log-likelihood: a cell whose gene ``g`` was perturbed
contributes no term for ``g`` (its mechanism was replaced), every other gene contributes its Gaussian residual. Acyclicity is
enforced with the NOTEARS constraint ``h(W) = tr(exp(W o W)) - n = 0`` by an augmented Lagrangian, plus an L1 penalty.

Differences from the paper's code: Gaussian likelihood on log-normalised expression (the paper also has a negative-binomial
model), a dense matrix exponential for the constraint (the paper uses a low-rank trick that scales to thousands of genes),
and full-batch Adam. Needs torch. Output: ``A_est[regulator, target] = W``.
"""
from __future__ import annotations

import time
from typing import Any, List

import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult


class DCDFGLite(BaseInferenceAlgorithm):
    def __init__(self, k_interactions: int = 3, rank: int = 10, l1: float = 0.02, outer: int = 5, inner: int = 300,
                 lr: float = 0.02, seed: int = 0):
        super().__init__(name="dcdfg_lite")
        self.k_interactions = k_interactions
        self.rank, self.l1, self.outer, self.inner, self.lr, self.seed = rank, l1, outer, inner, lr, seed

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        raise NotImplementedError("DCD-FG fits cells: use fit_adata(adata, gene_names)")

    def fit_adata(self, adata: Any, gene_names: List[str], perturbation_key: str = "perturbation",
                  control_label: str = "non-targeting") -> InferenceResult:
        import torch
        from grn_sim.perturbseq.ace import log_normalise

        t0 = time.time()
        torch.manual_seed(self.seed)
        index = {g: i for i, g in enumerate(gene_names)}
        n = len(gene_names)
        Y = log_normalise(adata.X).astype(np.float32)
        lab = adata.obs[perturbation_key].astype(str).to_numpy()
        ctrl = lab == control_label
        mu, sd = Y[ctrl].mean(axis=0), Y[ctrl].std(axis=0) + 1e-3
        Y = (Y - mu) / sd
        M = np.ones_like(Y)                                  # 1 = this gene's mechanism is observed in this cell
        for c, g in enumerate(lab):
            if g in index:
                M[c, index[g]] = 0.0
        Yt, Mt = torch.tensor(Y), torch.tensor(M)
        r = min(self.rank, n)
        Uf = torch.nn.Parameter(0.1 * torch.randn(n, r))
        Vf = torch.nn.Parameter(0.1 * torch.randn(n, r))
        log_s = torch.nn.Parameter(torch.zeros(n))
        eye = torch.eye(n)
        opt = torch.optim.Adam([Uf, Vf, log_s], lr=self.lr)
        alpha, rho = 0.0, 1.0
        cells = float(len(Y))

        def weights():
            return (Uf @ Vf.T) * (1.0 - eye)

        for _ in range(self.outer):
            for _ in range(self.inner):
                opt.zero_grad()
                W = weights()
                resid = Yt - Yt @ W
                nll = (Mt * (0.5 * resid ** 2 * torch.exp(-2 * log_s) + log_s)).sum() / cells
                h = torch.trace(torch.matrix_exp(W * W)) - n
                loss = nll + self.l1 * W.abs().sum() + alpha * h + 0.5 * rho * h * h
                loss.backward()
                opt.step()
            with torch.no_grad():
                h_val = float(torch.trace(torch.matrix_exp(weights() ** 2)) - n)
            alpha += rho * h_val
            rho = min(rho * 4.0, 1e4)
            if h_val < 1e-6:
                break
        W = weights().detach().numpy().astype(float)
        adj = np.zeros((n, n), dtype=int)
        mags = np.abs(W)
        for rr in range(n):
            top = np.argsort(-mags[rr])[: self.k_interactions]
            adj[rr, top[mags[rr, top] > 1e-8]] = np.sign(W[rr, top[mags[rr, top] > 1e-8]]).astype(int)
        return InferenceResult(self.name, W, adj, time.time() - t0, {"acyclicity_h": h_val, "rank": r})
