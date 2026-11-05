"""
Short, human-written descriptions of every inference algorithm (one entry per registry key, plus the experimental methods that
are not registered). ``scripts/generate_algorithm_docs.py`` turns them, together with what the registry knows (orientation,
partial-screen support), into ``docs/algorithms/``; ``tests/test_algorithm_docs.py`` fails when a registered algorithm has no
description or the committed pages are out of date.

``measured`` quotes numbers from this repository's own benchmarks (simulator screens, 3-4 seeds, wide spreads: read them as
orders of magnitude, see the linked result files), never from the original papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AlgorithmDescription:
    key: str
    name: str
    family: str
    status: str                 # core | extended | baseline | null | experimental
    summary: str
    how: str
    reference: str
    module: str
    measured: str


FAMILIES = [
    "Regression on perturbation data (thesis family)",
    "Response-matrix and transfer methods",
    "Co-expression and statistical baselines",
    "Causal and DAG methods",
    "Neural and latent-space methods",
    "Experimental (cell-level, not in the registry)",
]

_P = "500 genes, 30 perturbed (3 seeds, edges among perturbed genes / edges to unperturbed genes)"

D: List[AlgorithmDescription] = [
    # ---- thesis regression family ---------------------------------------------------------------------------------------
    AlgorithmDescription(
        "s2g", "Simple-to-General (S2G)", FAMILIES[0], "core",
        "Forward stepwise regression: for every gene, regulators are added one at a time until the description-length penalty stops improving.",
        "Solves A X = -U row by row. Starts from the column of X most correlated with the residual, swaps out negligible regressors "
        "and stops when N log(mean E^2) + k log N no longer drops. No fixed number of regulators.",
        "Crampin / Judd & Mees; Wildenhain thesis ch. 3 (getvar.m, s2g.m)", "inference/s2g.py",
        "100-gene screens, 40 cells: direct AUPR 0.35, the best of the thesis family. Partial screens (adapted): 0.21 / 0.14 at 30 perturbed, "
        "0.27 / 0.24 at 100. Needs the partial adapter."),
    AlgorithmDescription(
        "g2s", "General-to-Specific (G2S)", FAMILIES[0], "core",
        "Backward elimination: starts from every gene as a regulator and drops the least useful one until AIC stops improving.",
        "Row-wise regression of A X = -U with the full basis, removing the column whose removal raises the squared error least. "
        "A fast exact engine (`_g2s_fast.py`) reproduces the MATLAB result and runs on a process pool for larger networks.",
        "Crampin et al.; Wildenhain thesis ch. 3 (gen2spec.m, g2s.m)", "inference/g2s.py",
        "100-gene screens, 40 cells: direct AUPR 0.08 with the leave-one-out shortcut (the weakest regression variant). Partial (adapted): "
        "0.26 / 0.19 at 30 perturbed, 0.08 / 0.14 at 100."),
    AlgorithmDescription(
        "vote", "Voting Ensemble", FAMILIES[0], "core",
        "Keeps an edge only when S2G and G2S (and NIR on Hill data) agree on its presence and sign.",
        "Intersection of the members' supports: few false positives, lower recall than either member.",
        "Wildenhain thesis section 5.1.7", "inference/ensemble.py",
        "Partial (adapted): 0.27 / 0.15 at 30 perturbed, 0.11 / 0.18 at 100; about as good as its members, not better."),
    AlgorithmDescription(
        "score_vote", "Score Vote (evidence-weighted)", FAMILIES[0], "extended",
        "S2G, G2S and optionally NIR combined through a signed evidence score per edge instead of a presence/sign vote.",
        "Each member contributes the t-statistic of the edge's coefficient, mapped to a z-score; the z-scores are summed with a correction "
        "for the members' correlation. Keeps how strong each call is.",
        "This repository", "inference/score_vote.py",
        "Partial (adapted): 0.30 / 0.15 at 30 perturbed, 0.18 / 0.18 at 100."),
    AlgorithmDescription(
        "nir", "NIR (Network Identification by Multiple Regression)", FAMILIES[0], "core",
        "Finds exactly K regulators per gene by a greedy forward search with the best few candidates kept at each step.",
        "Multiple linear regression on A X = -U with a top-D forward search, optional ridge and BIC model-order choice. For genes without "
        "a perturbation it regresses the gene's own response on the others, so it accepts partial screens as they are.",
        "Gardner et al., Science 2003 (nir_v3.m)", "inference/nir.py",
        "100-gene screens: direct AUPR 0.28. Partial (native): 0.27 / 0.29 at 30 perturbed, 0.37 / 0.33 at 100 (the best single method there)."),
    AlgorithmDescription(
        "fast", "FAST (SVD inversion)", FAMILIES[0], "core",
        "Inverts the response matrix with a truncated SVD and keeps the K largest coefficients per row.",
        "A0 = (X - U) V S^-1 U^T, then the top-K entries per row are refitted by least squares.",
        "Farina et al. (fast.m, fast_modified.m)", "inference/fast_svd.py",
        "Natively scores only the perturbed genes (0.30 / 0.006 at 30 perturbed, which is chance for unperturbed genes); adapted 0.07 / 0.05."),
    AlgorithmDescription(
        "ridge_cv", "RidgeCV (multi-output ridge)", FAMILIES[0], "baseline",
        "L2-regularised regression of every gene's perturbation on the responses, alpha chosen by cross-validation.",
        "Follows the Kaggle NeurIPS 2023 single-cell perturbation baseline. In additive fits the perturbed gene's own term is a regressor "
        "(without it the fit is near random; this was the default until 2026-11-04). The own-gene coefficient is not reported.",
        "Kouagou (NeurIPS 2023 Open Problems, perturbation prediction)", "inference/ridge_cv.py",
        "100-gene screens: direct AUPR 0.27 with the own-gene term, 0.03 without. Partial (adapted): 0.36 / 0.20 at 30 perturbed."),
    AlgorithmDescription(
        "by_chance", "By-Chance (random)", FAMILIES[0], "null",
        "Random network with the given connectivity and activator/inhibitor ratio: the null every other method must beat.",
        "Bernoulli draws; ignores the data.", "Wildenhain thesis (by_chance.m)", "inference/by_chance.py",
        "AUPR equals the base rate: 0.011 / 0.006 at 30 perturbed (500 genes)."),
    # ---- response-matrix and transfer ------------------------------------------------------------------------------------
    AlgorithmDescription(
        "inspre", "inspre (sparse inversion of the response matrix)", FAMILIES[1], "extended",
        "Turns the matrix of total knockdown effects into direct effects by finding a sparse inverse of it; handles feedback loops.",
        "R[a, j] = effect of perturbing a on gene j relative to its own effect. ADMM finds V with R V = I and L1 sparsity; "
        "direct effects G = I - V diag(1/diag V). Per-node weights, optional per-edge penalties (priors) and per-entry weights. On partial "
        "screens it fits the perturbed block and scores edges to unperturbed genes as V R. Assumes stable feedback (reports `unstable` otherwise).",
        "Brown, Morris, Lappalainen & Knowles (Nat. Commun. 2024)", "inference/inspre.py",
        "100-gene screens: direct AUPR 0.24 raw, 0.31 on denoised effects. Partial (native): 0.53 / 0.22 at 30 perturbed, 0.28 / 0.28 at 100. "
        "Real data: E. coli SOS 0.63 (random 0.57), Norman 16 TFs 0.23 (random 0.11)."),
    AlgorithmDescription(
        "mean_difference", "Mean difference", FAMILIES[1], "baseline",
        "Scores edge a -> j by the size of gene j's response when a is perturbed.",
        "The unnormalised effect matrix; a CausalBench challenge winner. Scores total, not direct, effects, so it is perfect against a "
        "total-effect truth by construction.", "CausalBench (Chevalley et al.)", "inference/interventional_baselines.py",
        "100-gene screens: direct AUPR 0.25. Partial (native): 0.41 / 0.35 at 30 perturbed, 0.33 / 0.30 at 100."),
    AlgorithmDescription(
        "netdeconv", "Network Deconvolution", FAMILIES[1], "extended",
        "Removes indirect transitive effects from an observed dependency matrix: G_dir = G_obs (I + G_obs)^-1.",
        "Applies G_dir = G_obs (I + G_obs)^-1 to the observed dependency matrix.", "Feizi et al., Nat. Biotechnol. 2013",
        "inference/network_deconv.py", "Partial (adapted): 0.14 / 0.007 at 30 perturbed, so it scores only edges among perturbed genes."),
    AlgorithmDescription(
        "dmdgrn", "DMD-GRN (Koopman / dynamic mode decomposition)", FAMILIES[1], "extended",
        "Fits a linear operator from perturbation to response with a truncated SVD (controlled DMD).",
        "Delta X ~ -A^-1 U; best-fit operator A = -U W S^-1 V^T, off-diagonal entries discretised.",
        "Schmid 2010; Brunton, Kutz et al.", "inference/dmdgrn.py",
        "Partial (native): 0.02 / 0.43 at 30 perturbed, 0.13 / 0.49 at 100 (strong on edges to unperturbed genes, weak among perturbed ones)."),
    AlgorithmDescription(
        "atlasshift", "AtlasShift (empirical cross-atlas transfer)", FAMILIES[1], "extended",
        "Predicts a perturbation's response from the same perturbation in other cell contexts (source atlases) by weighted averaging.",
        "Port of a Virtual Cell Challenge 2026 leaderboard entry (the docstring gives public rank #82, score 0.15456; not verified here). "
        "As a structure scorer it reads A[j, i] = response of i to perturbing j, fused over any registered source atlases.",
        "kaipengm2/Virtual-Cell-Challenge-2026", "inference/atlasshift.py",
        "Partial (native, no sources): 0.18 / 0.12 at 30 perturbed. Transfer across simulated contexts: see docs/CROSS_CONTEXT_TRANSFER.md."),
    # ---- co-expression / statistical ---------------------------------------------------------------------------------------
    AlgorithmDescription(
        "pearson", "Pearson |r|", FAMILIES[2], "baseline",
        "Absolute correlation of two genes across the conditions.", "Symmetric, ignores which gene was perturbed; the cheapest control.",
        "-", "inference/interventional_baselines.py",
        "100-gene screens: direct AUPR 0.16. Partial: 0.19 / 0.51 at 30 perturbed (high on edges to unperturbed genes)."),
    AlgorithmDescription(
        "genie3", "GENIE3 (random forest)", FAMILIES[2], "extended",
        "Predicts every gene from all others with tree ensembles and ranks regulators by feature importance.",
        "Extra-trees per target gene; the sign comes from the correlation. Winner of DREAM4/5.",
        "Huynh-Thu et al., PLoS ONE 2010", "inference/genie3.py",
        "Norman 16 TFs: AUPR 0.27 (random 0.11, the highest there). Partial (native): 0.007 / 0.24 at 30 perturbed."),
    AlgorithmDescription(
        "grnboost", "GRNBoost-lite (boosted trees)", FAMILIES[2], "baseline",
        "Gradient-boosted trees per target gene, regulators ranked by importance (the GRNBoost2 recipe, with scikit-learn).",
        "Not the Dask implementation of GRNBoost2. Many candidate genes and few conditions limit it.",
        "Moerman et al., Bioinformatics 2019", "inference/interventional_baselines.py",
        "100-gene screens: direct AUPR 0.06. Partial: 0.011 / 0.036 at 30 perturbed."),
    AlgorithmDescription(
        "arachne", "ARACNE (mutual information + DPI)", FAMILIES[2], "extended",
        "Keeps edges with high mutual information and removes the weakest edge of every triangle (data processing inequality).",
        "Gaussian-copula mutual information with Ledoit-Wolf shrinkage; the direction is assigned with the perturbation data.",
        "Margolin et al., Nat. Protoc. 2006; Basso et al. 2005", "inference/arachne.py",
        "Partial (native): 0.012 / 0.45 at 30 perturbed, 0.13 / 0.37 at 100."),
    AlgorithmDescription(
        "gene_features", "Gene-level features (hubness only)", FAMILIES[2], "baseline",
        "Scores a -> j as (how much a moves anything) x (how much j moves under anything), ignoring the specific pair.",
        "A control that shows how much of a score is gene-level hubness and not pair-specific.", "-",
        "inference/interventional_baselines.py", "100-gene screens: direct AUPR 0.05 but 0.3-0.5 against total effects."),
    # ---- causal / DAG ------------------------------------------------------------------------------------------------------
    AlgorithmDescription(
        "causalgrn", "CausalGRN", FAMILIES[3], "extended",
        "Builds an undirected skeleton by partial correlation and directs edges with the interventions (do-calculus style).",
        "First-order partial correlations remove edges explained by a third gene; the perturbed gene gives the direction.",
        "Causal discovery for perturbation screens (partial correlation + intervention orientation)", "inference/causalgrn.py",
        "Partial (native): 0.18 / 0.46 at 30 perturbed, 0.18 / 0.45 at 100."),
    AlgorithmDescription(
        "notears", "NOTEARS (continuous DAG)", FAMILIES[3], "extended",
        "Learns a directed acyclic graph by continuous optimisation with the constraint tr(exp(W o W)) - d = 0.",
        "min 1/2M ||(I - W) X + U||^2 + lambda |W|_1 with an augmented Lagrangian. Cannot represent feedback loops.",
        "Zheng et al., NeurIPS 2018", "inference/notears.py",
        "Near chance: 0.03 / 0.10 at 30 perturbed, 0.013 / 0.007 at 100."),
    # ---- neural / latent -----------------------------------------------------------------------------------------------------
    AlgorithmDescription(
        "cellbox", "CellBox (neural ODE)", FAMILIES[4], "extended",
        "Fits x_i = phi(sum_j W_ij x_j + u_i) / alpha_i to the steady states with an L1 penalty on W.",
        "Needs PyTorch (a SciPy fallback exists for up to 200 genes). Inputs are rescaled by 1/std(X): log fold changes of ~0.1 otherwise leave "
        "tanh in its linear range and the fit learns nothing (AUPR 0.03-0.06 before, 0.45-0.60 after on 100-gene screens). Optional GPU.",
        "CellBox (Sander lab), Cell Systems", "inference/cellbox.py",
        "100-gene screens: direct AUPR 0.45-0.60. Partial (native): 0.52 / 0.25 at 30 perturbed, 0.15 / 0.03 at 100 (varies between seeds)."),
    AlgorithmDescription(
        "deeprig", "DeepRIG (graph autoencoder)", FAMILIES[4], "extended",
        "Scores gene pairs with a graph autoencoder (GCN encoder, inner-product decoder) on a co-expression prior graph.",
        "Covariance-shrunk prior graph, low-rank bottleneck.", "DeepRIG, PLoS Comput. Biol. 2024", "inference/deeprig.py",
        "Partial (native): 0.022 / 0.16 at 30 perturbed, 0.043 / 0.10 at 100."),
    AlgorithmDescription(
        "sfinn", "SFINN-style scoring", FAMILIES[4], "extended",
        "Perturbation sensitivity weighted by a shared-factor association between genes.",
        "A deterministic rule, not a neural network: |S| (1 + |G|) with S = -X pinv(U) and G the correlation after smoothing conditions over "
        "their nearest neighbours. On partial screens S is read from the perturbed genes' columns.",
        "SFINN, Briefings in Bioinformatics 2024 (inspired by)", "inference/sfinn.py",
        "Partial (native): 0.27 / 0.19 at 30 perturbed, 0.23 / 0.20 at 100."),
    AlgorithmDescription(
        "cpa_lite", "CPA-Lite (latent perturbation autoencoder)", FAMILIES[4], "extended",
        "A small autoencoder whose latent space adds perturbation vectors; gene influence read from its Jacobian.",
        "Simplified from the compositional perturbation autoencoder; not the published model.",
        "Lotfollahi et al., Mol. Syst. Biol. 2023 (simplified)", "inference/cpa_lite.py",
        "At chance on structure recovery: 0.009 / 0.008 at 30 perturbed."),
    # ---- experimental (not registered) ----------------------------------------------------------------------------------------
    AlgorithmDescription(
        "bicycle_lite", "bicycle-lite (steady-state OU likelihood)", FAMILIES[5], "experimental",
        "Maximum-likelihood direct effects from per-node mean shifts and one pooled covariance of an Ornstein-Uhlenbeck steady state.",
        "Mean shifts -delta_a T[a, :] with T = (I - G)^-1, covariance from the Lyapunov equation, L1 penalty, L-BFGS-B from an inspre start. "
        "No latent layer. The covariance term only helps at a small weight on Hill-kinetics data. Needs `estimate_ace` output.",
        "Rohbeck et al., CLeaR 2024 (bicycle; simplified)", "inference/bicycle_lite.py",
        "100-gene screens: direct AUPR 0.39. Partial: 0.34 / 0.12 at 30 perturbed. Practical to about 250 genes."),
    AlgorithmDescription(
        "dotears_lite", "DOTEARS-lite (interventional DAG)", FAMILIES[5], "experimental",
        "NOTEARS with the intervened gene's equation dropped in its own cells and per-gene noise variances.",
        "Linear SEM with the acyclicity constraint, free entries limited to inspre's strongest candidates (optionally all pairs).",
        "DOTEARS (NOTEARS with interventions); NOTEARS: Zheng et al., NeurIPS 2018", "inference/dotears.py",
        "100-gene screens: direct AUPR 0.40 (0.41 with all pairs free). Partial: 0.32 / 0.13 at 30 perturbed."),
    AlgorithmDescription(
        "interventional_consensus", "Rank consensus", FAMILIES[5], "experimental",
        "Average of the rank-normalised scores of several interventional-aware methods (inspre, mean difference, bicycle-lite, DOTEARS-lite).",
        "Borda count over the scorable pairs; worth keeping only if it beats its best member.", "-",
        "inference/interventional_consensus.py",
        "100-gene screens: direct AUPR 0.51 on raw effects (best member 0.40) but 0.35 on denoised ones (best member 0.31). "
        "Partial: 0.36 / 0.23 at 30 perturbed, 0.48 / 0.27 at 100."),
]

BY_KEY: Dict[str, AlgorithmDescription] = {d.key: d for d in D}
