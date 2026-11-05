# Inference algorithms

One page per algorithm. All of them take the same input, `fit(X, U)`, and are scored against the simulator's ground truth; see [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md) for how they compare and [`docs/PERTURBSEQ.md`](../PERTURBSEQ.md) for the screens they are run on.

- **Input.** X is genes × perturbations (responses, e.g. log fold changes), U is zero except for each perturbed gene's own response. Fewer perturbation columns than genes is a *partial screen*.
- **Output.** `A_est` (continuous scores) and `adjacency_pred` (signed -1/0/+1). The orientation differs between algorithms and is normalised for scoring by `grn_sim.inference.orientation`.
- **Partial screens.** `native` algorithms accept them as they are; `adapter` ones need `registry.build_partial` (`PartialAdapter`: fit on the perturbed block, extended to the unperturbed genes).
- **Adding one.** Write the class, `register(AlgorithmSpec(...))` in `inference/registry.py`, add a description in `inference/descriptions.py`, run `python scripts/generate_algorithm_docs.py`. A test fails until all three exist.

## Regression on perturbation data (thesis family)

| algorithm | status | partial | in one line |
|---|---|---|---|
| [Simple-to-General (S2G)](s2g.md) | core | adapter | Forward stepwise regression: for every gene, regulators are added one at a time until the description-length penalty stops improving. |
| [General-to-Specific (G2S)](g2s.md) | core | adapter | Backward elimination: starts from every gene as a regulator and drops the least useful one until AIC stops improving. |
| [Voting Ensemble](vote.md) | core | adapter | Keeps an edge only when S2G and G2S (and NIR on Hill data) agree on its presence and sign. |
| [Score Vote (evidence-weighted)](score_vote.md) | extended | adapter | S2G, G2S and optionally NIR combined through a signed evidence score per edge instead of a presence/sign vote. |
| [NIR (Network Identification by Multiple Regression)](nir.md) | core | native | Finds exactly K regulators per gene by a greedy forward search with the best few candidates kept at each step. |
| [FAST (SVD inversion)](fast.md) | core | adapter | Inverts the response matrix with a truncated SVD and keeps the K largest coefficients per row. |
| [RidgeCV (multi-output ridge)](ridge_cv.md) | baseline / control | adapter | L2-regularised regression of every gene's perturbation on the responses, alpha chosen by cross-validation. |
| [By-Chance (random)](by_chance.md) | null model | native | Random network with the given connectivity and activator/inhibitor ratio: the null every other method must beat. |

## Response-matrix and transfer methods

| algorithm | status | partial | in one line |
|---|---|---|---|
| [inspre (sparse inversion of the response matrix)](inspre.md) | extended | native | Turns the matrix of total knockdown effects into direct effects by finding a sparse inverse of it; handles feedback loops. |
| [Mean difference](mean_difference.md) | baseline / control | native | Scores edge a -> j by the size of gene j's response when a is perturbed. |
| [Network Deconvolution](netdeconv.md) | extended | adapter | Removes indirect transitive effects from an observed dependency matrix: G_dir = G_obs (I + G_obs)^-1. |
| [DMD-GRN (Koopman / dynamic mode decomposition)](dmdgrn.md) | extended | native | Fits a linear operator from perturbation to response with a truncated SVD (controlled DMD). |
| [AtlasShift (empirical cross-atlas transfer)](atlasshift.md) | extended | native | Predicts a perturbation's response from the same perturbation in other cell contexts (source atlases) by weighted averaging. |

## Co-expression and statistical baselines

| algorithm | status | partial | in one line |
|---|---|---|---|
| [Pearson |r|](pearson.md) | baseline / control | native | Absolute correlation of two genes across the conditions. |
| [GENIE3 (random forest)](genie3.md) | extended | native | Predicts every gene from all others with tree ensembles and ranks regulators by feature importance. |
| [GRNBoost-lite (boosted trees)](grnboost.md) | baseline / control | native | Gradient-boosted trees per target gene, regulators ranked by importance (the GRNBoost2 recipe, with scikit-learn). |
| [ARACNE (mutual information + DPI)](arachne.md) | extended | native | Keeps edges with high mutual information and removes the weakest edge of every triangle (data processing inequality). |
| [Gene-level features (hubness only)](gene_features.md) | baseline / control | native | Scores a -> j as (how much a moves anything) x (how much j moves under anything), ignoring the specific pair. |

## Causal and DAG methods

| algorithm | status | partial | in one line |
|---|---|---|---|
| [CausalGRN](causalgrn.md) | extended | native | Builds an undirected skeleton by partial correlation and directs edges with the interventions (do-calculus style). |
| [NOTEARS (continuous DAG)](notears.md) | extended | native | Learns a directed acyclic graph by continuous optimisation with the constraint tr(exp(W o W)) - d = 0. |

## Neural and latent-space methods

| algorithm | status | partial | in one line |
|---|---|---|---|
| [CellBox (neural ODE)](cellbox.md) | extended | native | Fits x_i = phi(sum_j W_ij x_j + u_i) / alpha_i to the steady states with an L1 penalty on W. |
| [DeepRIG (graph autoencoder)](deeprig.md) | extended | native | Scores gene pairs with a graph autoencoder (GCN encoder, inner-product decoder) on a co-expression prior graph. |
| [SFINN-style scoring](sfinn.md) | extended | native | Perturbation sensitivity weighted by a shared-factor association between genes. |
| [CPA-Lite (latent perturbation autoencoder)](cpa_lite.md) | extended | native | A small autoencoder whose latent space adds perturbation vectors; gene influence read from its Jacobian. |

## Experimental (cell-level, not in the registry)

| algorithm | status | partial | in one line |
|---|---|---|---|
| [bicycle-lite (steady-state OU likelihood)](bicycle_lite.md) | experimental | experimental | Maximum-likelihood direct effects from per-node mean shifts and one pooled covariance of an Ornstein-Uhlenbeck steady state. |
| [DOTEARS-lite (interventional DAG)](dotears_lite.md) | experimental | experimental | NOTEARS with the intervened gene's equation dropped in its own cells and per-gene noise variances. |
| [Rank consensus](interventional_consensus.md) | experimental | experimental | Average of the rank-normalised scores of several interventional-aware methods (inspre, mean difference, bicycle-lite, DOTEARS-lite). |
