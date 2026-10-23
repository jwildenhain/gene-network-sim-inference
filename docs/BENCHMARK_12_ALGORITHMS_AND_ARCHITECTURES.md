# Systematic Benchmark: 12 Reverse-Engineering Algorithms across 3 Network Graph Architectures

This document provides the complete technical specification, mathematical methodology, cross-architecture empirical results, and user-interface integrations for the comprehensive benchmark evaluated across **all 12 algorithms** and **3 network topologies** in GRN-Sim / phpIoRN 2.0.

---

## 1. Executive Summary & Benchmark Design

The GRN-Sim systematic benchmark framework evaluates reverse-engineering inference algorithms across 4 continuous and topological parameter dimensions, using non-linear Hill differential equation dynamics:

$$\frac{d x_i}{dt} = \frac{v_i \left(1 + \sum_{j \in \text{Act}(i)} \left(\frac{x_j}{K_{ij}}\right)^{n_{ij}}\right)}{1 + \sum_{j \in \text{Act}(i)} \left(\frac{x_j}{K_{ij}}\right)^{n_{ij}} + \sum_{k \in \text{Inh}(i)} \left(\frac{x_k}{K_{ik}}\right)^{n_{ik}}} - \gamma_i x_i$$

### Parameter Sweep Dimensionality:
- **Graph Topologies (3)**:
  1. **Random Erdős-Rényi ($G(N, p)$)**: Homogeneous Poisson degree distribution, uncorrelated edge wiring.
  2. **Scale-Free / Power-Law (Barabási-Albert)**: Preferential attachment degree distribution $P(k) \sim k^{-3}$, master regulator hubs.
  3. **Hierarchical Modular (Ravasz-Barabási)**: Iterative nested module replication, high clustering coefficient $C(k) \sim 1/k$.
- **Network Dimensions ($N$)**: $N \in [6, 8, 10, 12, 16]$ genes.
- **Average Connectivity ($k_{\text{av}}$)**: $k_{\text{av}} \in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]$ edges/gene.
- **Experimental Measurement Noise ($\sigma_m$)**: Gaussian noise $\sigma_m \in [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]$ added to steady-state expression ratios.

### Evaluation Metrics:
Across every run and algorithm, inferred directed signed networks are compared against the ground-truth adjacency matrix $A^* \in \{-1, 0, +1\}^{N \times N}$ using the 9-state misclassification matrix:
- **$F_1$ Score**: Harmonic mean of Precision (PPV) and Recall (Power):
  $$F_1 = 2 \cdot \frac{\text{PPV} \cdot \text{Power}}{\text{PPV} + \text{Power}}$$
- **Power / Sensitivity / Recall (%)**: $\frac{TP}{TP + FN} \times 100$
- **Specificity (%)**: $\frac{TN}{TN + FP} \times 100$
- **Precision / PPV (%)**: $\frac{TP}{TP + FP} \times 100$
- **Signed PPV (%)**: Percentage of true-positive predictions where the regulatory sign ($+$ activation vs $-$ inhibition) correctly matches ground truth.
- **False Discovery Rate (FDR %)**: $\frac{FP}{TP + FP} \times 100 = 100 - \text{PPV}$
- **Total Errors**: $FP + FN + \text{Sign Errors}$

---

## 2. The 12 Evaluated Algorithms

| # | Algorithm Key | Official Name | Methodological Family | Algorithmic Mechanism |
|:---:|:---|:---|:---|:---|
| 1 | `atlasshift` | **AtlasShift (Empirical)** | Cross-Atlas Latent Transfer | Zero-shot empirical latent subspace projection & cosine similarity |
| 2 | `s2g` | **Simple-to-General (S2G)** | Forward Stepwise Model Selection | Greedy forward edge inclusion penalized by Akaike Information Criterion (AIC) |
| 3 | `vote` | **Voting Ensemble** | Consensus Ensemble | Strict consensus intersection ($S2G \cap G2S$) requiring bidirectional agreement |
| 4 | `g2s` | **General-to-Specific (G2S)** | Backward Stepwise Elimination | Full-model initialization with backward edge pruning via AIC stopping criterion |
| 5 | `nir` | **NIR (Multiple Regression)** | Linear Matrix Inversion | Multiple linear regression assuming near-equilibrium Jacobian linearization |
| 6 | `ridge` | **Multi-Output RidgeCV** | L2 Regularized Inverse | Ridge regression with internal cross-validation for L2 penalty $\alpha$ |
| 7 | `cellbox` | **CellBox Neural ODE** | Differentiable Neural ODE | Adjoint gradient sensitivity optimization of non-linear interaction matrix $W$ |
| 8 | `netdeconv` | **Network Deconvolution** | Direct-Indirect Eigen Filtering | Eigenvalue scaling to remove transitive indirect cascade effects from correlation |
| 9 | `fast` | **FAST (SVD Inversion)** | SVD Truncation | Truncated singular value decomposition inverse filter on perturbation matrix |
| 10 | `genie3` | **GENIE3 (Random Forest)** | Tree Ensemble Feature Importance | Random Forest regressor per target gene, ranking regulatory importance scores |
| 11 | `cpa_lite` | **CPA-Lite Autoencoder** | Deep Latent Autoencoder | Deep autoencoder disentangling basal cellular state from perturbation vectors |
| 12 | `by_chance` | **By-Chance (Null Model)** | Random Baseline | Bernoulli independent edge sampling calibrated to average network density |

---

## 3. Comprehensive Leaderboard (720 Evaluations across 60 Networks)

Results aggregated across all 60 synthetic sweep networks in SQLite database `rgn_db.sqlite`:

| Rank | Algorithm | Runs | Mean $F_1$ | Power (%) | Specificity (%) | PPV (%) | Signed PPV (%) | FDR (%) | Total Errors | Standing Badge |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **#1** | **AtlasShift (Empirical)** | 60 | **0.594** | **67.1%** | 85.7% | **58.6%** | **57.5%** | **41.4%** | 19.1 | `#1 Overall Winner` |
| **#2** | **Simple-to-General (S2G)** | 60 | **0.446** | 58.6% | 76.9% | 38.5% | 34.7% | 61.5% | 29.8 | `#2 Top Mechanistic` |
| **#3** | **Voting Ensemble** | 60 | **0.441** | 37.0% | **94.1%** | 58.5% | 57.3% | 41.5% | **16.9** | `#3 Consensus Ensemble` |
| **#4** | **General-to-Specific (G2S)** | 60 | 0.418 | 46.1% | 85.4% | 39.0% | 36.9% | 61.0% | 23.4 | `#4 Top Tier` |
| **#5** | **NIR (Multiple Regression)** | 60 | 0.394 | 43.2% | 84.0% | 36.4% | 33.8% | 63.6% | 25.0 | `#5 Top Tier` |
| **#6** | **Multi-Output RidgeCV** | 60 | 0.315 | 64.4% | 48.6% | 21.9% | 18.3% | 78.1% | 57.1 | `#6 Contender` |
| **#7** | **CellBox Neural ODE** | 60 | 0.301 | 47.2% | 65.9% | 22.9% | 8.6% | 77.1% | 45.7 | `#7 Contender` |
| **#8** | **Network Deconvolution** | 60 | 0.297 | 73.5% | 35.3% | 18.9% | 13.1% | 81.1% | 70.9 | `#8 Contender` |
| **#9** | **FAST (SVD Inversion)** | 60 | 0.294 | 32.4% | 82.1% | 27.1% | 24.1% | 72.9% | 29.1 | `#9 Contender` |
| **#10** | **GENIE3 (Tree Forest)** | 60 | 0.294 | 81.2% | 25.3% | 18.2% | 12.3% | 81.8% | 78.8 | `#10 Contender` |
| **#11** | **CPA-Lite Autoencoder** | 60 | 0.278 | 78.8% | 21.5% | 17.1% | 9.2% | 82.9% | 88.2 | `#11 Contender` |
| **#12** | **By-Chance (Null Model)** | 60 | 0.205 | 22.7% | 80.0% | 18.8% | 9.2% | 81.2% | 34.4 | `#12 Null Baseline` |

---

## 4. Graph Architecture Cross-Comparison Analysis

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Graph Architecture Sweeps                │
                    └────────────────────────────────────────────────────────┘
                               /                 |                 \
                              /                  |                  \
                             v                   v                   v
            ┌───────────────────┐ ┌────────────────────┐ ┌──────────────────────┐
            │   Random (ER)     │ │  Scale-Free (BA)   │ │ Hierarchical Modular │
            │   Mean F1: 0.363  │ │  Mean F1: 0.360    │ │    Mean F1: 0.346    │
            │   Power: 55.2%    │ │  Power: 54.3%      │ │    Power: 53.7%      │
            │   Spec:  64.2%    │ │  Spec:  66.1%      │ │    Spec:  65.8%      │
            │   FDR:   68.4%    │ │  FDR:   68.2%      │ │    FDR:   69.4%      │
            └───────────────────┘ └────────────────────┘ └──────────────────────┘
                     |                     |                        |
            Top: AtlasShift (0.563) Top: AtlasShift (0.621)  Top: AtlasShift (0.599)
            #2: NIR (0.457)         #2: S2G (0.496)          #2: S2G (0.446)
            #3: Voting (0.445)      #3: Voting (0.483)       #3: Voting (0.397)
            #4: G2S (0.438)         #4: G2S (0.425)          #4: G2S (0.391)
            #5: S2G (0.395)         #5: NIR (0.381)          #5: NIR (0.344)
```

### 4.1 Topology-Specific Observations

1. **Scale-Free Advantage for Stepwise AIC (S2G)**:
   - S2G attains its highest performance on Scale-Free graphs ($F_1 = 0.496$, Power = $62.1\%$), outperforming its Random graph performance ($F_1 = 0.395$) by $+0.101$.
   - **Mechanism**: The presence of heavily connected hub genes creates large perturbation shifts that stand out cleanly in regression residuals, allowing forward AIC selection to capture master regulators before the penalty halts expansion.

2. **Random Graph Advantage for Linear Inversion (NIR)**:
   - NIR achieves its strongest ranking on Random graphs ($F_1 = 0.457$, rank #2), outperforming S2G on that topology.
   - **Mechanism**: Random Erdős-Rényi wiring exhibits minimal collinearity and uniform eigenvalue spectra, maximizing stability during pseudo-inverse matrix computation.

3. **Hierarchical Modular Graph Resistance**:
   - Hierarchical networks prove the most challenging topology across all algorithms (mean $F_1 = 0.346$, vs $0.363$ Random and $0.360$ Scale-Free).
   - **Mechanism**: Dense intra-module clusters create high secondary correlation cascades that mimic direct causal edges, elevating false discovery rates across non-ensemble methods.

4. **Voting Ensemble Consensus Robustness**:
   - Across all three graph topologies, Voting Ensemble ($S2G \cap G2S$) consistently delivers the lowest total error count ($18.6$ Random, $15.1$ Scale-Free, $16.9$ Hierarchical) and specificity exceeding $92.9\%$.

---

## 5. Web Interface & Interactive Integration

### 5.1 Tab 5 ("5. Overall Statistics")
- **KPI Ribbon**: Total Evaluated Networks ($60$), Algorithmic Evaluations ($720$), Top Overall Performer (`AtlasShift`), Top Mechanistic Method (`S2G`), and Consensus Ensemble (`Voting Ensemble`).
- **Leaderboard Table (`#tbl-overall-leaderboard`)**: Sortable columns, live $F_1$ highlighting, and standing badges across all 12 algorithms.
- **Architecture Cross-Comparison Cards (`#overall-arch-cards-grid`)**: 3 cards displaying topology properties, aggregate metrics, top performer highlight, and top-5 ranked mini-tables.
- **Canvases**:
  - Figure 1A: Network Size vs Power (%)
  - Figure 1B: Network Size vs False Discovery Rate (%)
  - Figure 2: Grouped bar chart comparing Random, Scale-Free, and Hierarchical Modular across Power, Specificity, $F_1 \times 100$, and FDR.
  - Figure 3: Average Connectivity Sweep ($k_{\text{av}} \in [1.0, 3.5]$)
  - Figure 4: Measurement Noise Sensitivity ($\sigma_m \in [0.0, 0.20]$)
- **Historical Gallery Switcher**: One-click toggle preserving authentic 2004 MATLAB `.mat` figures gallery.

### 5.2 Tab 3 ("3. Previous Simulations Archive")
- **Restricted View**: Restricts display strictly to the best 15 simulations per set (Responsive, Flat Levels, Oscillating, Topologies).
- **12-Algorithm Selector (`#hist-detail-alg`)**: Allows inspecting any of the 12 algorithms for any archived run, dynamically updating the 9-state Graphviz reconstruction and misclassification matrix.

---

## 6. Verification & Automated Test Suite

The Playwright end-to-end verification suite in `tests/test_playwright_overall_stats_and_best15.py` validates:
- [x] Top navigation tab title: `"5. Overall Statistics"`
- [x] KPI metrics: Total Runs $= 60$, Algorithmic Evaluations $= 720$
- [x] Architecture cards: Exactly 3 cards (Random, Scale-Free, Hierarchical Modular)
- [x] Leaderboard rows: Exactly 12 rows, Rank #1 is AtlasShift
- [x] Tab 3 Inspector dropdown: Exactly 12 algorithm options
- [x] Full visual screenshots captured:
  - `docs/screenshots/47_tab5_overall_statistics_sweep.png`
  - `docs/screenshots/51_tab5_architecture_cards_and_figures.png`
  - `docs/screenshots/52_tab5_figures_2_3_4.png`
  - `docs/screenshots/53_tab3_detail_inspector_algorithm_view.png`
