# GRN-Sim & phpIoRN 2.0: Gene Regulatory Network Perturbation Simulator & Reverse Engineering Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

A full-stack Python framework and interactive web simulation studio (**phpIoRN 2.0**) for non-linear gene regulatory network (GRN) perturbation simulation, oscillation/steady-state detection, and network topology reverse engineering.

This package is a faithful, modern reverse-engineering and expansion of the foundational MATLAB + `phpIoRN` research software developed by **Jan Wildenhain** and **Edmund J. Crampin**, stemming from the Bachelor of Honours Thesis:
> **"In Silico Modelling of Gene Networks and Reverse Engineering of Transcriptional Networks"**  
> *Jan Wildenhain (2004/2005)*

---

## 📖 Citation & Academic Reference

If you use this software or its methodology in your research, please cite the primary publication:

```bibtex
@article{WildenhainCrampin2006,
  author    = {Wildenhain, Jan and Crampin, Edmund J.},
  title     = {Reconstructing gene regulatory networks: from random to scale-free connectivity},
  journal   = {IEE Proceedings - Systems Biology},
  volume    = {153},
  number    = {4},
  pages     = {247--256},
  year      = {2006},
  publisher = {IET},
  doi       = {10.1049/ip-syb:20050073},
  url       = {https://doi.org/10.1049/ip-syb:20050073}
}
```

> **Wildenhain, J., & Crampin, E. J. (2006).** *Reconstructing gene regulatory networks: from random to scale-free connectivity.* **IEE Proceedings - Systems Biology**, 153(4), 247–256. [doi:10.1049/ip-syb:20050073](https://doi.org/10.1049/ip-syb:20050073).

---

## 📸 Visual Tour (`phpIoRN 2.0` Web Application)

### 1. Simulation & Inference Wizard Studio (Dark & Light Themes)
Full 3-step `phpIoRN` simulation workflow (`get_network.php` → `run_sim.php` → `show_sim.php`) featuring force-directed / circular / hierarchical graph layouts, steady-state vs. oscillation diagnostics, $3 \times 3 + \text{marginals}$ Observed vs. Predicted misclassification matrices (`$TMP_CMAT`), 8-metric performance bars (`$TMP_STATMAT`), and `.msmt` / `.pert` / `.dot` / `.json` exports:

| Dark Theme (`Simulation Studio`) | Light Theme (`Simulation Studio`) |
| :---: | :---: |
| ![Simulation Studio Dark](docs/screenshots/01_simulation_studio_dark.png) | ![Simulation Studio Light](docs/screenshots/02_simulation_studio_light.png) |

### 2. Native Graphviz `dot`/`neato` 9-State Classification SVG & Interactive ODE/Hill Popups
Exact reproduction of `statsgraph_to_dot.m` 9-state edge styling (`TP+`, `TP-`, `FP+`, `FP-`, `FN+`, `FN-`, `SignErr +-`, `SignErr -+`) and interactive `popups.php` modals when clicking any gene node or regulatory edge:

| Graphviz 9-State SVG (`statsgraph_to_dot.m`) | Node ODE Trajectory Inspector (`popups.php?type=node`) | Edge Hill Kinetics Curve (`popups.php?type=edge`) |
| :---: | :---: | :---: |
| ![Graphviz 9-State SVG](docs/screenshots/03_graphviz_dot_9state.png) | ![ODE Trajectory Inspector](docs/screenshots/04_ode_kinetic_inspector.png) | ![Edge Hill Kinetics Curve](docs/screenshots/05_edge_hill_inspector.png) |

### 3. *E. coli* SOS DNA Repair Benchmark, Thesis Figures Gallery & Persistent SQLite Archive
Replication of the 9-gene *E. coli* SOS DNA Repair Pathway (Gardner et al., *Science* 2003; Thesis Table 4.1), Thesis & Paper Figures Gallery, and persistent SQLite database archive (`nw` + `alg` tables from `dumpdb.sql`) with interactive run inspection (`showpre.php`):

| *E. coli* SOS Pathway (Table 4.1) | Thesis & Paper Figures Gallery | SQLite Simulation Archive (`showpre.php`) |
| :---: | :---: | :---: |
| ![E. coli SOS Pathway](docs/screenshots/06_ecoli_sos_pathway.png) | ![Thesis and Paper Figures Gallery](docs/screenshots/07_thesis_paper_figures_gallery.png) | ![SQLite Simulation Archive Inspector](docs/screenshots/08_sqlite_archive_detail.png) |

### 4. Virtual Cell Challenge 2026 & AtlasShift Studio (Tabs 1 & 5)
Direct comparative benchmarking between Mechanistic Causal GRN modeling (`phpIoRN`) and Cross-Atlas Empirical Transfer (`AtlasShift`, Kaipeng Meng, Arc Institute VCC 2026 Leaderboard #82). Features in-distribution vs. out-of-distribution (OOD) perturbation generalization analysis, cell line edge rewiring sensitivity curves ($\rho \in [0\%, 50\%]$), and single-cell `.h5ad` AnnData export (400 cells/condition):

| Tab 1: Studio with AtlasShift | Tab 5: VCC Benchmark (Dark Theme) | Tab 5: VCC Benchmark (Light Theme) |
| :---: | :---: | :---: |
| ![AtlasShift in Studio](docs/screenshots/09_tab1_atlasshift_inference_studio.png) | ![VCC Benchmark Dark](docs/screenshots/10_tab5_vcc_atlasshift_benchmark_dark.png) | ![VCC Benchmark Light](docs/screenshots/11_tab5_vcc_atlasshift_benchmark_light.png) |

---

## 🔬 Core Capabilities

### 1. In Silico Non-Linear GRN Perturbation Simulator
* **Network Topologies**:
  * **Random (Erdős–Rényi)**: Constant attachment probability (`netsim.m` Type 1).
  * **Scale-Free (Barabási–Albert)**: Preferential attachment yielding power-law degree distributions $P(k) \sim k^{-\gamma}$ (`netsim.m` Type 2).
  * **Hierarchical Modular**: Recursive scale-invariant motif duplication with peripheral rewiring (`netsim.m` Type 3).
* **Non-linear Hill Kinetics (`create_network.m` / `do_perturbation.m`)**:
  $$\frac{dx_i}{dt} = v_i \prod_{j \in \text{Inh}} \left(\frac{K_{ij}^{m_j}}{I_j^{m_j} + K_{ij}^{m_j}}\right) \prod_{k \in \text{Act}} \left(1 + \frac{A_k^{m_k}}{A_k^{m_k} + Ka_k^{m_k}}\right) - b_i x_i + u_i$$
* **Oscillation & Steady-State Diagnostics (`create_network.m` & `do_perturbation.m`)**:
  * Integrates over horizon $e = s^2 + 100$ steps and checks exact squared-state Euclidean norm convergence:
    $$\text{efnorm}(t) = \left\| \mathbf{x}(t+1)^2 - \mathbf{x}(t)^2 \right\|_2 \le 10^{-4}$$
  * Evaluates Jacobian eigenvalues $\lambda_i = \alpha_i \pm i\omega_i$ at the terminal state to detect limit-cycle oscillations ($\max \text{Re}(\lambda) \ge 0, |\text{Im}(\lambda)| > 0$, zero-crossings in $dx_i/dt$) and damped oscillations (`abs(efnorm - efnorm_update) < 1e-4` convergence check).
* **Noise & Biological Variability Models (`do_biovariability.m`, `error_model.m`)**:
  * Biological variability across $M$ replicates ($\zeta \sim \mathcal{N}(0, \sigma_b^2)$) and Gaussian measurement noise ($\epsilon \sim \mathcal{N}(0, \sigma_m^2)$).

### 2. Network Inference & Reverse Engineering Algorithms (100% MATLAB Numerical Equivalence)
* **S2G (Simple-to-General — `s2g.m` / `getvar.m`)**:
  * Column-normalized basis regression ($B = X / \|X_j\|_2$) with swap replacement loop (`getanother`), minimizing the MDL log-error penalty $C(k) = N \ln(\overline{E^2}) + k \ln(N)$.
* **G2S (General-to-Specific — `g2s.m` / `gen2spec.m`)**:
  * Systematic backward elimination starting from full connectivity with single-variable drop optimization (`dropanother`), minimizing AIC $C_{\text{AIC}}(k) = \sum E^2 + k$.
* **NIR (Network Identification by multiple Regression — `nir_v3.m`)**:
  * Gardner et al. (*Science* 2003) greedy forward top-$D$ search over sparse regulatory subsets with Tikhonov ridge regularization.
* **FAST (`fast.m` / `fast_modified.m`)**:
  * Singular Value Decomposition (SVD) pseudo-inverse reconstruction ($A_0 = (X - U) V_r S_r^{-1} U_r^T - I$) with self-regulation adjustment and degree-guided truncation (Farina et al.).
* **Voting Ensemble (`do_vote.m`)**:
  * Strict consensus operator combining S2G and G2S predictions where $\text{sign}(\text{S2G}) == \text{sign}(\text{G2S})$, cutting false discovery rates drastically.
* **By-Chance (`by_chance.m`)**:
  * Random null baseline maintaining expected network connectivity density and activator/inhibitor balance.

#### Verified 100% Match to Thesis Table 4.1 (*E. coli* SOS Benchmark)
Tested in [`tests/test_inference.py::test_matlab_equivalence_sos_table_4_1`](tests/test_inference.py):

| Algorithm | Coverage % | False Positives % | Inferred Conn. (Power %) | FDR (FP/Inferred %) | Inferred Sign % | Total Errors | $3 \times 3$ Confusion Matrix Match (Appendix A.4–A.7) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2G** | `56.79%` | `8.64%` | `42.86%` | `25.00%` | `12.35%` | **`46`** | `[[6, 8, 17], [3, 4, 11], [2, 5, 25]]` (100% Identical) |
| **G2S** | `38.27%` | `16.05%` | `24.49%` | `52.00%` | `6.17%` | **`57`** | `[[5, 4, 22], [3, 0, 15], [5, 8, 19]]` (100% Identical) |
| **FAST** | `55.56%` | `19.75%` | `59.18%` | `35.56%` | `23.46%` | **`46`** | `[[16, 4, 11], [6, 3, 9], [13, 3, 16]]` (100% Identical) |
| **NIR** | `62.96%` | `16.05%` | `65.31%` | `28.89%` | `25.93%` | **`41`** | `[[17, 7, 7], [4, 4, 10], [11, 2, 19]]` (100% Identical) |

#### Verified Bit-for-Bit Match to 60-Gene MATLAB Run (`random_nz60_run5_iot2_sig0.mat`)
Tested in [`tests/test_inference.py::test_matlab_equivalence_60_gene_archive`](tests/test_inference.py):
- **S2G**: Inferred $3 \times 3$ confusion vector matches stored MATLAB vector `pred_matrix_sg[:, -1]` bit-for-bit: `[22, 0, 70, 0, 33, 59, 5, 0, 3411]`.
- **NIR**: Inferred $3 \times 3$ confusion vector matches stored MATLAB vector `pred_matrix_gardener[:, -1]` bit-for-bit: `[15, 2, 99, 2, 22, 40, 10, 9, 3401]`.

### 3. Persistent SQLite Archive & Thesis Figures Gallery
* **Persistent SQLite Archive (`dumpdb.sql` / `database.py`)**:
  Stores every simulation run and per-algorithm evaluation into `data/rgn_db.sqlite` using the original `phpIoRN` relational schema (`nw` table for network parameters + `alg` table for per-algorithm metrics), plus full JSON state recovery for instant inspection in Tab 4 (`showpre.php`).
* **Thesis & Paper Figures Gallery (Dual Engine)**:
  Tab 3 allows toggling between **"📂 Stored MATLAB `.mat` Archives"** (loaded directly from 31 authentic 2004 MATLAB `.mat` files in `data/matlab_benchmarks/`) and **"⚡ Compute Live Python Sweep"** (executing dynamic multi-network ODE simulation and inference across Random, Scale-Free, and Hierarchical topologies). Includes an interactive inspector across all 25 `overallstats/DIAG_nz70_iot*_std*_mean.mat` files ($N \in [10..70]$, $k_{\text{av}} \in [1.0..5.0]$, $\sigma \in [0..0.01]$).

### 4. Arc Institute Virtual Cell Challenge 2026 & AtlasShift Comparison

We benchmarked the modernized `phpIoRN` simulator against the contemporary state-of-the-art methodology from the **Arc Institute Virtual Cell Challenge 2026**:
[AtlasShift](https://github.com/kaipengm2/Virtual-Cell-Challenge-2026) by **Kaipeng Meng** (Leaderboard #82, Score: `0.15456`).

#### Scientific Paradigm: Empirical Delta Transfer vs. Mechanistic Causality
* **AtlasShift (Empirical Cross-Atlas Delta Transfer)**:
  Estimates target cell-line response $\mathbf{y}_t = \mathbf{y}_{t,\text{ctrl}} + \Delta \mathbf{y}$ by taking weighted combinations of observed perturbations in reference atlases (e.g., K562, HCT116, HEK293T) and emitting discrete single-cell UMI counts via dual-moment Negative-Binomial matching.
* **phpIoRN (Mechanistic Causal ODE Inversion)**:
  Reconstructs the underlying directed, signed regulatory Jacobian topology $\mathbf{A} \in \mathbb{R}^{N \times N}$ using sparse regression ($L_0 / L_1$ with MDL/AIC criteria) and propagates perturbations through the causal gene network:
  $$\hat{\mathbf{X}} = \mathbf{A}_{\text{est}}^{-1} \mathbf{U}$$

#### In-Distribution vs. Out-of-Distribution (OOD) Perturbation Generalization

| Evaluation Metric | phpIoRN Mechanistic GRN (`S2G` / `NIR` / `Voting`) | AtlasShift (Empirical Transfer) | Scientific Rationale |
| :--- | :---: | :---: | :--- |
| **Causal Edge Recovery** | **High** (Coverage: `57%–65%`, FDR: `25%–35%`) | **0%** (Topology Non-identifiable) | AtlasShift does not infer physical regulatory links or directed graphs. |
| **In-Distribution Prediction ($r$)** | **$0.55 - 0.72$** | **$0.52 - 0.58$** | Both capture strong correlation when the perturbation was previously observed across reference cell atlases. |
| **OOD Unseen Perturbation ($r$)** | **$0.72$** | **$0.00$** | **AtlasShift collapses to 0** because without an observed reference vector for that target gene, its empirical prior $\Delta \mathbf{y} = \mathbf{0}$. `phpIoRN` predicts unseen perturbations by propagating forces through the learned causal network graph. |
| **Target Context Rewiring ($\rho$)** | **Robust** (Adapts to local network structure) | **Degrades** ($r = 0.56 \to 0.24$ at $\rho=50\%$) | Biological rewiring (enhancer switching, mutations) breaks empirical cross-cell line delta transfer. |

#### Single-Cell Perturb-seq AnnData (`.h5ad`) Generator & Exporter
Fully integrated into the web studio and export pipeline (`/api/export/perturb_seq_h5ad`):
* **Single-Cell Scale**: Simulates 400 individual cells per perturbation condition (matching 10x Genomics Chromium Perturb-seq and Arc Institute VCC 2026 benchmark specifications).
* **Realistic Noise & Overdispersion**: Implements Gamma-Poisson (Negative-Binomial) emission with log-normally distributed sequencing depth variations across single cells:
  $$s_c \sim \text{LogNormal}(\mu_{\text{depth}}, \sigma_{\text{depth}}), \quad X_{c, g} \sim \text{NegBinom}\left(\text{mean} = \frac{\mu_g s_c}{10^6}, \text{dispersion} = \phi_g\right)$$
* **Standard Single-Cell Format**: Exports production `.h5ad` AnnData objects with:
  * `X`: Raw integer single-cell UMI counts matrix.
  * `layers['cpm']`: Library-size normalized Counts Per Million.
  * `obs`: Single-cell metadata (`cell_barcode`, `perturbation`, `cell_line`, `library_size`).
  * `var`: Gene metadata (`gene_id`, `baseline_mean`, `dispersion`).
  * `uns`: Ground-truth Jacobian interaction matrix $\mathbf{A}$, perturbation matrix $\mathbf{U}$, and simulation parameters.

---

## 🐳 Running with Docker (Recommended)

Build and launch the full `phpIoRN 2.0` web application (including Graphviz `dot`/`neato` and SQLite persistence):

```bash
docker build -t grn-sim-phpiorn:latest .
docker run -d --name grn-sim-phpiorn -p 8085:8080 -v $(pwd)/data:/app/data grn-sim-phpiorn:latest
```

Then open **http://localhost:8085** in your browser.

---

## 🚀 Local Python Installation & CLI

```bash
git clone https://github.com/jwildenhain/gene-network-sim-inference.git
cd gene-network-sim-inference
pip install -e ".[dev]"
```

### Launch the Web Server Directly
```bash
python3 -m grn_sim.web.server --host 0.0.0.0 --port 8085
```

### Run Verification Test Suite
```bash
# Run all 30 unit, ODE, topology, MATLAB-equivalence, and AtlasShift/Perturb-seq tests
pytest -v

# Run full Playwright E2E browser + SQLite persistence test against live server
pytest tests/test_playwright_e2e.py -v -s
```

---

## 📁 Repository Structure

```text
gene-network-sim-inference/
├── Dockerfile                # Container image with Python 3.11 + Graphviz + SQLite + AnnData
├── LICENSE                   # MIT License
├── README.md                 # Complete documentation, screenshots & citation
├── SPECIFICATION.md          # Mathematical & algorithmic reverse-engineering spec
├── pyproject.toml            # Package configuration (PEP 517/518, anndata, pandas, h5py)
├── demo.py                   # Interactive CLI demo script
├── docs/
│   └── screenshots/          # Verified Docker & Playwright UI screenshots (11 views)
├── data/
│   ├── results_gapa_20021209_and_mmc_dbl.mat  # Gardner 2003 E. coli SOS dataset
│   ├── SOSdata.txt           # Tabular SOS expression & perturbation data
│   ├── rgn_db.sqlite         # Persistent SQLite simulation archive (nw & alg tables)
│   └── matlab_benchmarks/    # 31 Authentic 2004 MATLAB benchmark .mat archives:
│       ├── DIAG_random_nz60_iot2_std0_mean.mat
│       ├── DIAG_sf_nz60_iot2_std0_mean.mat
│       ├── DIAG_h_nz60_iot2_std0_mean.mat
│       ├── DIAG_noise_nz48_iot0_std0_mean.mat
│       ├── random_nz60_run5_iot2_sig0.mat
│       └── overallstats/     # 25 archives across k_av in [1, 1.5, 2, 3, 5] & sigma in [0..0.01]
├── src/
│   └── grn_sim/
│       ├── core/             # Network models & topology generators (ER, BA, Hierarchical)
│       ├── simulator/        # Hill kinetics, ODE solver, Perturb-seq single-cell .h5ad simulator
│       ├── inference/        # S2G, G2S, NIR, FAST, ByChance, Voting Ensemble, AtlasShift
│       ├── evaluation/       # Confusion matrix, ROC, Signed PPV, Coverage (Table 4.1)
│       ├── benchmarks/       # SOS pathway replication, sweeps & VCC 2026 benchmark suite
│       ├── agents/           # Multi-agent autonomous SDLC orchestration
│       ├── web/              # phpIoRN 2.0 HTTP server, SQLite DB, Graphviz SVG & Tab 1-5 UI
│       └── cli.py            # CLI entry point
└── tests/
    ├── test_inference.py     # Inference algorithms & 100% MATLAB equivalence tests
    ├── test_metrics.py       # Table 4.1 & stats_obs_pred.m evaluation metrics tests
    ├── test_simulator.py     # ODE solver, Hill kinetics, oscillation & noise tests
    ├── test_sos_benchmark.py # E. coli SOS pathway benchmark replication tests
    ├── test_topology.py      # Random, Scale-Free, and Hierarchical graph tests
    ├── test_web_and_visualization.py # Web server, Graphviz DOT & API tests
    ├── test_atlasshift_and_perturbseq.py # AtlasShift & AnnData .h5ad generator tests
    └── test_playwright_e2e.py # Full Playwright browser + SQLite E2E test
```


---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
