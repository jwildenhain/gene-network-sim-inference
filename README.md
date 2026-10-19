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

### 2. Network Inference & Reverse Engineering Algorithms
* **S2G (Simple-to-General — `s2g.m` / `getvar.m`)**:
  * Greedy forward basis selection with swap replacement (`getanother`), minimizing the MDL/AIC log penalty $C(k) = N \ln(\text{MSE}) + k \ln(N)$.
* **G2S (General-to-Specific — `g2s.m` / `gen2spec.m`)**:
  * Systematic backward elimination starting from full connectivity, minimizing AIC $C_{\text{AIC}}(k) = \text{SSE} + k$.
* **NIR (Network Identification by multiple Regression — `nir_v3.m`)**:
  * Top-$D$ beam search over sparse regulatory subsets with errors-in-variables Tikhonov regularization (Gardner et al., *Science* 2003).
* **FAST (`fast.m` / `fast_modified.m`)**:
  * Singular Value Decomposition (SVD) pseudo-inverse truncation with self-regulation correction (`+ I_N`) and degree-guided sparsity thresholding (Farina et al.).
* **Voting Ensemble (`do_vote.m`)**:
  * Strict consensus operator combining S2G and G2S predictions where `sign(S2G) == sign(G2S)`.
* **By-Chance (`by_chance.m`)**:
  * Random null baseline matching expected connectivity density and activator/inhibitor ratio.

### 3. Persistent SQLite Archive (`dumpdb.sql` / `database.py`)
* Stores every simulation run and per-algorithm evaluation into `data/rgn_db.sqlite` using the original `phpIoRN` relational schema (`nw` table for network parameters + `alg` table for per-algorithm metrics), plus full JSON state recovery for instant inspection in Tab 4 (`showpre.php`).

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

### Run Unit & Playwright E2E Tests
```bash
# Run core unit & MATLAB equivalence tests
pytest tests/test_grn.py -v

# Run full Playwright E2E browser + SQLite persistence test against live server
pytest tests/test_playwright_e2e.py -v -s
```

---

## 📁 Repository Structure

```text
gene-network-sim-inference/
├── Dockerfile                # Container image with Python + Graphviz + SQLite
├── LICENSE                   # MIT License
├── README.md                 # Complete documentation, screenshots & citation
├── SPECIFICATION.md          # Mathematical & algorithmic reverse-engineering spec
├── pyproject.toml            # Package configuration (PEP 517/518)
├── demo.py                   # Interactive CLI demo script
├── docs/
│   └── screenshots/          # Verified Docker & Playwright UI screenshots
├── data/
│   ├── results_gapa_20021209_and_mmc_dbl.mat  # Gardner 2003 E. coli SOS dataset
│   ├── SOSdata.txt           # Tabular SOS expression & perturbation data
│   └── rgn_db.sqlite         # Persistent SQLite simulation archive (nw & alg tables)
├── src/
│   └── grn_sim/
│       ├── core/             # Network models & topology generators (ER, BA, Hierarchical)
│       ├── simulator/        # Hill kinetics, ODE solver, oscillation diagnostics, noise
│       ├── inference/        # S2G, G2S, NIR, FAST, ByChance, Voting Ensemble
│       ├── evaluation/       # Confusion matrix, ROC, Signed PPV, Coverage (Table 4.1)
│       ├── benchmarks/       # SOS pathway replication & parameter sweeps
│       ├── agents/           # Multi-agent autonomous SDLC orchestration
│       ├── web/              # phpIoRN 2.0 HTTP server, SQLite DB, Graphviz SVG & UI
│       └── cli.py            # CLI entry point
└── tests/
    ├── test_grn.py           # Core mathematical & algorithmic pytest suite
    └── test_playwright_e2e.py # Full Playwright browser + SQLite E2E test
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
