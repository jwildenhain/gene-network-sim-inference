# GRN-Sim: Gene Regulatory Network Perturbation Simulator & Reverse Engineering Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance Python framework for non-linear gene regulatory network (GRN) perturbation simulation and network topology inference. 

This package is a modern, modular reverse-engineering and expansion of the foundational research published by **Jan Wildenhain** and **Edmund J. Crampin**, stemming from the Bachelor of Honours Thesis:
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

## 🔬 Core Capabilities

### 1. In Silico Non-Linear GRN Perturbation Simulator
* **Network Topologies**:
  * **Random (Erdős–Rényi)**: Constant attachment probability.
  * **Scale-Free (Barabási–Albert)**: Preferential attachment yielding power-law degree distributions $P(k) \sim k^{-\gamma}$.
  * **Hierarchical Modular**: Recursive scale-invariant motif duplication with peripheral rewiring.
* **Non-linear Hill Kinetics**:
  $$\frac{dx_i}{dt} = v_i \prod_{j \in \text{Inh}} \left(\frac{K_{ij}^{m_j}}{I_j^{m_j} + K_{ij}^{m_j}}\right) \prod_{k \in \text{Act}} \left(1 + \frac{A_k^{m_k}}{A_k^{m_k} + Ka_k^{m_k}}\right) - b_i x_i + u_i$$
* **Multi-experiment Perturbation Protocols**:
  * Steady-state pre-equilibration ($\frac{dx_i}{dt} = 0$).
  * Systematic single-gene transcriptional perturbations (overexpression & down-regulation).
  * Dynamic ODE integration using adaptive Runge-Kutta / Radau solvers (`scipy.integrate.solve_ivp`).
  * Combined noise model: biological parameter variability ($\sim \mathcal{N}(0, \sigma_b^2)$) and experimental Gaussian measurement noise ($\sim \mathcal{N}(0, \sigma_m^2)$).

### 2. Network Inference & Reverse Engineering Algorithms
* **S2G (Simple-to-General)**:
  * Greedy forward basis selection followed by backward pruning.
  * Parsimony-guided model selection using Akaike Information Criterion (AIC) and Minimum Description Length (MDL).
* **G2S (General-to-Specific)**:
  * Systematic backward elimination starting from full connectivity, minimizing AIC.
* **NIR (Network Identification by multiple Regression)**:
  * Top-$D$ beam search over sparse regulatory subsets with Tikhonov ridge regression (Gardner et al., *Science* 2003).
* **FAST**:
  * Singular Value Decomposition (SVD) and pseudo-inverse truncation for underdetermined or ill-conditioned systems.
* **Voting Ensemble**:
  * Strict consensus operator combining S2G and G2S predictions, suppressing false positive rates while maintaining high precision.
* **By-Chance**:
  * Monte Carlo null model for statistical significance benchmarking.

### 3. Benchmarks & Real Biological Data
* Direct replication of the **9-gene *E. coli* SOS DNA Repair Pathway** from Gardner et al. (*Science* 2003).
* Re-creation of **Table 4.1** from the original thesis: exact performance comparisons across S2G, G2S, NIR, and Ensemble models.

### 4. Autonomous Multi-Agent SDLC Architecture
* Integrated collaborative agent system:
  * `SimulationAgent`: Configures topologies, kinetics, and perturbation experiments.
  * `InferenceAgent`: Executes algorithmic inference pipelines and ensembles.
  * `EvaluationAgent`: Computes confusion matrices, ROC metrics, coverage, and signed PPV.
  * `CoordinatorAgent`: Orchestrates parameter sweeps and automated report generation.

---

## 🏛 Historical Background: From phpIoRN to `grn-sim`

In 2004–2005, an early web implementation named **phpIoRN / phpMyIoRN** (*Inference of Regulatory Networks*) was built to interface with MATLAB in the background. The PHP frontend gathered user parameters, spawned headless MATLAB workers (`matlab -nodisplay -nojvm -nosplash`), and dynamically rendered SVG/PNG network graphs via Graphviz (`dot` / `neato`) alongside PHP-formatted confusion matrices.

This modern Python package completely supersedes the legacy MATLAB/PHP stack with a native, zero-dependency, pure-Python architecture, parallel execution support, and modern scientific computing standards (`numpy`, `scipy`).

---

## 🚀 Installation

```bash
git clone https://github.com/jwildenhain/gene-network-sim-inference.git
cd gene-network-sim-inference
pip install -e .
```

To install development dependencies (pytest):
```bash
pip install -e ".[dev]"
```

---

## 💻 Quickstart

### Python API

```python
import numpy as np
from grn_sim.core.topology import TopologyGenerator
from grn_sim.core.network import GeneRegulatoryNetwork
from grn_sim.simulator.experiment import PerturbationExperiment
from grn_sim.inference.s2g import S2GInference
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.evaluation.metrics import BenchmarkMetrics

# 1. Generate a Scale-Free network of 10 genes
topo = TopologyGenerator(random_state=42)
adj = topo.generate_scale_free(num_genes=10, k_average=2.0, act_ratio=0.5)

# 2. Simulate steady-state perturbation experiment
grn = GeneRegulatoryNetwork(adj)
experiment = PerturbationExperiment(grn, noise_level=0.05, random_state=42)
dataset = experiment.run_perturbation_series()

# 3. Infer network structure using S2G
s2g = S2GInference()
inferred_matrix = s2g.infer(dataset.expression_ratios, dataset.perturbation_matrix)

# 4. Evaluate performance against ground truth
metrics = BenchmarkMetrics.compute(grn.adjacency_matrix, inferred_matrix)
print(f"Coverage: {metrics['coverage']:.2%}")
print(f"False Positive Rate: {metrics['false_positive_rate']:.2%}")
print(f"Signed PPV: {metrics['signed_ppv']:.2%}")
```

### Command Line Interface (CLI)

```bash
# Run the interactive end-to-end prototype demo
python3 demo.py

# Replicate the E. coli SOS Pathway Benchmark (Table 4.1)
grn-sim sos-benchmark

# Run an automated multi-agent simulation sweep
grn-sim multi-agent --genes 15 --topology power_law --k-av 2.0 --noise 0.05

# Run test suite
pytest -v
```

---

## 📊 Verification & Test Suite

The test suite validates the mathematical solver, Hill kinetics, each inference algorithm, and the *E. coli* SOS pathway replication:

```bash
pytest
```
```text
============================= test session starts ==============================
collected 14 items

tests/test_grn.py::test_network_topologies PASSED                        [  7%]
tests/test_grn.py::test_kinetics_evaluation PASSED                       [ 14%]
tests/test_grn.py::test_ode_steady_state PASSED                          [ 21%]
tests/test_grn.py::test_perturbation_experiment PASSED                   [ 28%]
tests/test_grn.py::test_noise_models PASSED                              [ 35%]
tests/test_grn.py::test_s2g_inference PASSED                             [ 42%]
tests/test_grn.py::test_g2s_inference PASSED                             [ 50%]
tests/test_grn.py::test_nir_inference PASSED                             [ 57%]
tests/test_grn.py::test_fast_inference PASSED                            [ 64%]
tests/test_grn.py::test_voting_ensemble PASSED                           [ 71%]
tests/test_grn.py::test_by_chance PASSED                                 [ 78%]
tests/test_grn.py::test_evaluation_metrics PASSED                        [ 85%]
tests/test_grn.py::test_sos_pathway_benchmark PASSED                     [ 92%]
tests/test_grn.py::test_multi_agent_system PASSED                         [100%]

============================== 14 passed in 3.42s ==============================
```

---

## 📁 Repository Structure

```text
gene-network-sim-inference/
├── LICENSE                   # MIT License
├── README.md                 # Complete documentation & citation
├── SPECIFICATION.md          # 32 KB comprehensive mathematical reverse-engineering spec
├── pyproject.toml            # Package configuration (PEP 517/518)
├── demo.py                   # Full interactive demo script
├── data/
│   ├── results_gapa_20021209_and_mmc_dbl.mat  # Gardner 2003 SOS pathway dataset
│   └── SOSdata.txt
├── src/
│   └── grn_sim/
│       ├── core/             # Network models & topology generators
│       ├── simulator/        # Hill kinetics, ODE solver, noise models, perturbation experiment
│       ├── inference/        # S2G, G2S, NIR, FAST, ByChance, Voting Ensemble
│       ├── evaluation/       # Confusion matrix, ROC, Signed PPV, Coverage
│       ├── benchmarks/       # SOS pathway replication & parameter sweeps
│       ├── agents/           # Multi-agent autonomous SDLC system
│       └── cli.py            # CLI entry point
└── tests/
    └── test_grn.py           # Comprehensive pytest suite
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
