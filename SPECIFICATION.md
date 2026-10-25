# Specification: In Silico Gene Regulatory Network Simulation and Reverse-Engineering

This specification reverse-engineers the mathematical formulations, network topologies, simulation protocols, noise models, inference algorithms, evaluation metrics, and biological benchmark datasets from Jan Wildenhain's Bachelor of Science Honours Thesis: *"In Silico Modelling of Gene Networks: A Reverse Engineering Approach"* (University of Edinburgh, 2004/2005) and its original MATLAB codebase.

---

## 1. Differential Equation Kinetics & Hill Terms

### 1.1 Mathematical Formulation

The biological simulation environment models gene expression dynamics using coupled, continuous, non-linear Ordinary Differential Equations (ODEs). The abundance of mRNA transcript $x_i(t)$ of gene $i \in \{1, \dots, n\}$ is governed by the balance between regulated transcription synthesis $s_i(\mathbf{x})$ and linear mass-action degradation:

$$\frac{d x_i}{dt} = s_i(x_1, \dots, x_n) - b_i x_i \tag{Eq 2.2}$$

where:
- $x_i \ge 0$ is the continuous concentration of mRNA transcript for gene $i$.
- $b_i > 0$ is the first-order degradation/breakdown rate constant for transcript species $i$.
- $s_i(\mathbf{x})$ is the non-linear transcription synthesis rate law incorporating all incoming regulatory interactions.

### 1.2 Transcription Rate Law with Hill Saturation

Transcription is mediated by a discrete, saturable number of transcription complexes. Activators and inhibitors modulate the basal transcription rate $v_i$ via multiplicative Hill-type saturation terms:

$$s_i(x_1, \dots, x_n) = v_i \cdot \prod_{j \in \text{Inh}(i)} \left( \frac{\alpha_{ji}^{m_{ji}}}{I_j^{m_{ji}} + \alpha_{ji}^{m_{ji}}} \right) \times \prod_{k \in \text{Act}(i)} \left( 1 + \frac{A_k^{m_{ki}}}{A_k^{m_{ki}} + \beta_{ki}^{m_{ki}}} \right) + u_i \tag{Eq 2.3}$$

where:
- $v_i$: Basal transcription rate of gene $i$ in the absence of external regulation.
- $\text{Act}(i) = \{k \mid T_{ki} = +1\}$: Set of upstream activating genes regulating target gene $i$.
- $\text{Inh}(i) = \{j \mid T_{ji} = -1\}$: Set of upstream repressing/inhibiting genes regulating target gene $i$.
- $A_k = x_k$: Concentration of activator species $k$.
- $I_j = x_j$: Concentration of inhibitor species $j$.
- $\alpha_{ji}$ (or $K_{I, ji}$ in MATLAB): Repressor half-saturation constant (concentration of $I_j$ at which inhibition is half-maximal).
- $\beta_{ki}$ (or $K_{A, ki}$ in MATLAB): Activator half-saturation constant (concentration of $A_k$ at which activation is half-maximal).
- $m_{ji}, m_{ki} \ge 1$: Hill coefficients governing the cooperativity / sigmoidicity of regulatory curves ($m = 1$ is hyperbolic Michaelis-Menten; $m > 1$ yields sigmoidal cooperativity).
- $u_i$: External transcriptional perturbation input applied to gene $i$.

### 1.3 Kinetic Parameter Distributions and Ranges

The total number of kinetic parameters in a network of $n$ genes and $l$ directed edges is $2(n + l)$. In the simulation environment, parameters are drawn from bounded uniform and normal distributions:

| Parameter | Symbol (Thesis) | Variable (MATLAB) | Thesis Range (Table 2.3) | MATLAB Implementation Range (`create_network.m`) |
|---|---|---|---|---|
| Basal transcription rate | $v_i$ | `v(i, 1)` | $\mathcal{U}[1.0, 2.0]$ | $1.05 + 0.10 \times \mathcal{N}(0, 1)$ |
| mRNA degradation rate | $b_i$ | `B(i, 1)` | $\mathcal{U}[0.0, 1.0]$ | $0.75 + 0.10 \times \mathcal{N}(0, 1)$ |
| Activator half-saturation | $\beta_{ki}$ | `Ka(j, k)` | $\mathcal{U}[1.0, 2.0]$ | $1.0 + 1.5 \times \mathcal{U}[0, 1] \in [1.0, 2.5]$ |
| Inhibitor half-saturation | $\alpha_{ji}$ | `Ki(j, k)` | $\mathcal{U}[1.0, 2.0]$ | $1.0 + 1.5 \times \mathcal{U}[0, 1] \in [1.0, 2.5]$ |
| Activator Hill coefficient | $m_{ki}$ | `n_k(j, k)` | $\mathcal{U}[0.0, 5.0]$ | $1.0 + 5.0 \times \mathcal{U}[0, 1] \in [1.0, 6.0]$ |
| Inhibitor Hill coefficient | $m_{ji}$ | `n_i(j, k)` | $\mathcal{U}[0.0, 5.0]$ | $1.0 + 5.0 \times \mathcal{U}[0, 1] \in [1.0, 6.0]$ |
| Initial mRNA concentrations | $\mathbf{s}_0$ | `series(:, 1)` | $\mathcal{U}[10.0, 20.0]$ | $0.5 + \mathcal{U}[0, 1] \in [0.5, 1.5]$ |

### 1.4 Multi-Paradigm Simulation Dynamics Engines (`src/grn_sim/simulator/model_engine.py`)

To systematically compare mechanistic modeling against modern neural and empirical techniques, `phpIoRN 2.0` implements four distinct simulation dynamics paradigms:

1. **Non-linear Hill ODE (`hill_nonlinear`)**:
   Solves the classical coupled differential equations (Eq 2.2, 2.3) using Runge-Kutta 4/5 (`scipy.integrate.solve_ivp` with RK45) until the steady-state convergence norm $\|\mathbf{x}(t+1)^2 - \mathbf{x}(t)^2\|_2 \le 10^{-4}$ is reached.
2. **CellBox Continuous Neural ODE (`cellbox_neural_ode`)**:
   Simulates continuous relaxation dynamics according to:
   $$\frac{d x_i}{dt} = \tanh\left( \sum_{j=1}^n W_{ij} x_j + b_i + u_i \right) - \alpha_i x_i$$
   where $W_{ij} = \gamma T_{ij}$, degradation constants $\alpha_i \approx 1.0$, and saturation bounds prevent unphysical explosion.
3. **Linear Dynamical System (`linear_jacobian`)**:
   Computes analytical steady states via direct matrix inversion of the Jacobian:
   $$\mathbf{x} = (\mathbf{B} - \mathbf{A})^{-1} (\mathbf{v} + \mathbf{u})$$
   where $\mathbf{B} = \text{diag}(b_1, \dots, b_n)$ and $\mathbf{A} = \gamma \mathbf{T}^T$. Guarantees stable convergence when spectral radius $\rho(\mathbf{B}^{-1} \mathbf{A}) < 1$.
4. **Single-Cell Perturb-seq Count Simulator (`perturb_seq_counts`)**:
   Generates discrete integer single-cell UMI counts for 400 cells per condition (unperturbed control $+ n$ single-gene perturbations) under biological Gamma-Poisson overdispersion and log-normal sequencing depth variance ($s_c \sim \text{LogNormal}(9.9, 0.2)$). Output format strictly adheres to standard AnnData `.h5ad` structures matching 10x Genomics and VCC 2026.

---

## 2. Network Topology Models

Network connectivity is defined by an interaction matrix $\mathbf{T} \in \{-1, 0, +1\}^{n \times n}$.
**Matrix Convention:** In the simulation code (`netsim.m`, `create_network.m`), $T_{jk} = T(j, k)$ defines the directed effect of **regulator gene $j$** on **target gene $k$**:
- $T_{jk} = +1$: Gene $j$ activates gene $k$.
- $T_{jk} = -1$: Gene $j$ inhibits gene $k$.
- $T_{jk} = 0$: No direct transcriptional regulatory interaction.

Total directed interactions $k_{total} = \sum_{j, k} |T_{jk}|$, average connectivity $k_{av} = k_{total} / n$, and graph density $\zeta = k_{av} / n$. The ratio parameter $r \in [0, 1]$ modulates the fraction of activating edges (default $r \approx 0.52$).

### 2.1 Random Erdős-Rényi Networks (`nettype = 0`)
- **Degree distribution:** Poisson distribution $P(k) \approx \frac{\lambda^k e^{-\lambda}}{k!}$.
- **Construction algorithm (`netsim.m`):**
  1. Base assignment: For each gene $i \in \{1, \dots, n\}$, pick a random source regulator $j \sim \lceil n \cdot \mathcal{U}(0, 1) \rceil$ and set $T_{ji} = \text{sign}(\mathcal{U}(0, 1) - r)$.
  2. Edge expansion: While $n + i \le \lceil \frac{k_{av}}{2} \cdot n \rceil$, pick two nodes $j_1, j_2 \sim \lceil n \cdot \mathcal{U}(0, 1) \rceil$ and assign $T_{j_1, j_2} = \text{sign}(\mathcal{U}(0, 1) - r)$.

### 2.2 Scale-Free Barabási-Albert Networks (`nettype = 1`)
- **Degree distribution:** Power law $P(k) \propto k^{-\gamma}$ (with empirical exponent $\gamma \in [2.0, 3.0]$).
- **Preferential attachment:** The probability of node $i$ receiving a new link is proportional to its cumulative degree:
  $$p_i = \frac{d_k(i)}{\sum_{j=1}^n d_k(j)}$$
- **Construction algorithm (`netsim.m`):**
  1. Initialize cumulative degree array $d_k = \mathbf{1}_{n \times 1}$.
  2. First pass: For each target gene $i \in \{1, \dots, n\}$, sample candidate regulator $j \sim \lceil n \cdot \mathcal{U}(0, 1) \rceil$. Accept if $\frac{d_k(j)}{\sum d_k} > \mathcal{U}(0, 1)$. Set $T_{ji} = \text{sign}(r - \mathcal{U}(0, 1))$, increment $d_k(j) \leftarrow d_k(j) + 1$. Repeat until accepted.
  3. Additional edges: While total interactions $< \lceil \frac{k_{av}}{2} \cdot n \rceil$, sample pairs $(j_1, j_2)$. Connect with edge $T_{j_1, j_2} = \text{sign}(r - \mathcal{U}(0, 1))$ if both satisfy preferential attachment $\frac{d_k(j_1)}{\sum d_k} > \mathcal{U}(0, 1)$ and $\frac{d_k(j_2)}{\sum d_k} > \mathcal{U}(0, 1)$. Increment $d_k(j_1)$ and $d_k(j_2)$.

### 2.3 Hierarchical Modular Networks (`nettype = 2`)
- **Characteristics:** Scale-free degree distribution combined with high clustering coefficient and nested modularity, representing evolutionary gene duplication and divergence.
- **Construction algorithm (`netsim.m`):**
  1. Module division: Divide $n$ nodes into $S$ subdivisions (`subdiv`, default $S = 4$). Subnetwork size $M = n / S$.
  2. Base subnetwork: Construct a scale-free module of size $M$ using preferential attachment.
  3. Module duplication (cloning): Replicate the base connectivity block across the $S$ sub-blocks:
     $$T_{:, ((k-1)M + 1):(kM)} = T_{:, 1:M}, \quad \text{for } k = 2, \dots, S$$
  4. Inter-module communication: Add cross-module edges between distinct modules $s_1 \neq s_2$ by preferential attachment sampling on global degree $d_k$ until total degree target $\lceil \frac{k_{av}}{2} \cdot n \rceil$ is satisfied.

---

## 3. Steady-State Integration and Perturbation Simulation Protocol

### 3.1 Numerical Integration & Convergence Criterion
ODEs are discretized using forward Euler integration with step size $\Delta t = 1.0$:

$$x_k(t + 1) = x_k(t) + \left[ v_k \prod_{j \in \text{Inh}} \left( \frac{K_{I, jk}^{n_I}}{I_j^{n_I} + K_{I, jk}^{n_I}} \right) \prod_{l \in \text{Act}} \left( 1 + \frac{A_l^{n_A}}{A_l^{n_A} + K_{A, lk}^{n_A}} \right) - b_k x_k(t) \right]$$

- **Convergence threshold:** Steady state is declared when the squared Euclidean displacement satisfies:
  $$\| \mathbf{x}(t + 1)^2 - \mathbf{x}(t)^2 \|_2 \le \epsilon_{thrs} = 10^{-4}$$
- **Stability verification:** If any concentration $x_k(t) \le 0$ or convergence is not achieved within $t_{max} = n^2 + 100$ steps, the parameter set is rejected as non-stable and regenerated.
- **Reference steady state:** Recorded as vector $\mathbf{x}^{ss} \in \mathbb{R}^n$.

### 3.2 Perturbation Protocol & "Tagged mRNA" Implementation

To infer network connections without requiring dynamic time series, the system applies steady-state perturbations:
1. For each gene $j \in \{1, \dots, n\}$, an external transcriptional perturbation $p_j$ (default magnitude $p = 1.0$) is applied.
2. **Tagged mRNA Mechanism:** In a physical system, applying an external transcriptional perturbation directly into $\frac{dx_j}{dt}$ creates an artificial imbalance between $v_j$, $b_j$, and $u_j$, which naive regression algorithms misclassify as self-regulatory loops ($T_{jj} \neq 0$). To prevent this artifact, the simulation introduces "tagged" mRNA species:
   - The perturbation $P_j$ is added directly to the regulator concentration entering downstream Hill kinetics:
     $$A_j = x_j(t) + P_j, \quad I_j = x_j(t) + P_j$$
   - The unperturbed self-dynamics $\frac{dx_j}{dt} = s_j - b_j x_j$ remain governed by the intrinsic endogenous rate laws.
3. The perturbed system is integrated to its new steady state $\mathbf{x}^{ss, p_j}$.

### 3.3 Linearized Steady-State System & Ratio Calculation

Around the reference steady state $\mathbf{x}^{ss}$, a first-order Taylor expansion yields:

$$\frac{d \mathbf{x}}{dt} \approx \mathbf{A} \Delta \mathbf{x} + \mathbf{u} = \mathbf{0} \implies \mathbf{A} \mathbf{X} = -\mathbf{U} \tag{Eq 3.8}$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times n}$ is the relative expression ratio matrix:
  $$X_{ij} = \frac{x_i^{ss, p_j}}{x_i^{ss}} - 1 \tag{Eq 3.9}$$
  Here, column $j$ corresponds to the expression response profile across all $n$ genes when gene $j$ is perturbed.
- $\mathbf{U} \in \mathbb{R}^{n \times n}$ is the perturbation matrix:
  $$U_{ij} = P_j \cdot \delta_{ij}$$
  (diagonal matrix of normalized perturbation amplitudes).
- $\mathbf{A} \in \mathbb{R}^{n \times n}$ is the Jacobian/interaction coefficient matrix to be reverse-engineered.

---

## 4. Noise and Variance Models

To test robustness against realistic high-throughput experimental variation, the simulation environment incorporates biological variability and analytical measurement noise.

### 4.1 Biological Variability (`do_biovariability.m`)
Simulates genetic polymorphism, growth differences, and micro-environmental fluctuations by perturbing the intrinsic kinetic parameters across experimental repeats ($r = 1, \dots, \kappa$):

$$\theta_{perturbed} = \theta_{base} + \theta_1 \cdot \sqrt{\frac{p_{bio} \cdot \theta_{base}}{100}}$$

where $\theta_1 \sim \mathcal{N}(0, 1)$ and $p_{bio}$ is the biological variance percentage (default 10%, $f = 1$). This perturbation is applied independently to $v_i$, $b_i$, $K_{I, ji}$, $K_{A, ki}$, $m_{ji}$, and $m_{ki}$. Across $\kappa$ repeated runs, empirical concentration variance is recorded:

$$\sigma_{bio, ij}^2 = \text{Var}_r(x_{ij}^{(r)})$$

### 4.2 Analytical Measurement Noise (`netsim.m`, `do_perturbation.m`)
Simulates microarray hybridization error, spatial effects, dye biases, and detector noise via a composite normal-uniform Gaussian error model:

$$\tilde{x}_i = x_i \cdot \left[ 1 + \left( \sqrt{\delta} \cdot \theta_1 \cdot (\theta_2 - 0.5) \right) \right] \tag{Eq 2.5}$$

where:
- $\theta_1 \sim \mathcal{N}(0, 1)$ (bounded in $[-4, +4]$).
- $\theta_2 \sim \mathcal{U}[0, 1]$ (uniform random scale factor centered at zero).
- $\delta$ is the noise variance parameter ($\varepsilon = \delta$ for measurement noise, $\gamma = \delta$ for perturbation error).

### 4.3 Empirical Variance Calculation
For downstream regression weighting (e.g. in NIR), the variance of measurement discrepancies is assigned as:

$$\sigma_{ij}^2 = \begin{cases}
|\tilde{x}_{ij} - x_{ij}|^2 & \text{if } |\tilde{x}_{ij} - x_{ij}| < 1 \\
\sqrt{|\tilde{x}_{ij} - x_{ij}|} & \text{if } |\tilde{x}_{ij} - x_{ij}| \ge 1
\end{cases}$$

---

## 5. Reverse Engineering Algorithms

The objective of all inference algorithms is to estimate the interaction matrix $\mathbf{A}$ from the experimental data matrices $\mathbf{X}$ and $\mathbf{U}$, solving:

$$\mathbf{A} \mathbf{X} + \mathbf{U} \approx \mathbf{0} \iff \min_{\mathbf{A}} \| \mathbf{A} \mathbf{X} + \mathbf{U} \|_F^2$$

### 5.1 S2G: Simple-to-General Algorithm (`s2g_and_g2s.m`, `getvar.m`)
Adapted from Crampin et al. (2004) and Judd & Mees (1995). S2G is an iterative forward-selection basis pursuit algorithm that requires no *a priori* assumption on the number of regulatory inputs per gene $k$. It solves the network row-by-row:

$$\mathbf{y}_i = \mathbf{U}(i, :)' = \mathbf{X}^T \mathbf{a}_i$$

1. **Initialization:** Active basis index set $\Omega = \emptyset$, iteration $k = 1$, residual $\mathbf{E}^{(0)} = -\mathbf{y}_i$.
2. **Normalized Basis Matrix:** Let $\mathbf{V} = \mathbf{X}^T \mathbf{D}_x^{-1}$, where column norms are scaled to unit length ($D_{x, jj} = \| \mathbf{X}_{j, :} \|_2$).
3. **Projection & Inclusion:** Project the current reconstruction error $\mathbf{E}^{(k-1)}$ onto the pool of all basis functions:
   $$\boldsymbol{\mu} = | \mathbf{V}^T \mathbf{E}^{(k-1)} |$$
   Select index with maximum absolute projection:
   $$i_{in} = \arg\max_{j} \mu_j, \quad \Omega \leftarrow \Omega \cup \{ i_{in} \}$$
4. **Candidate Elimination (Backtracking):** Compute Moore-Penrose pseudo-inverse solution on current basis set:
   $$\mathbf{a}_\Omega = (\mathbf{V}_\Omega)^\dagger \mathbf{y}_i$$
   Identify candidate with minimum absolute magnitude:
   $$i_{out} = \arg\min_{j \in \Omega} |a_j|$$
   If $i_{out} \neq i_{in}$, prune $i_{out}$: $\Omega \leftarrow \Omega \setminus \{ i_{out} \}$ and re-evaluate projection.
5. **Cost Function Evaluation:** Compute model residuals $\mathbf{E}^{(k)} = \mathbf{V}_\Omega \mathbf{a}_\Omega - \mathbf{y}_i$. Evaluate cost:
   $$S(k) = N \ln\left( \frac{1}{N} \sum (E_j^{(k)})^2 \right) + k \ln N \quad \text{(MDL / BIC Penalty)}$$
   $$\text{or } AIC(k) = \frac{1}{N} (\mathbf{E}^{(k)})^T \mathbf{E}^{(k)} + k \tag{Eq 3.14}$$
6. **Termination:** If $S(k) < S(k - 1)$, increment $k \leftarrow k + 1$ and continue. If $S(k) \ge S(k - 1)$, terminate and return basis $\Omega^{(k-1)}$ with coefficients $\mathbf{a} = (\mathbf{X}_\Omega^T)^\dagger \mathbf{y}_i$.

### 5.2 G2S: General-to-Specific Algorithm (`s2g_and_g2s.m`, `gen2spec.m`)
G2S operates in reverse, starting from a fully connected model and performing greedy backward elimination with AIC penalization:

1. **Initialization:** Start with full basis set $\Omega = \{1, \dots, n\}$, model size $k = n$.
2. **Initial Fit:** Compute $\boldsymbol{\alpha} = (\mathbf{X}_\Omega^T)^\dagger \mathbf{y}_i$, residual $\mathbf{E}^{(n)} = \mathbf{X}_\Omega^T \boldsymbol{\alpha} - \mathbf{y}_i$, and initial cost:
   $$AIC(n) = \sum (\mathbf{E}_j^{(n)})^2 + n$$
3. **Pruning Step (`dropanother`):** For each candidate $j \in \Omega$:
   - Temporarily remove $j$: $\Omega_{-j} = \Omega \setminus \{ j \}$.
   - Fit reduced model $\mathbf{a}_{-j} = (\mathbf{X}_{\Omega_{-j}}^T)^\dagger \mathbf{y}_i$, residual $\mathbf{e}_{-j}$.
   - Compute mean squared error $\text{MSE}_j = \text{mean}(\mathbf{e}_{-j}^2)$.
   Select the basis whose removal causes the minimal damage to the fit:
   $$j^* = \arg\min_{j} \text{MSE}_j, \quad \Omega \leftarrow \Omega \setminus \{ j^* \}, \quad k \leftarrow k - 1$$
4. **Stopping Rule:** Evaluate $AIC(k) = \sum (E_j^{(k)})^2 + k$. If $AIC(k) < AIC(k + 1)$, continue pruning. As soon as $AIC(k) \ge AIC(k + 1)$, stop and select model $\Omega^{(k+1)}$.

### 5.3 NIR: Network Identification by Multiple Regression (`nir_v3.m`)
Published by Gardner et al. (*Science*, 2003). Assumes each gene is regulated by exactly $K$ incoming transcription factors ($K = k_{av}$, typically $K \in [2, 5]$).

1. **Model Formulation:** Solves $\mathbf{y}_i = \mathbf{b}_i^T \mathbf{Z} + \boldsymbol{\delta}_i^T$, where $\mathbf{y}_i = -\mathbf{u}_i$.
2. **Search Strategy:** Uses a greedy forward selection search with top-$D$ beam width (`Forward-TopD-ReEstK`, default $D = n$):
   - At stage $l \in \{1, \dots, K\}$, expand each of the current $D$ best candidate subsets by adding each remaining gene.
   - For each combination of $l$ predictors, form subset data matrix $\mathbf{Z} \in \mathbb{R}^{l \times m}$.
3. **Ridge Regression & Regularization:**
   $$\tilde{\mathbf{b}}_i = -\mathbf{u}_i \mathbf{Q} \mathbf{Z}^T (\mathbf{Z} \mathbf{Q} \mathbf{Z}^T + \lambda \mathbf{I})^{-1}$$
   where $\mathbf{Q} = \mathbf{W} \mathbf{W}^T$ is an experimental weight matrix, $\lambda$ is the ridge parameter (default $\lambda = 0$).
4. **Condition Number Filtering:** Subsets where $\text{cond}(\mathbf{Z} \mathbf{Q} \mathbf{Z}^T) > \text{CONDMAX} = 1000$ are rejected as ill-conditioned.
5. **Selection:** Evaluates Sum of Squared Errors:
   $$SSE_K = \sum_{j=1}^m \left( u_{ij} + \tilde{\mathbf{b}}_i^T \mathbf{z}_j \right)^2$$
   Retains the $D$ combinations with lowest $SSE$. The top combination at $l = K$ is selected for row $i$: $\mathbf{A}(i, \text{topD}_1) = \tilde{\mathbf{b}}_i$.
6. **Unperturbed Gene Handling:** If gene $i$ was not perturbed ($u_i = 0$), NIR extracts row $i$ as response $\mathbf{y} = -\mathbf{X}(i, :)$ and sets diagonal self-decay $A_{ii} = -1$.

### 5.4 FAST: Singular Value Decomposition Algorithm (`fast_modified.m`)
Published by Farina et al. (2003). Solves the inverse problem via truncated SVD and sparse row thresholding:

1. **SVD Decomposition:** Compute thin SVD of expression matrix $\mathbf{X} \in \mathbb{R}^{n \times m}$:
   $$\mathbf{X} = \mathbf{U}_s \mathbf{S} \mathbf{V}^T$$
2. **Truncated Rank-$(n-1)$ Moore-Penrose Inverse:**
   Drop the smallest singular value (last column of $\mathbf{U}_s, \mathbf{V}$ and last diagonal entry of $\mathbf{S}$) to form $\mathbf{U}_r, \mathbf{S}_r, \mathbf{V}_r$:
   $$\mathbf{A}_0 = (\mathbf{X} - \mathbf{U}) \mathbf{V}_r \mathbf{S}_r^{-1} \mathbf{U}_r^T - \mathbf{I}_{n}$$
3. **Row Sparsification:** For each row $i \in \{1, \dots, n\}$:
   - Rank the absolute magnitudes $|A_{0, ij}|$.
   - Select the $K$ indices with the largest absolute magnitude: $\mathcal{I}_K(i)$.
   - Re-estimate non-zero weights via pseudo-inverse on selected rows:
     $$\mathbf{A}_{est}(i, \mathcal{I}_K(i)) = -\mathbf{u}(i, :) \cdot \left( \mathbf{X}(\mathcal{I}_K(i), :) \right)^\dagger$$
   - All other $n - K$ entries in row $i$ are set strictly to zero.

### 5.5 By-Chance: Statistical Null Model (`by_chance.m`)
Generates random interaction networks to determine the null performance baseline:
1. Inputs: Network size $n$, target interaction degree $k$, activation ratio $r$.
2. Target number of edges: $\iota = \lceil k \cdot n \rceil$.
3. Sampling: For $l = 1, \dots, \iota$, choose random source $i \sim \mathcal{U}\{1, n\}$ and random target $j \sim \mathcal{U}\{1, n\}$. Assign directed sign:
   $$A_{rand}(i, j) = \text{sign}(r - \mathcal{U}(0, 1))$$
4. In benchmark evaluations, By-Chance statistics are averaged over 10,000 Monte Carlo realizations.

### 5.6 Voting Ensemble (`do_vote.m`)
Combines S2G and G2S to eliminate false positives:
- Qualitative network analysis demonstrates that S2G and G2S have distinct, uncorrelated false-positive edge profiles.
- **Consensus Rule:**
  $$\mathbf{A}_{vote}(i, j) = \begin{cases}
  \text{sign}(A_{s2g}(i, j)) & \text{if } \text{sign}(A_{s2g}(i, j)) == \text{sign}(A_{g2s}(i, j)) \\
  0 & \text{otherwise}
  \end{cases}$$
- An interaction is accepted if and only if both S2G and G2S agree on its existence and sign.

### 5.7 CellBox: Continuous Neural ODE Relaxation (`src/grn_sim/inference/cellbox.py`)
Infers the regulatory interaction matrix $\mathbf{W} \in \mathbb{R}^{n \times n}$ by modeling continuous dynamical relaxation:
$$\frac{dx_i}{dt} = \phi\left( \sum_{j=1}^n W_{ij} x_j + u_i \right) - \alpha_i x_i$$
At steady state, the concentration satisfies $x_i = \frac{1}{\alpha_i} \phi(\sum_j W_{ij} x_j + u_i)$.
- **Loss Function:**
  $$\mathcal{L}(W, \alpha) = \frac{1}{M} \sum_{m=1}^M \|\hat{\mathbf{x}}_m - \mathbf{x}_m\|_2^2 + \lambda_1 \|W\|_1 + \lambda_2 \|W\|_2^2$$
- Implemented with native PyTorch automatic differentiation (or SciPy L-BFGS-B fallback) with cosine annealing learning rate schedule and threshold discretization.

### 5.8 GENIE3: Regression Tree Ensemble (`src/grn_sim/inference/genie3.py`)
Decomposes the network reverse-engineering problem into $n$ separate non-linear feature importance regressions (Huynh-Thu et al., *PLoS ONE* 2010):
- For each target gene $i$, a tree ensemble (`ExtraTreesRegressor` / `RandomForestRegressor`) predicts $x_i$ from all potential regulators $\mathbf{x}_{-i}$:
  $$x_i = f_i(\mathbf{x}_{-i}) + \epsilon_i$$
- Regulatory edge weights correspond to feature importances $w_{ji} = \text{MDI}(x_j \to x_i)$.
- Directional signs are determined via the empirical perturbation sensitivity:
  $$\text{sign}(A_{ji}) = \text{sign}\left( \text{Corr}(x_j, x_i) \right)$$

### 5.9 Network Deconvolution (ND) (`src/grn_sim/inference/network_deconv.py`)
Analytically separates direct causal regulatory interactions from indirect transitive correlations (Feizi et al., *Nat Biotechnol* 2013):
- Given an observed symmetric or directed correlation matrix $\mathbf{G}_{obs}$:
  $$\mathbf{G}_{dir} = \mathbf{G}_{obs} (\mathbf{I} + \mathbf{G}_{obs})^{-1}$$
- Solved via eigen-decomposition $\mathbf{G}_{obs} = \mathbf{V} \mathbf{\Lambda}_{obs} \mathbf{V}^T$, where direct eigenvalues satisfy $\lambda_{dir} = \frac{\lambda_{obs}}{1 + \lambda_{obs}}$.
- Edge signs and directions are preserved by projecting back onto the perturbation response matrix.

### 5.10 Kaggle Multi-Output RidgeCV (`src/grn_sim/inference/ridge_cv.py`)
Multi-target $L_2$-regularized regression baseline reproducing the 1st-place approach in modern perturbation challenges:
$$\min_{\mathbf{A}} \|\mathbf{X} \mathbf{A}^T + \mathbf{U}\|_2^2 + \alpha \|\mathbf{A}\|_F^2$$
- Performs cross-validated grid search over $\alpha \in [10^{-3}, 10^3]$ to optimize out-of-sample generalization error.
- Enforces row-wise sparsity truncation to top-$K$ strongest interactions per gene.

### 5.11 CPA-Lite: Latent Perturbation Autoencoder (`src/grn_sim/inference/cpa_lite.py`)
Compositional Perturbation Autoencoder (Lotfollahi et al., *Nat Biotechnol* 2023) modeling transcriptional response in a low-dimensional bottleneck:
- **Encoder:** $\mathbf{z} = \mathbf{E}(\mathbf{x}) \in \mathbb{R}^d$ ($d \ll n$).
- **Latent Arithmetic:** $\mathbf{z}_{pert} = \mathbf{z}_{ctrl} + \Delta \mathbf{z}_j$.
- **Decoder:** $\hat{\mathbf{x}} = \mathbf{D}(\mathbf{z}_{pert})$.
- The effective gene regulatory influence matrix is derived from the sensitivity Jacobian:
  $$\mathbf{A}_{est} = \mathbf{D}_{linear}^T \mathbf{E}_{linear}$$

### 5.12 AtlasShift: Cross-Atlas Covariate Transfer (`src/grn_sim/inference/atlasshift.py`)
Empirical transfer algorithm (Kaipeng Meng, Arc Institute VCC 2026 Leaderboard #82) estimating target response by transferring observed differential expression shifts across cell atlases:
$$\hat{\mathbf{y}}_{\text{dest}} = \bar{\mathbf{y}}_{\text{dest, ctrl}} + \sum_{s \in \text{sources}} w_s \cdot (\bar{\mathbf{y}}_{s, \text{pert}} - \bar{\mathbf{y}}_{s, \text{ctrl}})$$
- Excellent in-distribution accuracy when target perturbations are pre-observed in source atlases.
- Provides a direct empirical baseline against mechanistic ODE models for OOD generalization and rewiring analysis.

---

## 6. Error Statistics and Metrics

Performance is evaluated by comparing the reverse-engineered network $\mathbf{A}_R$ against the true underlying network $\mathbf{A}_T$ using binary classification and signed edge metrics. A filter threshold $\delta = 0.001$ suppresses numerical noise ($|A_{ij}| \le \delta \implies A_{ij} = 0$).

### 6.1 Confusion Matrix Definition
All $n^2$ possible directed interactions fall into four cardinal counts:
- **True Positives (TP, $results(3)$):** An interaction exists in both true and reconstructed networks, irrespective of sign:
  $$TP = \sum_{i, j} \mathbf{1}(|A_{T, ij}| > \delta \text{ and } |A_{R, ij}| > \delta)$$
- **False Positives (FP, $results(2)$):** An edge is inferred where none exists:
  $$FP = \sum_{i, j} \mathbf{1}(|A_{T, ij}| \le \delta \text{ and } |A_{R, ij}| > \delta)$$
- **True Negatives (TN, $results(1)$):** Absence of edge correctly identified:
  $$TN = \sum_{i, j} \mathbf{1}(|A_{T, ij}| \le \delta \text{ and } |A_{R, ij}| \le \delta)$$
- **False Negatives (FN, $results(4)$):** An existing true edge is omitted:
  $$FN = \sum_{i, j} \mathbf{1}(|A_{T, ij}| > \delta \text{ and } |A_{R, ij}| \le \delta)$$
- **Signed True Positives (STP, $results(5)$):** Inferred interaction matches both existence and sign:
  $$STP = \sum_{i, j} \mathbf{1}(|A_{T, ij}| > \delta \text{ and } |A_{R, ij}| > \delta \text{ and } \text{sign}(A_{T, ij}) == \text{sign}(A_{R, ij}))$$
- **Yeung Total Errors ($E_{total}$, $results(6)$):** Total discrepancies:
  $$E_{total} = \sum_{i, j} \mathbf{1}(|\text{sign}(A_{R, ij}) - \text{sign}(A_{T, ij})| \ge 1)$$

### 6.2 The Eight Standard SDLC Metrics (`new_stats.m`)
The function `new_stats(A_true, A_silico)` returns the 8-element vector:

| Metric Index | Name | Formula | Interpretation |
|---|---|---|---|
| `stat(1)` | False Positive Rate (FPR) | $\frac{FP}{TN + FP}$ | Fall-out (fraction of zero edges misclassified as active) |
| `stat(2)` | False Negative Rate (FNR) | $\frac{FN}{TP + FN}$ | Miss rate (fraction of true edges missed) |
| `stat(3)` | Specificity | $1 - \text{FPR} = \frac{TN}{TN + FP}$ | True Negative Rate |
| `stat(4)` | Power / Sensitivity | $1 - \text{FNR} = \frac{TP}{TP + FN}$ | Recall / True Positive Rate |
| `stat(5)` | Positive Predictive Value (PPV) | $\frac{TP}{TP + FP}$ | Precision (Gardner Coverage ratio) |
| `stat(6)` | False Discovery Rate (FDR) | $\frac{FP}{TP + FP} = 1 - \text{PPV}$ | Fraction of inferred edges that are false |
| `stat(7)` | Signed PPV | $\frac{STP}{TP + FP}$ | Precision accounting for correct sign (+ / -) |
| `stat(8)` | Total Number of Errors | $E_{total} = FP + FN + \text{WrongSign}$ | Absolute error count across all $n^2$ entries |

### 6.3 Thesis Percentage Metrics (Chapter 4 & Table 5.1)
In the thesis summary tables (e.g. Table 5.1), metrics are expressed in normalized percentage forms:
1. **Coverage %:**
   $$\text{Coverage \%} = 100 \times \frac{TP + TN}{n^2}$$
2. **False Positives %:**
   $$\text{False Positives \%} = 100 \times \frac{FP}{n^2}$$
3. **Inferred Connections % (Recall / Sensitivity):**
   $$\text{Inferred Connections \%} = 100 \times \frac{TP}{k_{total}}$$
   where $k_{total} = TP + FN$ is the total number of true interactions.
4. **False Positives per Inferred Connection % ($100 \times \text{FDR}$):**
   $$\text{FP per Inferred Connection \%} = 100 \times \frac{FP}{\hat{k}_{total}} = 100 \times \frac{FP}{TP + FP}$$
   where $\hat{k}_{total} = TP + FP$ is the total number of edges inferred by the algorithm.
5. **Inferred Sign %:**
   $$\text{Inferred Sign \%} = 100 \times \frac{STP}{n^2}$$
6. **Total Errors:**
   $$E = \sum_{i, j} \mathbf{1}(|\text{sign}(A_{R, ij}) - \text{sign}(A_{T, ij})| \ge 1)$$

### 6.4 $3 \times 3$ Multi-Class Misclassification Table (`stats_obs_pred.m`)
Evaluates classification across three discrete states: Activator ($+1$), Repressor ($-1$), and No Interaction ($0$):

$$\mathbf{M}_{MT} = \begin{bmatrix}
N(+1 \to +1) & N(+1 \to -1) & N(+1 \to 0) \\
N(-1 \to +1) & N(-1 \to -1) & N(-1 \to 0) \\
N(0 \to +1) & N(0 \to -1) & N(0 \to 0)
\end{bmatrix}$$

- Rows indicate **Observed True Class** ($\text{Obs}+$, $\text{Obs}-$, $\text{Obs } 0$).
- Columns indicate **Predicted Inferred Class** ($\text{Pred}+$, $\text{Pred}-$, $\text{Pred } 0$).
- Class error rate for row $i$:
  $$\text{Error}_i = \frac{\sum_{j} M_{ij} - M_{ii}}{\sum_j M_{ij}}$$
- Total misclassification error:
  $$\text{Total Error} = \frac{\sum_{i \neq j} M_{ij}}{\sum_{i, j} M_{ij}}$$

---

## 7. E. coli SOS Pathway 9-Gene Benchmark Data Specification

The biological benchmark dataset is derived from Gardner et al. (*Science*, 2003), examining the SOS DNA repair transcriptional network in *Escherichia coli*.

### 7.1 Gene Index Ordering
The network consists of $n = 9$ genes in strict canonical order:
1. `recA`: RecA recombinase / coprotease (central SOS activator/sensor)
2. `lexA`: LexA repressor (master transcriptional repressor)
3. `ssb`: Single-stranded DNA-binding protein
4. `recF`: DNA replication / repair protein RecF
5. `dinI`: DNA damage-inducible protein I
6. `umuDC`: Error-prone DNA polymerase V subunit complex
7. `rpoD`: RNA polymerase sigma 70 factor ($\sigma^{70}$, housekeeping)
8. `rpoH`: RNA polymerase sigma 32 factor ($\sigma^{32}$, heat shock)
9. `rpoS`: RNA polymerase sigma 38 factor ($\sigma^S$, stationary phase)

### 7.2 Ground Truth Matrix $\mathbf{A}_t$ (`SOS_statistic.m`)
The literature-curated ground truth interaction matrix $\mathbf{A}_t \in \{-1, 0, +1\}^{9 \times 9}$ (49 total true edges: 29 activating, 20 inhibitory, 32 zero):

```
       recA  lexA   ssb  recF  dinI umuDC  rpoD  rpoH  rpoS
recA  [  +1,   -1,   -1,   +1,   +1,   -1,   +1,    0,    0 ]
lexA  [  +1,   -1,   -1,   +1,   +1,   -1,   +1,    0,    0 ]
ssb   [  +1,   -1,   -1,   +1,   +1,   -1,   +1,    0,    0 ]
recF  [   0,    0,    0,    0,    0,    0,   +1,    0,   +1 ]
dinI  [  +1,   -1,   -1,   +1,   +1,   -1,   +1,    0,    0 ]
umuDC [  +1,   -1,   -1,   +1,   +1,   -1,   +1,    0,    0 ]
rpoD  [  +1,   -1,   -1,   +1,   +1,   -1,   +1,   +1,    0 ]
rpoH  [   0,    0,    0,    0,    0,    0,   +1,   +1,    0 ]
rpoS  [   0,    0,    0,    0,    0,    0,   +1,    0,   +1 ]
```

### 7.3 Gardner et al. 2003 Published Solution $\mathbf{A}_g$
Estimated using the NIR algorithm with $K = 5$:

```
A_g = [
 -0.597, -0.179, -0.010,  0.000,  0.096,  0.000, -0.011,  0.000,  0.000 ;
  0.387, -1.670, -0.014,  0.000,  0.087, -0.068,  0.000,  0.000,  0.000 ;
  0.044, -0.189, -1.275,  0.000,  0.053,  0.000,  0.027,  0.000,  0.000 ;
 -0.181,  0.238, -0.025, -1.000, -0.055,  0.000,  0.000,  0.000,  0.390 ;
  0.281,  0.000,  0.000,  0.000, -2.094,  0.156, -0.037,  0.012,  0.000 ;
  0.112, -0.403, -0.016,  0.000,  0.205, -1.147,  0.000,  0.000,  0.000 ;
 -0.171,  0.000, -0.017,  0.000,  0.025,  0.000, -1.513,  0.021,  0.000 ;
  0.096,  0.000,  0.001,  0.000, -0.009, -0.031,  0.000, -0.483,  0.000 ;
  0.217,  0.000,  0.000, -1.678,  0.672,  0.000,  0.077,  0.000, -3.921
];
```

### 7.4 Quantitative Real-Time PCR Experimental Data

The experimental dataset is loaded from `results_gapa_20021209_and_mmc_dbl.mat` containing structures `results.r`, `results.sr`, `results.p`, and `results.sp`.

#### 7.4.1 Expression Ratios Matrix $\mathbf{X}$ (`results.r`)
Relative expression fold ratios $x_{ij} = [mRNA]_{ij}^{perturbed} / [mRNA]_i^{unperturbed}$:

| Target Gene \ Perturbed | `recA` | `lexA` | `ssb` | `recF` | `dinI` | `umuDC` | `rpoD` | `rpoH` | `rpoS` |
|---|---|---|---|---|---|---|---|---|---|
| **`recA`** | 1.905555 | 0.868345 | 0.861054 | 1.186925 | 1.291093 | 0.939370 | 0.923314 | 0.983293 | 0.975068 |
| **`lexA`** | 1.212287 | 1.383155 | 0.883135 | 1.063567 | 1.168792 | 0.913342 | 1.038531 | 1.124702 | 1.083575 |
| **`ssb`** | 1.017714 | 0.893287 | 11.52350 | 1.060467 | 1.079840 | 1.012476 | 1.063861 | 1.088825 | 0.929870 |
| **`recF`** | 1.103823 | 0.950045 | 0.727520 | 1.139419 | 1.179683 | 1.145695 | 1.068899 | 0.995759 | 1.274612 |
| **`dinI`** | 1.118732 | 0.902862 | 1.055664 | 1.314605 | 3.147100 | 1.141627 | 0.931561 | 1.134836 | 1.113060 |
| **`umuDC`** | 1.076417 | 0.810607 | 0.875685 | 1.249595 | 1.347226 | 3.017441 | 0.932769 | 0.827960 | 0.977645 |
| **`rpoD`** | 0.878146 | 0.952870 | 0.898286 | 0.892920 | 0.989381 | 1.103995 | 4.067558 | 1.365330 | 1.216845 |
| **`rpoH`** | 1.177758 | 0.817517 | 1.035472 | 0.930456 | 0.966034 | 0.845280 | 1.007877 | 27.63329 | 1.086940 |
| **`rpoS`** | 1.071939 | 0.872117 | 1.073434 | 1.080616 | 1.304674 | 1.051433 | 0.938818 | 1.273816 | 1.672422 |

#### 7.4.2 Expression Standard Deviations $s\mathbf{X}$ (`results.sr`)
Measured standard deviations on qPCR expression ratios:

```
sX = [
 0.128087, 0.107428, 0.080026, 0.112162, 0.057145, 0.076654, 0.057313, 0.103571, 0.098259 ;
 0.092085, 0.180196, 0.074687, 0.088206, 0.066735, 0.078399, 0.058426, 0.119696, 0.108590 ;
 0.071265, 0.101732, 0.677265, 0.088730, 0.060119, 0.103974, 0.057429, 0.094889, 0.075958 ;
 0.094876, 0.117260, 0.096967, 0.103167, 0.068745, 0.100204, 0.069642, 0.100852, 0.135552 ;
 0.095750, 0.111124, 0.101212, 0.119705, 0.186694, 0.095780, 0.063630, 0.125603, 0.117924 ;
 0.095376, 0.112673, 0.093491, 0.115817, 0.101965, 0.270635, 0.063834, 0.077507, 0.095653 ;
 0.062354, 0.123976, 0.081875, 0.135945, 0.089026, 0.123075, 0.258915, 0.163781, 0.184236 ;
 0.062511, 0.104144, 0.103186, 0.085720, 0.054683, 0.090677, 0.059327, 3.606553, 0.119693 ;
 0.082181, 0.108274, 0.130515, 0.118449, 0.095676, 0.089666, 0.062591, 0.198186, 0.255500
];
```

#### 7.4.3 Perturbation Matrix $\mathbf{P}$ (`results.p`) and Standard Deviations $s\mathbf{P}$ (`results.sp`)
Non-zero diagonal entries corresponding to episomal over-expression plasmid perturbations:

| Gene | Perturbation Magnitude $P_{ii}$ | Perturbation Std Dev $sP_{ii}$ |
|---|---|---|
| `recA` | 0.652884 | 0.175194 |
| `lexA` | 1.171144 | 0.200329 |
| `ssb` | 13.412000 | 0.288356 |
| `recF` | 1.670500 | 0.694000 |
| `dinI` | 4.541497 | 1.163493 |
| `umuDC` | 2.355475 | 0.522731 |
| `rpoD` | 4.708279 | 0.956570 |
| `rpoH` | 12.865759 | 3.116857 |
| `rpoS` | 4.108893 | 1.045356 |

All off-diagonal perturbation entries are identically 0.

### 7.5 Benchmark Performance Summary on SOS Data (Thesis Table 5.1)

| Evaluation Metric | S2G | G2S | FAST | NIR | By-Chance (Null) |
|---|---|---|---|---|---|
| **Coverage %** | 56.79% | 38.27% | 55.56% | 62.96% | 50.47% $\pm$ 5.26% |
| **False Positives %** | 8.64% | 16.05% | 19.75% | 16.05% | 15.43% $\pm$ 2.84% |
| **Inferred Connections %** | 42.86% | 24.49% | 59.18% | 65.31% | 43.63% $\pm$ 5.12% |
| **FP per Inferred Connection %** | 25.00% | 52.00% | 35.56% | 28.89% | 36.89% $\pm$ 6.29% |
| **Inferred Sign %** | 12.35% | 6.17% | 23.46% | 25.93% | 13.66% $\pm$ 5.40% |
| **Total Number of Errors** | 46.00 | 57.00 | 46.00 | 41.00 | 50.44 $\pm$ 3.90 |

---

## 8. Architectural & Implementation Guidelines for Modern Python Port

To port this simulation and inference framework to a modern, robust scientific Python package (e.g. `numpy`, `scipy`, `networkx`, `pytest`):

1. **Topology Generation:** Implement functional network builders returning adjacency matrices with strict validation of degree sequence, connected components, and power-law distribution exponents.
2. **ODE Solver:** Implement both vectorized Euler integration (`scipy.integrate.odeint` / `solve_ivp` with `RK45` or `LSODA`) and the exact thesis discrete Euler step for bit-for-bit backward verification.
3. **Tagged mRNA:** Structure the simulation API so perturbation vectors $\mathbf{u}$ are decoupled from intrinsic basal rates, strictly replicating the tagged mRNA protocol.
4. **Vectorized Inference:**
   - Implement `S2G` and `G2S` using `scipy.linalg.pinv` and `scipy.linalg.lstsq`.
   - Implement `NIR` with greedy forward beam search and ridge shrinkage.
   - Implement `FAST` with truncated SVD (`scipy.linalg.svd`).
   - Implement `VotingEnsemble` consensus operator.
5. **Validation Suite:**
   - Unit tests against the exact Gardner 2003 9-gene matrices $\mathbf{X}$, $\mathbf{P}$, $\mathbf{A}_t$, and $\mathbf{A}_g$.
   - Regression verification matching the outputs of `new_stats.m` and `stats_obs_pred.m`.

---

## 9. `phpIoRN 2.0` Web Application, Oscillation Diagnostics & MATLAB Numerical Equivalence

### 9.1 Original `phpIoRN` MATLAB Bridge Workflow (`toronto_backup2/thesis/`)
- **`get_network.php` $\rightarrow$ `web_getparam.m` + `web_netsimsim.m`**:
  Reads environment variables (`SIM_NETSIZE`, `SIM_NETTYPE`, `SIM_CON`, `SIM_ARATIO`, `MATLAB_FILENAME`), calls `netsim.m` + `create_network.m`, exports initial Graphviz DOT topology (`graph_to_dot.m`), and saves `graphs/mat/<id>.m`.
- **`run_sim.php` $\rightarrow$ `web_run_p_model.m` + `web_run_rgn.m`**:
  1. `web_run_p_model.m` iterates $j = 1 \dots N$, runs `do_perturbation(test, n, N^2+100, j, 1, 1, 0, 0)`, renders SVG perturbation curves (`plot2svg`), writes `graphs/node/<id>_<j>.php`, and exports `.msmt` and `.pert` ASCII tab-separated matrices.
  2. `web_run_rgn.m` runs selected algorithms (`s2g.m`, `g2s.m`, `do_vote.m`, `nir_v3.m`, `fast_modified.m`) and calls `php_and_stats.m` to write `$TMP_STATMAT` and `$TMP_CMAT` PHP arrays alongside `statsgraph_to_dot.m` 9-state DOT files.
- **Persistent SQLite Archive (`dumpdb.sql` $\rightarrow$ `database.py`)**:
  Persists all runs in `data/rgn_db.sqlite` across `nw` (`name`, `type`, `size`, `con`, `aratio`, `biovar`, `noise`, `time`, `ip`) and `alg` (`nw_id`, `alg`, `fpr`, `fnr`, `spec`, `sens`, `ppv`, `fdr`, `sppv`, `note`).

### 9.2 Oscillation & Steady-State Detection (`create_network.m` & `do_perturbation.m`)
1. **Unperturbed Background Convergence (`create_network.m`)**:
   Integrates over horizon $e = s^2 + 100$ steps and checks:
   $$\text{efnorm}(i) = \left\| \mathbf{x}(i+1)^2 - \mathbf{x}(i)^2 \right\|_2 \le 10^{-4}, \quad \min(\mathbf{x}) > 0$$
2. **Perturbed Trajectory Convergence (`do_perturbation.m`)**:
   Checks both $\text{efnorm}(t) \le 10^{-4}$ and $\left| \text{efnorm}(t) - \text{efnorm}(t-1) \right| < 10^{-4}$ (`not_stable == 0`).
3. **Limit-Cycle & Damped Oscillation Spectral Diagnostics (`ode_solver.py`)**:
   Evaluates the Jacobian $\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ at the terminal state and computes eigenvalues $\lambda_i = \alpha_i \pm i\omega_i$ plus sign-flips in $dx_i/dt$ over the last 30% of the trajectory to classify regimes into `STEADY_STATE`, `DAMPED_OSCILLATION`, `LIMIT_CYCLE_OSCILLATION`, or `UNSTABLE_DIVERGENT`.

---

## 10. 100% MATLAB Numerical Equivalence & Benchmark Archive Matrix

To guarantee exact reproducibility with the original 2004 MATLAB codebase and published benchmark curves, the Python implementation adheres to the following bit-for-bit mathematical and data-structure conventions:

### 10.1 Algorithm Formulation Details

#### 10.1.1 S2G (`s2g.m` & `getvar.m`)
- **Column Normalization**:
  The candidate regulator basis matrix $X$ is normalized column-wise by Euclidean 2-norm:
  $$B = \left[ \frac{X_{*, 1}}{\|X_{*, 1}\|_2}, \dots, \frac{X_{*, N}}{\|X_{*, N}\|_2} \right]$$
- **Log MDL Penalty**:
  For target gene $i$, given selected candidate index set $\mathcal{I}_k$ of size $k$:
  $$\mathbf{a}_k = B_{*, \mathcal{I}_k}^{\dagger} \, (-P_{i, *}^T)$$
  $$\mathbf{e} = B_{*, \mathcal{I}_k} \mathbf{a}_k - (-P_{i, *}^T)$$
  $$C_{\text{MDL}}(k) = N \ln\left( \frac{1}{N} \sum_{m=1}^N e_m^2 \right) + k \ln(N)$$
- **Forward Swap Loop (`getanother`)**:
  Once a candidate $j^*$ is chosen by forward greedy selection, an inner single-variable replacement loop tests swapping each variable in $\mathcal{I}_k$ with any remaining variable in $\{1 \dots N\} \setminus \mathcal{I}_k$.
- **Stopping Rule & Unconditional Trim**:
  Forward selection terminates when $C_{\text{MDL}}(k) \ge C_{\text{MDL}}(k-1)$. In accordance with `getvar.m` line 80, the variable added in the terminal rejected step is unconditionally pruned: $\mathcal{I} \leftarrow \mathcal{I}_{1 \dots k-1}$.
- **Storage Convention**:
  The unnormalized regression coefficients are stored into column $i$ of the estimated Jacobian:
  $$A_{\text{est}}[\mathcal{I}, i] = \mathbf{a}_0$$
  matching the system convention $T_{j, k}$ (regulator in row $j$, target in column $k$).

#### 10.1.2 G2S (`g2s.m` & `gen2spec.m`)
- **Backward Elimination**:
  Starts with full connectivity $\mathcal{I} = \{1 \dots N\}$.
- **AIC Penalty**:
  $$C_{\text{AIC}}(k) = \sum_{m=1}^N e_m^2 + k$$
- **Single-Deletion Swap Optimization (`dropanother`)**:
  At each elimination step, the variable whose removal yields the lowest residual sum of squares is pruned, followed by a swap check (`dropanother`).
- **Acceleration**:
  For $N \le 25$, exact refitting via `lstsq` is performed. For $N > 25$, Sherman-Morrison rank-1 downdating is used to evaluate all candidate leave-one-out models in $O(N^2)$ time.

#### 10.1.3 FAST (`fast.m` & `fast_modified.m`)
- **SVD Pseudo-Inverse**:
  Given SVD of the perturbation matrix $X = U_r S_r V_r^T$:
  $$A_0 = (X - U) V_r S_r^{-1} U_r^T - I_N$$
  For experimental benchmarks where self-degradation is embedded in the background state (e.g. Gardner 2003 *E. coli* SOS), `fast_modified.m` adds $+ I_N$.
- **Sparsity Truncation**:
  For each gene, only the top $k_{\text{interactions}}$ coefficients with largest absolute magnitude are retained; all others are set to 0.

#### 10.1.4 NIR (`nir_v3.m`)
- **Top-$D$ Greedy Search**:
  Iterates over candidate subsets of size $k_{\text{interactions}} \le 3$, ranking subsets by error-in-variables residual norm.
- **Tie-Breaking**:
  Identical mergesort sorting order ensures identical subset selection to MATLAB when singular values or SSE values are tied.

---

### 10.2 Empirical Verification Matrix

The Python implementation is verified against both published experimental datasets and stored 2004 MATLAB simulation archives:

#### A. 9-Gene *E. coli* SOS DNA Repair Pathway (`tests/test_inference.py`)
Matches Thesis Table 4.1 (`chap4.tex` lines 169–175) and Appendix Figures A.4–A.7 (`TablesAppendix.tex` lines 50–116):

| Algorithm | Coverage % | False Positives % | Inferred Connections % | FDR (FP/Inferred %) | Inferred Sign % | Total Errors | $3 \times 3$ Confusion Matrix Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S2G** | 56.79% | 8.64% | 42.86% | 25.00% | 12.35% | **46** | `[[6, 8, 17], [3, 4, 11], [2, 5, 25]]` |
| **G2S** | 38.27% | 16.05% | 24.49% | 52.00% | 6.17% | **57** | `[[5, 4, 22], [3, 0, 15], [5, 8, 19]]` |
| **FAST** | 55.56% | 19.75% | 59.18% | 35.56% | 23.46% | **46** | `[[16, 4, 11], [6, 3, 9], [13, 3, 16]]` |
| **NIR** | 62.96% | 16.05% | 65.31% | 28.89% | 25.93% | **41** | `[[17, 7, 7], [4, 4, 10], [11, 2, 19]]` |

#### B. 60-Gene Synthetic Benchmark Run (`random_nz60_run5_iot2_sig0.mat`)
Matches stored MATLAB `pred_matrix` vectors bit-for-bit:
- **S2G**: `pred_matrix_sg[:, -1] = [22, 0, 70, 0, 33, 59, 5, 0, 3411]` (Exact match)
- **NIR**: `pred_matrix_gardener[:, -1] = [15, 2, 99, 2, 22, 40, 10, 9, 3401]` (Exact match)

#### C. 25-File `overallstats/` Grid
Archives in `data/matlab_benchmarks/overallstats/` spanning $k_{\text{av}} \in \{1.0, 1.5, 2.0, 3.0, 5.0\}$ and $\sigma \in \{0.0, 0.001, 0.0025, 0.005, 0.01\}$ across network sizes $N \in \{10, 20, 30, 40, 50, 60, 70\}$. All metrics (`Power`, `FDR`, `TotErrors`, `Runtime`) are sanitized with `np.nan_to_num` and served through the interactive inspector in Tab 3.

---

## 11. Single-Cell Perturb-seq Simulator & Virtual Cell Challenge (VCC 2026) Specification

### 11.1 Non-Linear ODE-to-Single-Cell Count Emission Pipeline

The single-cell simulation module (`src/grn_sim/simulator/perturb_seq.py`) bridges continuous ODE steady-state concentrations directly into discrete, single-cell integer UMI counts matching 10x Genomics Chromium Perturb-seq and Arc Institute VCC 2026 specifications ($400$ cells per condition).

Given an $N$-gene network with unperturbed steady state $\mathbf{x}^{\text{ss}}$ and single-gene transcriptional perturbation steady states $\mathbf{x}^{p, j}$ for $j \in \{1, \dots, N\}$, the deterministic expression profile matrix is:

$$\mathbf{M} = \big[\mathbf{x}^{\text{ss}}, \quad \mathbf{x}^{p, 1}, \quad \mathbf{x}^{p, 2}, \quad \dots, \quad \mathbf{x}^{p, N}\big] \in \mathbb{R}^{N \times (N+1)}$$

For each condition $k \in \{0, \dots, N\}$ (where $k=0$ is unperturbed control and $k \ge 1$ is perturbation of gene $k$), the relative expression probability distribution is normalized:

$$\mathbf{p}_k = \frac{\mathbf{M}_{:, k}}{\sum_{i=1}^N M_{ik}}$$

### 11.2 Gamma-Poisson Overdispersion & Sequencing Depth Model

Each individual single cell $c \in \{1, \dots, C\}$ (with $C = 400$) is generated through a hierarchical Gamma-Poisson emission model:

1. **Cell Sequencing Depth (Library Size)**:
   Sequencing depth $s_c$ exhibits log-normal biological and technical capture variation:
   $$s_c \sim \text{LogNormal}\left(\ln(\mu_{\text{depth}}) - \frac{1}{2}\sigma_{\text{depth}}^2, \, \sigma_{\text{depth}}\right)$$
   where $\mu_{\text{depth}} = 20{,}000$ UMIs and $\sigma_{\text{depth}} = 0.35$.

2. **Gene Expression Probability Perturbation (Gamma Biological Shock)**:
   Gene-specific expression probabilities fluctuate due to bursty transcription and transcriptional noise:
   $$\lambda_{c, i} \sim \text{Gamma}\left(\text{shape} = \alpha, \, \text{scale} = \frac{p_{k, i}}{\alpha}\right), \quad \alpha = \frac{1}{\phi_{\text{dispersion}}}$$
   where $\phi_{\text{dispersion}} = 0.15$ governs overdispersion (Negative-Binomial parameterization). The sampled vector is re-normalized such that $\sum_{i=1}^N \lambda_{c, i} = 1$.

3. **Discrete UMI Count Sampling**:
   $$Y_{c, i} \sim \text{Poisson}(\lambda_{c, i} \cdot s_c), \quad Y_{c, i} \in \mathbb{N}_0$$

4. **Counts Per Million (CPM) Layer**:
   $$\text{CPM}_{c, i} = \frac{Y_{c, i}}{\sum_{j=1}^N Y_{c, j}} \times 10^6$$

### 11.3 AnnData (`.h5ad`) Data Schema

The resulting dataset is packed into a standard `anndata.AnnData` object:
- **`X`**: `scipy.sparse.csr_matrix` of shape $(C \cdot (N+1), N)$ containing integer counts $Y_{c, i}$.
- **`layers['cpm']`**: Sparse normalized CPM matrix.
- **`obs`** (`pd.DataFrame`):
  - `cell_barcode`: Unique identifier `f"{condition}_cell_{c:04d}"`.
  - `perturbation`: Perturbed gene symbol (or `"non-targeting"` for control).
  - `condition`: Condition name (`"control"`, `"Gene_01"`, etc.).
  - `is_control`: Boolean flag.
  - `total_counts`: Integer library size $\sum_i Y_{c, i}$.
  - `detected_genes`: Integer number of non-zero genes.
- **`var`** (`pd.DataFrame`):
  - `gene_name`: Gene identifier.
  - `is_perturbed_target`: Boolean flag.
  - `mean_control_cpm`: Mean baseline CPM.
- **`uns`** (`dict`):
  - `ground_truth_network`: True Jacobian adjacency matrix $\mathbf{A}$.
  - `perturbation_matrix`: Transcriptional perturbation matrix $\mathbf{U}$.
  - `cells_per_condition`: $400$.

### 11.4 AtlasShift Empirical Cross-Atlas Perturbation Transfer Algorithm

Reverse-engineered from Kaipeng Meng's Arc Institute VCC 2026 solution (`kaipengm2/Virtual-Cell-Challenge-2026`, Leaderboard #82, Score `0.15456`):
- **Effect Fusion**: Combines differential expression shifts $\Delta \mathbf{y}^{(s)}$ from $S$ source reference cell lines (K562, HCT116, HEK293T) using weights $\mathbf{w}$:
  $$\Delta \mathbf{y}_{\text{fused}} = \frac{\sum_{s=1}^S w_s \Delta \mathbf{y}^{(s)}}{\sum_{s=1}^S w_s}$$
- **Desired Mean Calculation**: Scales target baseline control expression $\mathbf{y}_{\text{ctrl}}$ in log2 fold-change space:
  $$\mathbf{y}_{\text{target}} \propto \mathbf{y}_{\text{ctrl}} \cdot 2^{\gamma \Delta \mathbf{y}_{\text{fused}}}$$
- **Dual-Moment Iterative Count Emission**: Re-scales a template single-cell matrix via iterative biproportional fitting (Sinkhorn-like alternating row and column scaling) to simultaneously match single-cell library depths and target gene mean probabilities.

### 11.5 Comparative Evaluation: Mechanistic GRN vs. Empirical Transfer

Evaluated in `src/grn_sim/benchmarks/vcc_benchmark.py` (`run_vcc_comparative_benchmark`). One scale-free base network is simulated as three source atlases (K562-, HCT116- and HEK293T-like, differing in basal rates) and a **target context** whose edges are rewired by `target_rewire_fraction`.

1. **Edge recovery.** All 18 algorithms are fit on the complete target experiment set and scored against the rewired ground truth (`evaluate_inference`). Algorithms that expect $-\mathbf{U}$ (NIR, FAST, RidgeCV) receive it (`FIT_INPUT_SIGN`).

2. **Response prediction with a genuine hold-out.** The last $\max(2, \lfloor 0.25N \rfloor)$ perturbation targets are *held out*. The response models are refit with those experiments **masked** ($\mathbf{X}_{:,j}=\mathbf{U}_{:,j}=\mathbf{0}$; masking instead of dropping keeps the data square, which S2G/G2S/Vote require, and adds nothing to the least-squares problems). Each model's response operator is
   $$\mathbf{R} = s_m\,(\mathbf{A}_{\text{est}} + \lambda\mathbf{I})^{+}, \qquad \hat{\mathbf{X}}_{:,j} = \mathbf{R}\,\mathbf{u}_j,$$
   with $\lambda = 0.05$ only when the diagonal of $\mathbf{A}_{\text{est}}$ is all zero. The sign $s_m$ differs per algorithm because their $\mathbf{A}_{\text{est}}$ follow different conventions: $+1$ for Vote, S2G, G2S, RidgeCV, NIR and CellBox, $-1$ for DMD-GRN (`RESPONSE_MODELS`). A model whose fit is unusable (all-zero or non-finite $\mathbf{A}_{\text{est}}$, a fit error, or CPA-Lite whose $\mathbf{A}_{\text{est}}$ is a latent projection) is reported in `parameters.response_models[].reason` and **never replaced by another model's predictions**. AtlasShift predicts the source-atlas mean for seen targets and $\mathbf{0}$ for held-out ones. Because a gene's row of $\mathbf{A}$ is identified only by the experiment that perturbs that gene, held-out predictions of the S2G/G2S/Vote family are exactly zero ($r = 0$); regularised models retain a small positive correlation. Reference numbers ($N=16$): Vote seen $r = 0.493$, held-out $r = 0.000$; AtlasShift seen $r = 0.493$, held-out $r = 0.000$.

3. **Context rewiring sensitivity.** Rewiring moves an edge to another target **together with its kinetic parameters** (`_rewire_edges`; parameters of non-edges are zero, so a bare adjacency swap would create edges that regulate nothing). The operators fitted on the full target data are applied to contexts rewired by $\rho \in \{0, 10, 20, 35, 50\}\%$; mean Pearson $r$ over all targets is reported per model. AtlasShift decays from $r = 0.662$ to $0.256$-$0.322$ for $\rho \ge 35\%$; the Vote operator stays near zero ($|r| < 0.08$) and RidgeCV keeps $r = 0.13$-$0.27$.

4. **Double-perturbation (epistasis) screen.** Each pair $G_i+G_j$ is simulated on the true network (perturbation $\mathbf{u}_i+\mathbf{u}_j$) without measurement noise, and
   $$\varepsilon_{ij} = \left\lVert L^{(ij)} - \left(L^{(i)} + L^{(j)}\right) \right\rVert_2, \qquad L = \log_2\!\left(x_{\text{perturbed}}/x_{\text{basal}}\right)$$
   measures deviation from the multiplicative null (independent perturbations multiply fold changes). Log ratios are used because $x/x_{\text{basal}}-1$ lets a lowly expressed gene dominate $\varepsilon$; noise is excluded because it alone puts a floor of about 0.18 under $\varepsilon$. Classes: $\varepsilon \le 0.25$ additive / independent, $0.25 < \varepsilon \le 0.45$ buffering / epistatic shift, $\varepsilon > 0.45$ synergistic / synthetic lethal. Cost: $N(N-1)/2$ ODE solves.

5. **Virtual single-cell sequencer.** Poisson counts (rate $60\,x$, floor $0.2$) for 20 control cells and 15 cells for each of the first 8 perturbation conditions, using the perturbed steady state of the corresponding **column** of `perturbed_states`; 2D PCA of the count matrix.

### 11.5a Held-Out Prediction: Hill Refit, Reference Predictors and the Knockdown Generator

Implemented in `src/grn_sim/benchmarks/hill_response.py` and used by `run_vcc_comparative_benchmark`.

**Why the linear operator is exactly 0.** In $\mathbf{A}\mathbf{X}=-\mathbf{U}$ the row of $\mathbf{A}$ that belongs to gene $j$ is identified only by the experiment that perturbs gene $j$; with that experiment masked the row is zero and $\hat{\mathbf{X}}_{:,j}=0$. The simulator moreover perturbs the *regulator input* of Hill kinetics, a different model class from the additive Gardner model.

**Hill refit (per algorithm).** 1. `infer_signed_structure`: the top-$q$ ($q=2$, `HILL_STRUCTURE_Q`) regulators of every target from $|\mathbf{A}_{\text{est}}|$; both orientations (matrix and transpose) are tried and the one with the smaller *training* error of a linear regulator-input refit $\mathbf{X}=\mathbf{F}(\mathbf{X}+\mathbf{U})$ is kept; signs come from that refit. 2. `HillResponseModel`: for each target $k$, $x_k = c_k\prod_{l}h_l(x_l+u_l)$ with $h=1+\tfrac{e^m}{e^m+K^m}$ (activator) or $\tfrac{K^m}{e^m+K^m}$ (inhibitor); $\log c_k$ is solved in closed form, $K\in[0.05,20]$, $m\in[0.5,8]$ by bounded nonlinear least squares on the seen experiments plus the control. 3. The response to a perturbation is the fixed point $\mathbf{x}=g(\mathbf{x}+\mathbf{u})$ (damped iteration), reported as $\mathbf{x}/\mathbf{x}_{\text{ctrl}}-1$. Held-out columns are never used (tested by corrupting them).

**Reference predictors** (`hill_models()`): `null` = mean response of the seen experiments (leave-one-out for seen columns); `hill_fs` = `hill_forward_selection`, a BIC-guided greedy search (at most 3 regulators per target, activator or inhibitor, 6 correlation-screened candidates per step) matched to the refit model class; `oracle` = true structure and signs (upper bound, not an algorithm); `self_prior` = $-e$ at the perturbed gene and 0 elsewhere (knockdown mode only).

**Perturbation modes** (`PerturbationSimulator(perturbation_mode=...)`, REST parameter `perturbation`): `input` (default, thesis convention): $\mathbf{U}_{jj}=+1$ is added to the regulator concentration of gene $j$; its own transcript is not set. `knockdown` (CRISPRi-like): production of gene $j$ is multiplied by $1-e_j$ (`production_scale` in `evaluate_derivatives`/`ODESolver.solve_with_diagnostics`), $e_j=\text{clip}(\text{knockdown\_efficiency}+\text{efficiency\_spread}\cdot\mathcal{N}(0,1), 0.05, 0.99)$, $\mathbf{U}_{jj}=-e_j$; the benchmark uses $e=0.8$ (`KNOCKDOWN_EFFICIENCY`). `HillResponseModel(mode="knockdown")` then fits $x_k/(1+u_k)=c_k\prod_l h_l(x_l)$. The solver's convergence test uses the perturbed criterion whenever a production scale differs from 1.

**Scores.** Every predicted profile is scored by Pearson $r$, cosine, RMSE and `offtarget_r` ($r$ with the perturbed gene removed from both vectors). In knockdown data the target's own drop is about 77% of the profile norm, so $r$ over all genes mostly measures whether a model knows *which* gene was hit; `offtarget_r` and the `self_prior` row make that visible. `held_out_offtarget_minus_null` is the paired difference to the null on the same network; the UI labels a model *above null* only if its own off-target $r>0.1$ and the difference exceeds $0.1$. A single network is a noisy comparison — see `scripts/holdout_study.py` and `docs/HOLDOUT_RESEARCH.md` for the paired multi-network study.

### 11.6 VCC Benchmark REST Contract

`GET /api/benchmark/vcc_atlasshift?num_genes=12&k_av=1.8&seed=2026&rewire_fraction=0.15&perturbation=input&refresh=0`
- Parameters are validated (`validate_vcc_params`): `num_genes` an integer in $[4, 32]$, `k_av` in $(0, \min(5, N-1)]$, `rewire_fraction` in $[0, 1]$, all finite, and `perturbation` either `input` (default) or `knockdown`. Violations return **HTTP 400** with `{"status": "error", "error": "..."}`; unexpected failures return 500 with `{"error": "..."}`.
- The payload is serialised with `allow_nan=False` before any header is sent, so a failure is a clean 500 and never a 200 with a broken body.
- Results are cached per `(num_genes, k_av, seed, rewire_fraction, perturbation)` (continuous values rounded to 2 decimals), at most 16 entries; concurrent identical requests share one computation; `refresh=1` recomputes.
- Response fields: `summary_table` (18 rows, order of `ALGORITHMS`), `in_distribution` / `out_of_distribution`, `rewire_decay` (an empty series means that model has no usable fit), `target_profiles[]` (`y_true`, `y_atlas`, linear `predictions` / `metrics` and Hill `hill_predictions` / `hill_metrics`, every metrics object with `pearson_r`, `cosine_sim`, `rmse`, `offtarget_r`), `response_table[]` (one row per `method` linear / hill / reference and model: `available`, `reason`, `seen`, `held_out`, `held_out_minus_null`, `held_out_offtarget_minus_null`), `combinatorial_screen` (`epistasis_matrix`, `top_synergy_pairs`, `unsteady_pairs`), `single_cell_pca`, `parameters.response_models[]` and `parameters.hill_models[]` (`key`, `label`, `kind`, `available`, `reason`), from which the UI builds its model dropdown for the selected response method, plus `parameters.perturbation_mode`, `knockdown_efficiency` and `hill_structure_q`. `rewire_decay.hill_r` is the Hill refit on the Voting structure; `in_distribution` / `out_of_distribution` additionally carry `hill_search`, `null` and (out of distribution) `oracle`.

---

## 12. Real-World Benchmark Datasets Catalog Specification (`src/grn_sim/benchmarks/benchmark_datasets.py`)

`phpIoRN 2.0` integrates a comprehensive catalog of 4 benchmark datasets representing microbial, in silico, and single-cell human perturbation paradigms.

### 12.1 E. coli SOS DNA Repair (`ecoli_sos`)
- **Organism:** *Escherichia coli*
- **Dimensions:** $N = 9$ genes, $M = 9$ single-gene perturbation experiments.
- **Gene Set:** `recA, lexA, ssb, recF, dinI, umuDC, rpoD, rpoH, rpoS`.
- **Ground Truth:** 43 directed interactions verified by Gardner et al. (*Science* 2003) and Jan Wildenhain Honours Thesis (Table 4.1).
- **Perturbations:** Normalized relative mRNA expression changes under UV-irradiation and single-gene plasmid overexpression.

### 12.2 DREAM4 In Silico Gold Standard Network (`dream4_10gene`)
- **Organism:** In silico synthetic benchmark generated from *E. coli* and *S. cerevisiae* subgraphs.
- **Dimensions:** $N = 10$ genes, $M = 10$ knockout perturbation conditions.
- **Gene Set:** `G1, G2, G3, G4, G5, G6, G7, G8, G9, G10`.
- **Ground Truth:** 11 directed interactions including transcriptional feedback loops (e.g. $G_{10} \to G_1$, $G_5 \dashv G_1$).
- **Dynamics:** Non-linear biochemical simulation with cooperative binding and translation delays.

### 12.3 Norman et al. 2019 Hematopoiesis TF Circuit (`norman_2019_tf16`)
- **Organism / Cell Line:** Human K562 erythroleukemic cell line.
- **Dimensions:** $N = 16$ genes, $M = 16$ single/combinatorial CRISPRa perturbation conditions.
- **Gene Set:** Key hematopoietic lineage specification transcription factors:
  `GATA1, SPI1 (PU.1), CEBPA, KLF1, RUNX1, TAL1, LMO2, ZFPM1 (FOG1), GATA2, ETV6, IKZF1, MEF2C, FOXO1, EP300, CREBBP, MYB`.
- **Ground Truth:** 25 curated directed regulatory interactions capturing the canonical GATA1-SPI1 mutual antagonism circuit governing erythroid vs. myeloid cell fate decisions.
- **AnnData:** Accompanied by matched 400 cells/condition single-cell count matrices.

### 12.4 Replogle et al. 2022 Core p53 Circuit (`replogle_2022_p53`)
- **Organism / Cell Lines:** Human K562 (chronic myelogenous leukemia) vs. hTERT-RPE1 (retinal pigment epithelial cells).
- **Dimensions:** $N = 12$ genes, $M = 12$ CRISPRi essential-gene perturbations.
- **Gene Set:** Core p53 pathway and cell cycle checkpoint regulators:
  `TP53, MDM2, CDKN1A (p21), ATM, CHEK2, BAX, BBC3 (PUMA), PMAIP1 (NOXA), GADD45A, CDK2, CCNE1, RB1`.
- **Ground Truth:** 23 directed causal interactions describing p53 activation, MDM2 negative feedback autoregulation, and downstream CDK2/Rb arrest cascades.
- **Cross-Cell Line Tasks:** Matches the multi-context task formulation of the Virtual Cell Challenge 2026.

### 12.5 Benchmark Catalog REST API Contract
Accessible via:
- `GET /api/benchmark/catalog`: Returns JSON array of all registered benchmarks with metadata.
- `GET /api/benchmark/dataset?id={dataset_id}`: Returns complete expressions $\mathbf{X}$, perturbations $\mathbf{U}$, true matrix $\mathbf{A}_{\text{true}}$, Graphviz DOT/SVG layouts, and 12-algorithm comparison metrics.
- `POST /api/benchmark/load_to_studio`: Loads any benchmark dataset into the Tab 1 interactive simulation studio for real-time ODE inspection and parameter exploration.

---

## 13. Network Graph Views & Built-in Layered Layout

### 13.1 One view chooser for every network graph
Six graphs share the same **Choose your view** bar (`registerGraphPanel` in `static/index.html`): Step 3 ground truth, Step 6 9-state graph, Real-World Benchmarks (known network, 9-state reconstruction) and Previous Simulations (archived ground truth, archived 9-state graph).

| Button | Content | Notes |
| :--- | :--- | :--- |
| **Force Canvas** | circular HTML5 canvas drawn from the graph JSON (`nodes`, `links`) | click selects a node; the selected node is highlighted |
| **Hierarchical** | SVG with regulators above their targets | **Layout: Layered \| Grid** switch |
| **Spring** | SVG (Graphviz `neato`, or a circular layout without Graphviz) | |
| **Graph file** | the Graphviz DOT text | Step 3 only |

The Layered/Grid choice is global, stored in `localStorage['grn.hierLayout']`, and applied to every hierarchical graph. Only the current view's SVG is kept in the page (never two copies), and view buttons keep their previous element ids (`btn-gt-dot`, `btn-inf-neato`, ...) for existing scripts.

### 13.2 SVG production (`visualization/graph_export.py`)
`render_graphviz_svg(dot, engine, matrix, gene_names, is_9state, layout="layered"|"grid")`:
- `layered` with the Graphviz binary installed: `dot -Tsvg`. Without it: the built-in layered layout below.
- `grid`: a compact staggered grid drawn by the built-in renderer (never Graphviz).
- Built-in drawing: shafts run between node outlines, arrow heads are drawn explicitly (triangle `data-arrowhead="normal"` for activation, perpendicular bar `data-arrowhead="tee"` for inhibition; 9-state edges use their own style), reciprocal edges are offset side by side, self-loops sit beside the node (above it in the circular layout), and nodes use Graphviz markup (`<g class="node"><title>name</title>`) so the page's click and highlight code is identical for both renderers.

### 13.3 Built-in layered layout (`visualization/layered_layout.py`)
A Sugiyama-style pipeline: (1) cycle removal with the Eades-Lin-Smyth ordering heuristic, reversing the fewest feasible edges (for a graph with $m$ mutual pairs at least $m$ edges must point upward); (2) longest-path layering, then tightening so sources sit directly above their targets; (3) one dummy node per crossed layer for long edges; (4) barycentre crossing reduction (best of 24 sweeps); (5) iterative neighbour-averaging x-coordinates with minimum node separations; edges are routed through the dummy nodes and drawn as rounded polylines.

### 13.4 Payload fields and archived runs
Simulation payload: `svg_hierarchical`, `svg_hierarchical_grid`, `svg_spring`; inference payload: `svgs_dot[alg]`, `svgs_dot_grid[alg]`, `svgs_neato[alg]`, `comparison_graphs[alg]`; benchmark payload: `svg_true_dot`, `svg_true_dot_grid`, `svg_true_neato`, `true_graph` and the same per-algorithm maps. When an archived run is loaded, `_ensure_graph_views` adds the Grid versions it lacks and redraws SVGs that came from the first built-in renderer (no arrow heads); SVGs drawn by Graphviz are kept as stored.

---

## 14. Web Server Behaviour

- **Binding / CLI:** `python -m grn_sim.web.server [PORT] [--port N] [--host H]`, falling back to `$PORT` / `$HOST`, default `127.0.0.1:8080`. The Docker image sets `HOST=0.0.0.0`.
- **Cross-origin protection:** requests carrying a foreign `Origin` header are answered with 403 and have no effect; CORS headers are only sent for the single origin in `GRN_CORS_ORIGIN`.
- **Shared state:** simulation, inference and archive loading hold one re-entrant lock; the history list is replaced atomically; the VCC computation is serialised.
- **Request validation:** malformed POST bodies or `Content-Length` return 400; `cells_per_condition` for `.h5ad` export must be in $[1, 2000]$ (400 otherwise).
- **Client disconnects** (navigating away, aborted fetches) are ignored silently instead of logging tracebacks.
- **Benchmark exports** (`data/benchmarks/raw/*`) are written through a temporary file and renamed, so a file this user cannot overwrite (for example one exported earlier by the Docker container as root) no longer makes a dataset fail to load; if the folder itself is read-only the files go to a temp folder. Dataset load failures leave the previous dataset on screen and show an error banner.
- **Database:** `RGN_DB_PATH` selects the SQLite archive; an unwritable location falls back to a temp file.
