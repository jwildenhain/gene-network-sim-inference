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

Evaluated in `src/grn_sim/benchmarks/vcc_benchmark.py`:

1. **In-Distribution Generalization**:
   When the target gene perturbation was observed across source reference atlases, AtlasShift borrows empirical shifts ($r \approx 0.526$, RMSE $0.228$) outperforming or matching mechanistic GRN ($r \approx 0.35 - 0.55$) by leveraging real empirical cross-cell line co-dependencies.

2. **Out-of-Distribution (OOD) Generalization**:
   When a perturbation target was unseen across all source atlases:
   - **AtlasShift collapses to $r = 0.0$** (zero-prior collapse) because without empirical data, $\Delta \mathbf{y}_{\text{fused}} = \mathbf{0}$.
   - **phpIoRN Mechanistic GRN predicts unseen targets ($r = 0.561$, RMSE $0.079$)** by inverting the reconstructed causal network Jacobian:
     $$\hat{\mathbf{X}} = \mathbf{A}_{\text{est}}^{-1} \mathbf{U}$$
     perturbation forces propagate along physical network cascades to downstream genes.

3. **Context Rewiring Sensitivity**:
   As target cell line rewiring increases ($\rho = 0\% \to 50\%$ edge mutations/enhancer switching):
   - AtlasShift correlation decays rapidly from $r = 0.662 \to 0.276$.
   - Demonstrates that empirical transfer fails when cell lines undergo significant transcriptional network rewiring, necessitating mechanistic causal modeling.


