# Simulation options

Every way GRN-Sim can generate perturbation data, and which option belongs to which part of the studio. For the
mathematics behind the kinetics see [`SPECIFICATION.md`](../SPECIFICATION.md) §1-4; for the Perturb-seq stages see
[`PERTURBSEQ.md`](PERTURBSEQ.md); for the visual overview see [`slides/index.html`](slides/index.html).

## The two simulators at a glance

| | **Studio** (tab 1, `/api/simulate`, job queue) | **Virtual Cell Challenge** (tab 4, `/api/benchmark/vcc_atlasshift`) |
|---|---|---|
| Purpose | One network you configure, perturb, then reverse-engineer | A fixed multi-context study that compares many algorithms and a transfer baseline |
| Entry point | `server._execute_simulation` -> `run_simulation_with_engine` | `benchmarks.vcc_benchmark.run_vcc_comparative_benchmark` |
| Network | Any topology, 6-2000 genes, optional calibrated preset | Always power-law, ratio 0.55, 4-32 genes |
| Contexts | One | Three source atlases + one rewired target |
| Perturbation | Regulator input (+`perturbation_factor`) | `input` (+1) or `knockdown` (-80% production) |
| Measurement noise | `noise` (default 0.05) | fixed 0.01 |
| Held-out evaluation | none | 75% of perturbations seen / 25% held out |

Both use the same core: Hill kinetics in `simulator/kinetics.py`, the ODE steady-state solver, and
`PerturbationSimulator`, which perturbs one gene per experiment and returns `X` (relative responses) and `U` (the
perturbation matrix).

## Part 1: how a perturbation propagates

The network is a set of coupled ODEs. For gene *i*:

```
dx_i/dt = s_i(x) - b_i x_i
s_i(x)  = v_i * prod_{j in Inh(i)} [ K_I^m / (I_j^m + K_I^m) ] * prod_{k in Act(i)} [ 1 + A_k^m / (A_k^m + K_A^m) ] + u_i
```

`PerturbationSimulator.run` first solves the unperturbed steady state `x_ss`. Then, for every gene *j* in turn, it
applies the perturbation to gene *j* only, integrates to the new steady state and records
`X[:, j] = x_perturbed / x_ss - 1`. Gene *j*'s change alters the production of the genes it regulates, those change
their targets, and so on until the system settles. Indirect effects, feedback and attenuation along paths are
therefore part of every column of `X`. Nothing is drawn from a correlation model.

### Perturbation modes

| Mode | What is applied | `U[j, j]` | Used by |
|---|---|---|---|
| `input` (default) | `perturbation_factor` is added to gene *j*'s concentration *as seen by its regulators* (the thesis "tagged mRNA" device: gene *j*'s own dynamics are untouched, so a regression does not read the perturbation as self-regulation) | `+perturbation_factor` | Studio, VCC "Activation" |
| `knockdown` | gene *j*'s production is scaled by `1 - e` (CRISPRi; default e = 0.8; `efficiency_spread` adds per-gene scatter) | `-e` | VCC "CRISPRi", Perturb-seq |
| `knockout` | production set to 0 | `-1` | Perturb-seq (`--mode knockout`) |

Knockdown data are dominated by the targeted gene's own drop. Score inferred networks on off-target genes
(the VCC tab reports `offtarget_r` for this reason).

### Steady state and diagnostics

The solver stops when the squared displacement per step drops below `efnorm <= 1e-4`. A network that never settles
is flagged **oscillating**; the studio classifies each run as *responsive*, *flat* (largest change under 5%) or
*oscillating*, and Perturb-seq refuses oscillating networks.

## Part 2: Studio options (tab 1)

All options are request fields of `/api/simulate` and of the job queue's `params`. Ranges are enforced by
`_validate_sim_params`; the job queue also rejects unknown names.

### Network

| Option | Field | Range / values | Default | Effect |
|---|---|---|---|---|
| Genes | `num_genes` | 6-2000 | 12 | Network size. Above 60 genes trajectories and profiles load on demand and the graph panels show at most 25 modules. |
| Topology | `topology` | `random`, `power_law`, `hierarchical`, `modular` | `power_law` | See below |
| Mean connectivity | `k_av` | 0.1-10 | 1.8 | Mean number of edges per gene; target edge count is `ceil(k_av * n)` |
| Activator ratio | `act_ratio` | 0-1 | 0.5 | Probability that an edge activates (otherwise it inhibits) |
| Subdivisions | `subdiv` | integer | 2 | `hierarchical`: number of cloned blocks (must divide *n*, else 2 or 1 is used). `modular`: number of modules (default: one per 10 genes, 2-8). |
| Calibration preset | `calibration_preset` | `thesis`, `replogle_like` | none | Replaces the topology controls with a Perturb-seq calibrated network (see Part 4) |
| Seed | `seed` | 0 to 2^32-1 | 42 | Network, kinetic parameters and noise |

**Topologies** (`core/topology.py`):

| Topology | Degree distribution | How it is built |
|---|---|---|
| `random` | Poisson (Erdős-Rényi) | Every gene gets one random regulator, then random extra edges until the target is reached |
| `power_law` | Heavy-tailed (scale-free) | Preferential attachment: a node receives a link with probability proportional to its degree |
| `hierarchical` | Scale-free with clustering | A scale-free block is cloned `subdiv` times, then cross-block edges are added by preferential attachment |
| `modular` | Narrow in-degree, heavy-tailed out-degree | Each gene gets `1 + Poisson(k_av - 1)` regulators; a regulator comes from the gene's own module with probability `p_within` (0.85), chosen with probability proportional to `out_degree + 0.5` |

The thesis samplers (`power_law`, `hierarchical`) stall above about 60 genes; the missing edges are then drawn by the
same degree-preferential rule (`_top_up_edges`), so smaller networks are unchanged.

### Kinetics engine (`kinetics_engine`)

| Engine | Dynamics | Steady state | Typical use |
|---|---|---|---|
| `hill_nonlinear` (default) | Hill ODE above | numerical integration | the reference model the thesis algorithms are built for |
| `cellbox_neural_ode` | `dx/dt = tanh(sum_j W_ij x_j + b_i + u_i) - alpha_i x_i`, `W = gamma * T` | integration | data that matches the CellBox model class |
| `linear_jacobian` | `dx/dt = (A - B) x + v + u` | closed form `x = (B - A)^-1 (v + u)`; adjacency scaled to spectral radius < 1 | the linear model that A X = -U assumes exactly |
| `perturb_seq_counts` | Hill ODE, then 400 cells per condition of Gamma-Poisson counts with log-normal depth | means of the cells | count-like data, `.h5ad` style |

### Dynamics preset (`dynamics_preset`)

| Preset | What it sets |
|---|---|
| `responsive` (default) | Nothing: the clear steady-state shifts the algorithms need |
| `flat` | `perturbation_factor` 0.005 and `noise` 0.001 unless you give them: a perturbation too small to see |
| `oscillating` | `act_ratio` 0.1, `k_av` at least 2.5, plus an odd-length negative-feedback ring (5 genes, or 3) with Hill coefficient 4.5, K_I 1.2, v = 8, b = 1 and `perturbation_factor` 0.5 so the network cycles; the run is labelled non-steady |

### Noise and perturbation size

| Option | Field | Range | Default | Effect |
|---|---|---|---|---|
| Perturbation factor | `perturbation_factor` | > 0 | 2.0 | Size of the regulator-input perturbation |
| Measurement noise | `noise` | 0-1 | 0.05 | `x * (1 + sqrt(delta) * N(0,1) * (U(0,1) - 0.5))` applied to each observed steady state (Eq. 2.5) |
| Biological variability | `bio_var` | 0-1 | 0.02 | Stored and shown with the run. **The Studio's simulate path does not pass it to the simulator, so it currently has no effect.** `PerturbationSimulator(biological_variability=...)` implements it (kinetic constants perturbed with the noise model above) and can be used from Python. |

### Real-data modes (not simulations, but they use the same pipeline)

`dataset_id` / `mode: benchmark_*` loads a measured dataset (E. coli SOS, DREAM4, Norman, Replogle) and offers two
`loading_mode`s:

- `wrangled_direct`: the algorithms run directly on the measured `X` and `U`.
- `biophysical_fit`: Hill parameters (v, b, K_A, K_I, m = 2) are estimated from the measurements and the network is
  re-simulated.

See [`REAL_WORLD_BENCHMARK_PROVENANCE_AND_USAGE.md`](REAL_WORLD_BENCHMARK_PROVENANCE_AND_USAGE.md).

## Part 3: Virtual Cell Challenge options (tab 4)

| Option | Field | Values | Effect |
|---|---|---|---|
| Cell context | `rewire_fraction` | 0.05 / 0.15 / 0.30 / 0.45 (API: 0-1) | Fraction of the target network's edges moved to another target. A moved edge carries the kinetic parameters of the edge it replaces and never lands on an existing edge, so the edge count is preserved. |
| Genes | `num_genes` | 4-32 (menu: 8, 12, 16, 24) | The analysis is O(n^3), hence the cap |
| k_av | `k_av` | up to 5 (menu: 1.5, 1.8, 2.2) | Mean connectivity |
| Perturbation | `perturbation` | `input`, `knockdown` | See perturbation modes |
| Seed | `seed` | default 2026 | Source atlases use `seed`, `seed+10`, `seed+20`; the target uses `seed+999` |
| Re-simulate | `refresh` | | Bypasses the cache (results are cached per parameter set) |
| VCC bystander genes | `bystanders` | 0 (default) or up to 5000 (menu: none, 1,000) | Unregulated genes added to the panel that Table 2b scores with the Virtual Cell Challenge metrics, so a knockdown is a small share of the library as on a real panel; see [`VCC2026_SCORING.md`](VCC2026_SCORING.md) |

What the benchmark builds, in order:

1. **Three source atlases** (simulated source contexts A, B and C; no cell-line data): the same power-law network with basal
   rates modulated by `1 + 0.15 sin(idx + 1 + gene index)`.
2. **One target context**: the same network after rewiring.
3. **Edge recovery**: every algorithm is fitted on the target's `X`, `U` and scored against the true network.
4. **In- versus out-of-distribution response prediction**: 75% of perturbations seen, the last 25% (at least 2)
   held out; every model is refit without the held-out experiments. Predictors per algorithm: the linear Jacobian
   `R = sign * pinv(A_est)` (exactly 0 for held-out genes) and the structure -> Hill refit (top-2 regulators per
   target, see [`HOLDOUT_RESEARCH.md`](HOLDOUT_RESEARCH.md)). References: mean-profile null, true-structure oracle,
   target-gene prior (knockdown only), and the cross-context transfer predictors
   ([`CROSS_CONTEXT_TRANSFER.md`](CROSS_CONTEXT_TRANSFER.md)).
   4b. **Virtual Cell Challenge scale** (Table 2b): the held-out perturbations as 400 virtual cells each, every predictor
   above emitted as cells and scored with the six leaderboard metrics against the context-mean baseline and a replicate
   anchor measured on the same cells ([`VCC2026_SCORING.md`](VCC2026_SCORING.md)).
5. **Epistasis screen**: every pair is simulated on the true network without measurement noise;
   `eps_ij = ||L_ij - (L_i + L_j)||` with `L = log2(x_perturbed / x_basal)`. Above 0.45 is "Synergistic / Synthetic
   Lethal", above 0.25 intermediate, the rest additive.
6. **Virtual single cells**: about 140 Poisson-count cells (controls plus the first few conditions) and a 2D PCA.
7. **Rewiring sensitivity**: the model fitted on the target is applied to contexts rewired by 0, 10, 20, 35 and 50%.

## Part 4: Single-cell Perturb-seq options

The Perturb-seq panel of tab 4, `grn-sim perturbseq`, and `grn_sim.perturbseq.pipeline.run_screen`. The stages are in
[`PERTURBSEQ.md`](PERTURBSEQ.md); this table lists the options.

### Screen design

| Option | CLI / API | Values | Default | Effect |
|---|---|---|---|---|
| Perturbation type | `--mode` / `perturbation` | `knockdown`, `knockout`, `input` | `knockdown` | See perturbation modes |
| Double perturbations | `--pairs` / `pairs` | integer, `"all"`, or explicit pairs | 0 | Pair steady states at nominal doses; epistasis against the log-additive null |
| Pair pool | `pair_pool` | `effective`, `all` | `effective` | Draw pairs among the informative singles or all genes |
| Perturbed genes | `--perturbed` / `perturbed` | 0 = every gene | 0 | Partial screen: a few perturbed genes, all genes measured |
| Selection | `--selection` / `selection` | `effective`, `random` | `effective` | Which genes are perturbed in a partial screen |
| Network | `--preset` / `preset` | `current`, `replogle_like`, `thesis` | studio: `current`; CLI: `replogle_like` (200 genes) | Use the open network or a preset of the same size |
| Cells per condition | `--cells` | integer | 100 (studio export: 400) | Cells per perturbation |
| Control cells | `--controls` | 0-max | same as cells per condition | Non-targeting cells |

### Cell model (`CellConfig`)

| Option | Default | Effect |
|---|---|---|
| `noise_model` | `lna` | `lna`: linear-noise approximation around each condition's steady state (fast, accurate for well-expressed genes). `sde`: chemical-Langevin integration of every cell with Euler-Maruyama (`sde_dt` 0.02, `sde_time` 15; exact, much slower; differs mainly for low-copy genes and steep Hill curves). The CLI also accepts `legacy`. |
| `guides_per_gene`, `n_control_guides` | 2, 4 | Guides per target and distinct non-targeting guides |
| `guide_efficiency_mean` | 0.8 (preset `replogle_like`: 0.92) | Mean per-guide knockdown; Beta-distributed with `guide_efficiency_concentration` 15 and per-cell `cell_efficiency_concentration` 12 |
| `escape_rate` | 0.1 | Perturbed cells without an effective perturbation |
| `knockout_biallelic_rate` | 0.8 | Knockout: share of non-escaped cells with full loss (rest: half dose) |
| `extrinsic_gene_sd`, `extrinsic_shared_sd` | 0.15, 0.1 | Log-sd of per-gene and cell-wide production rate scatter |
| `omega` | none | System size for intrinsic noise (none = no intrinsic noise) |

### Measurement model (`MeasurementConfig`)

| Option | Default | Effect |
|---|---|---|
| `mean_library_depth`, `depth_sd` | 20000, 0.35 | UMIs per droplet and log-sd between droplets |
| `dispersion` | 0.05 | Negative-binomial overdispersion (0 = Poisson) |
| `gene_capture_sd` | 0.3 | Fixed per-gene capture efficiency |
| `n_batches`, `batch_gene_sd`, `batch_depth_sd` | 4, 0.1, 0.15 | Batch effects |
| `ambient_fraction`, `ambient_concentration` | 0.03, 50 | RNA from the soup |
| `doublet_rate` | 0.04 | Droplets holding two cells |
| `guide_umi_mean`, `guide_dispersion`, `guide_dropout`, `ambient_guide_umis`, `assign_min_umis`, `assign_min_fraction` | 30, 0.5, 0.05, 0.5, 3, 0.2 | Guide capture and the guide-calling rule |

"Apply measurement artefacts" in the studio (`measurement=0` to turn off) switches the whole measurement stage on or
off; the truth is stored next to every observation either way.

### Calibrated network presets (`perturbseq/calibration.py`)

| Preset | Topology | k_av | Activating share | Kinetics |
|---|---|---|---|---|
| `thesis` | power-law | 1.8 | 0.5 | thesis kinetics (about a quarter of edges sit on a flat part of their Hill curve) |
| `replogle_like` | modular, 5-10 modules, `p_within` 0.85 | 2.5 | 0.8 | anchored: every half-saturation constant tied to its regulator's steady-state level (`K = x_ss * f`, `f ~ LogNormal(0, 0.5)`, Hill coefficient 1-3), so knockdowns propagate; silent edges fall to 0-1% for 200-500 genes |

Against the seven genome-scale realism targets `replogle_like` meets six at 500 genes; see
[`PERTURBSEQ.md`](PERTURBSEQ.md) for the numbers and the one it misses.

## Which option for which question?

| You want to... | Use |
|---|---|
| Test whether an algorithm can recover a known network | Studio, `responsive`, `hill_nonlinear`, a modest `noise` |
| Test robustness to measurement noise | Raise `noise` (try 0.01, 0.05, 0.10) |
| Check an algorithm that assumes linearity | `linear_jacobian` (the model A X = -U is exact there) |
| See what failure looks like | `flat` or `oscillating` presets |
| Compare methods on unseen perturbations or a rewired context | Virtual Cell Challenge tab |
| Get CRISPRi-like single-cell data with ground truth | Perturb-seq panel with `replogle_like` and `knockdown` |
| Study double perturbations / synthetic lethality | `pairs` in Perturb-seq, or the VCC epistasis view |
| Many networks or algorithms without waiting | Job queue (tab 6): same `params` as `/api/simulate` |
