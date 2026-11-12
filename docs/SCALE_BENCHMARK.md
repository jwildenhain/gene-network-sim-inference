# The 500 / 1000 / 2000 gene benchmark: what the data are, and why scores fall with size

## Large networks use the same mechanistic simulation

There is no separate generator for big networks. Every size, 2000 genes included, is simulated by the mechanistic model of the
thesis (`hill_nonlinear`: Hill kinetics with saturating regulation and degradation, Wildenhain and Crampin), solved to a steady
state for every condition (`simulator/steady_state.py`). The statistical generators (`grn_sim.statgen`) belong to the Virtual
Cell tab and are not used by the studio, the job queue or this benchmark.

What changes with size is the parameterisation and the experiment, not the simulator:

| | thesis-scale runs (12-16 genes in the archive) | scale run (500-2000 genes) |
|---|---|---|
| kinetics | default random parameters | `replogle_like` calibrated kinetics (`perturbseq/calibration.py`): dissociation constants anchored to the steady state, because the default parameters do not reach one at this size (a 2000-gene network with the defaults oscillated) |
| topology | scale-free / random / hierarchical | `modular` (5-10 modules), about 2.5 edges per gene before pruning (1245 true edges at 500 genes, 2536 at 1000) |
| experiment | one perturbation per gene (N conditions) | the same: N genes, N single-gene perturbations of +2 on the regulator input, 5% measurement noise |
| data model | regulator-input perturbation (the response depends on X + U) | the same |
| trajectories | stored | served on demand above 60 genes (display only, does not change the data) |

## Scores at thesis scale and at scale (stored runs, F1 / AUPR)

| algorithm | 12-16 genes (mean of 191-192 runs) | 500 genes | 1000 genes |
|---|---|---|---|
| S2G | 0.37 / 0.30 | 0.11 / 0.10 | 0.08 / 0.11 |
| G2S | 0.32 / 0.25 | 0.14 / 0.05 | 0.16 / 0.07 |
| Voting ensemble | 0.44 / 0.38 | 0.22 / 0.09 | 0.23 / 0.10 |
| NIR | 0.51 / 0.52 | 0.25 / 0.06 | 0.18 / 0.03 |
| FAST | 0.27 / 0.31 | 0.14 / 0.02 | 0.02 / 0.00 |
| RidgeCV | 0.30 / 0.33 | 0.27 / 0.05 | 0.25 / 0.04 |
| AtlasShift | 0.52 / 0.51 | 0.27 / 0.14 | 0.30 / 0.17 |
| CausalGRN | 0.34 / 0.32 | 0.28 / 0.12 | 0.27 / 0.15 |
| SFINN | 0.54 / 0.52 | 0.29 / 0.13 | 0.28 / 0.18 |
| NOTEARS | 0.28 / 0.37 | 0.22 / 0.05 | 0.20 / 0.09 |

The 500 and 1000 gene columns are one network each (seed 7, results before the NIR / ridge / inspre / NOTEARS fixes where
those algorithms were not re-run); the small-network column is the archive mean. The two columns are different networks with
different parameterisations, so the fall in score is a property of the setting, not of one change.

## Why scores fall (what the data support, and what is a hypothesis)

Supported by measurements in this repository:
- With one condition per gene the data are saturated: as many unknown regulators per gene as experiments. Methods that start
  from the full set of columns (G2S) or add columns by residual (S2G) pick 2-13 per gene on these data, whatever the true
  degree (about 2.5).
- The effects are mostly indirect. The calibrated preset reaches a mediated share of 0.79 at 500 genes (0.85 at 1000), against
  about 0.985 in genome-scale screens (`docs/PERTURBSEQ.md`): a perturbation moves many genes through the network, and the
  direct edges are a small part of the signal that methods must separate out.
- The perturbed gene itself barely responds in the regulator-input mode (median own response -5e-5 at 500 genes), which is
  what made inspre (built for knockdown data) return an empty graph until it was given the response of X + U.

Hypotheses, not tested here (no ablation was run): that the fall is mainly the number of candidate regulators (500 against 12),
and that the noise level matters more at scale. Disentangling them needs the same preset at 12-60 genes, which is not in the
archive.

## Edge counts

Algorithms that need a number of edges per gene (NIR, FAST, RidgeCV, GENIE3, CellBox, CPA-Lite, ARACNE, NOTEARS, CausalGRN,
DMD-GRN, DeepRIG, SFINN, NetDeconv) receive `k`, the true mean in-degree rounded up (`k_est`), when the truth is known. S2G and
G2S were designed to choose their own size by MDL / AIC; on these data they do not recover it (S2G predicts 13-22 edges per gene,
G2S exactly 2), so they are also given `k` as a cap on Hill data, like the others. This uses information a real screen
does not have. AUPR does not depend on the cap and is the figure to compare when edge counts differ.
