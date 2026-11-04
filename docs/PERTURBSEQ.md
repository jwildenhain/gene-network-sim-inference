# Perturb-seq simulation (`grn_sim.perturbseq`)

GRN-Sim can turn a gene regulatory network into a realistic pooled CRISPR screen with a single-cell readout
(Perturb-seq / CROP-seq style), together with the ground truth a benchmark needs. The pipeline has four stages:

| Stage | Module | What it adds |
|---|---|---|
| 1. Conditions | `conditions.py` | Steady states for every single knockdown / knockout (and optional double perturbations); ground truth: total log2 effects, edge elasticities, linear response, a label per perturbation (`no_targets`, `no_downstream_effect`, `effective`), epistasis of pairs against the log-additive null. |
| 2. Cells | `cells.py` | Guides with their own efficiency, per-cell knockdown doses, escapers, knockout alleles; cell-to-cell noise (per-gene and shared production noise, optional intrinsic noise) propagated through the network, with a fast linear-noise model (`lna`) or exact chemical-Langevin integration (`sde`). |
| 3. Measurement | `measurement.py` | Doublets, ambient RNA, gene capture bias, batches, library size, Poisson / NB UMI sampling, guide UMIs and guide calling; the truth is stored next to every observation. |
| 4. Realism | `calibration.py`, `realism.py` | Anchored Hill kinetics and network presets; statistics of the simulated screen against genome-scale Perturb-seq targets. |

`pipeline.py` runs all stages in one call; the CLI, the studio and the tests use it.

## Python

```python
from grn_sim.perturbseq.pipeline import build_network, run_screen

net = build_network("replogle_like", 500, seed=1)          # or any GeneNetwork
run = run_screen(net, mode="knockdown", pairs=50, cells_per_condition=100, cell_model="lna", seed=1)
run.adata.write_h5ad("screen.h5ad")                         # counts, obs truth, uns["ground_truth"]
report = run.realism()                                      # statistics + rows vs TARGETS + score
```

Lower-level building blocks: `simulate_conditions`, `simulate_cells` (`CellConfig`), `measure_population`
(`MeasurementConfig`), `population_to_anndata`, `truth_statistics`, `observed_statistics`, `realism_report`,
`calibrate_kinetics`, `PRESETS`.

## Command line

```bash
grn-sim perturbseq --genes 500 --preset replogle_like --pairs 50 --cells 100 --out screen.h5ad --report realism.json
```

Options: `--mode knockdown|knockout|input`, `--noise lna|sde|legacy`, `--controls N`, `--no-measurement`,
`--no-report`, `--seed`.

## Studio

Tab 4 (Virtual Cell Challenge) has a *Single-Cell Perturb-seq Screen* panel: perturbation type, cell-noise model,
network (current network or a preset network of the same size), number of double perturbations, cells per
perturbation and whether to apply measurement artefacts. **Download .h5ad** exports the screen; **Realism report**
shows the statistics against the targets. The same screen is cached, so a report followed by a download simulates it
once.

API:
- `GET /api/export/perturb_seq_h5ad?perturbation=&noise=&preset=&pairs=&cells_per_condition=&measurement=&controls=&run_id=`
- `GET /api/perturbseq/realism?` (same parameters) returns `{score, report, statistics, settings, seconds}`.
- Networks above 60 genes serve trajectories and gene profiles on demand: `GET /api/trajectory?perturbation=`,
  `GET /api/gene_profile?gene=`.

## Realism targets and calibration

| Target | Range | Source |
|---|---|---|
| On-target knockdown (CRISPRi) | 0.80-0.95 | Replogle 2022 |
| Perturbations with a transcriptional phenotype (other genes) | 0.25-0.45 | Replogle 2022, Aguirre 2025 |
| Affected ordered gene pairs | 0.015-0.05 | Aguirre 2025 |
| Of those, bidirectional | 0.01-0.06 | Aguirre 2025 |
| Mediated (indirect) share of effects | 0.90-0.995 | Aguirre 2025, inspre |
| Share of effects from the top 5% perturbations | 0.25-0.70 | Nadig 2025 |
| Non-targeting false-positive rate | 0-0.05 | Replogle 2022 |

The thesis kinetics leave about a quarter of all edges on a flat part of their Hill curve ("silent edges").
`calibrate_kinetics` anchors every half-saturation constant to its regulator's steady-state level (damped, adaptive
fixed-point iteration), which brings silent edges to 0-1% for 200-500 genes.

The `replogle_like` preset (modular preferential attachment, k_av 2.5, 80% activating edges, anchored kinetics,
guide efficiency 0.92) was tuned with `scripts/calibrate_perturbseq.py` at 500 genes, where it meets 6 of the 7
targets. It misses the mediated share (0.79 vs >= 0.90): denser networks reach it but then exceed the pair density and
bidirectional share. Genome-scale screens combine low density with ~98.5% mediated effects because they cover
thousands of genes; below that size the two cannot both be met. Statistics of networks much smaller than 500 genes
are not comparable with the targets.

## Partial screens (a few perturbed genes, all genes measured)

A genome-scale screen perturbs a few hundred genes and measures thousands. `ScreenDesign` (`perturbseq/design.py`) separates
*which genes are perturbed* from how the screen is generated:

```python
run = run_screen(net, n_perturbed=30)                              # 30 effective genes (default selection)
run = run_screen(net, n_perturbed=30, perturbed_selection="random")
run = run_screen(net, genes=[3, 17, 42])                           # explicit list
```
CLI: `grn-sim perturbseq --genes 500 --perturbed 30 --selection effective`. API / studio: `?perturbed=30&selection=effective`
(*Perturbed genes* and *Selection* in the Perturb-seq panel; 0 = every gene).

- **Selection.** `effective` (default) draws the genes at random among those whose knockdown moves other genes in the simulated
  network (a noise-free single-knockdown screen of the whole network gives the `effective` label; one extra pass over all genes).
  If the network has fewer effective genes than requested, all of them are used and the shortfall is reported
  (`selection_info["shortfall"]`, the CLI and the studio say so). `random` draws among all genes.
- **Statistics category.** A partial screen has its own statistics (`run.realism()["design"]["categories"]`, shown below the
  realism table in the studio): `design` (perturbed genes, effective share, shortfall), `identifiability` (how many true edges have
  a perturbed regulator at all: P->P, P->Q, from unperturbed genes), `coverage` (unperturbed genes that respond, perturbations per
  responding gene), `blocks` (effect density and mediated share for P->P and P->Q). With `effective` selection the genome-scale
  statistics that the selection shifts (`fraction_with_phenotype`, `affected_pair_density`, `bidirectional_share`,
  `top5pct_perturbation_effect_share`) are still reported but marked `comparable = False` and left out of the score.
- **Output.** All genes are in `X`; `adata.uns["screen_design"]` records the design and the perturbed genes. 500 genes with 30
  effective perturbed genes: about 12% of the true edges have a perturbed regulator and can be identified from interventions.
- **Algorithms.** `registry.describe()` lists `partial: native | adapter` per algorithm and `registry.build_partial(key)` wraps the
  ones that need `PartialAdapter`; scoring per block is `grn_sim.evaluation.partial_screen`. Measured results:
  `docs/partial_screen_benchmark_500genes.md`, `docs/INFERENCE_GAP_ANALYSIS.md`.

## Notes and limits

- `lna` is the default cell model; it is accurate for well-expressed genes. `sde` is exact but much slower, and
  differs from `lna` mainly for genes with very few molecules and on steep Hill curves (noise shifts the mean).
- The phenotype test is a two-sample Hotelling T^2 on principal components with the targeted gene excluded
  (Benjamini-Hochberg over perturbations). An energy-distance permutation test is available
  (`observed_statistics(..., phenotype_test="energy")`), but its p-value floor of 1/(n_perm+1) misses strong
  phenotypes unless many permutations are run.
- Double perturbations use the pair steady state at nominal doses for cells where both guides act.
- Score inferred networks on off-target genes: in knockdown data the target gene dominates every profile.
