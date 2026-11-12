# Virtual Cell Challenge 2026 scoring in the Virtual Cell tab

The Virtual Cell tab scores the simulations it runs the way https://virtualcellchallenge.org scores a submission: with the six
metrics of the leaderboard, each placed between the context's mean response (0) and a replicate experiment (1). The scorer
is a port of Arc's `cell-eval2` (`vcc2026` metric profile, specification `docs/vcc2026_metrics/vcc2026-metrics.md` of that
repository, 2026-08-19) to the simulator's data; `cell-eval2` needs Python 3.11 and cannot be installed next to this package.

| leaderboard column | `cell-eval2` member | what it measures |
|---|---|---|
| PDS | `pds_cosine` | is the predicted effect closer to its own real effect than to any other perturbation's (mid-rank of the cosine distance, every panel target gene removed) |
| MSE | `expr_mse_unbiased_capped_norm` | sampling-corrected squared distance between predicted and real profiles, over the real effect's size (one ratio over the panel) |
| JAC | `de_wilcoxon_sig_jaccard` | overlap of the significant gene sets |
| NMAE | `de_wilcoxon_lfc_nmae` | normalised mean absolute error of the log2 fold changes over the reference's significant genes (gate of at least 10) |
| FID | `de_wilcoxon_direction_fidelity_yield_raw` | share of the predicted calls moving in the right direction, scaled by yield |
| REACH | `de_wilcoxon_direction_reach_raw` | depth to which the prediction's own ranking of the reference's significant genes stays 90 % pure |

## What is ported, step by step (`grn_sim.evaluation.vcc2026`)

* **Profiles.** Counts summed over a group's cells, normalised to `TS = 5e4`, log1p. Effects are profile minus control profile.
* **Sampling noise.** The delete-one-cell jackknife `C` of a profile, per gene.
* **Differential expression.** Counts per million per cell; two-sided Wilcoxon rank-sum test per gene, normal approximation with
  tie correction (bit-identical to scanpy's `rank_genes_groups(method="wilcoxon", tie_correct=True)`,
  `tests/test_vcc2026_scorer.py`); genes tested only when the control's mean CPM exceeds 5; Benjamini-Hochberg within each
  perturbation (identical to scipy's); log2 fold change of the arithmetic CPM means with `epsilon = 1e-9`.
* **The six members** with the perturbed gene's own row excluded, the conventions of the specification (mid-rank ties, the
  0/0 fidelity dropped, an empty Jaccard union = 1, a missing predicted fold change = 0, the NMAE gate of 10 genes, the reach
  prefix "deepest, not first").
* **Anchors.** Baseline = the context's mean per-perturbation count profile (each gene's own perturbation left out) emitted as
  resampled, rescaled control cells (`cell-eval2`'s `emit="dispersed"`); replicate = the reference scored against itself over
  five disjoint half-splits (seeds from `SeedSequence(0)`), NMAE with its full-gate estimator.
* **Scale.** `s = (u - b) / (r - b)`; MSE clamped to [0, 1], NMAE floored at -6, the other four unclamped; overall = the plain
  mean. Predicted effects and the prediction's DE are measured against the *reference's* control cells (`control_source=real`).

One deliberate difference from the official scorer: the perturbed gene's own share of the three sampling corrections is
removed along with its share of the distances. `cell-eval2` keeps the corrections whole (its cached moments hold one scalar
per group), an approximation worth 0.01 % of the denominator on 18,533 genes but enough, on a 60-gene panel, to push a
replicate's error below zero. The specification names the gene-excluded form as the exact one.

## What the tab scores

**Cross-atlas comparison (Table 2b).** The held-out perturbations of the target context become a virtual Perturb-seq (400
Poisson-sampled cells each around the simulated steady states, plus controls). Every row of Table 2 (linear Jacobian, Hill
refit, cross-context transfer, AtlasShift) predicts their mean response; the prediction is emitted as control cells rescaled
gene by gene and rounded to counts, then scored as a submission. The simulator's noise-free means are scored as an oracle row.

**Screen simulations ("Simulate once").** A quarter of the perturbed genes is held out. Every algorithm of the screen benchmark
is fitted on the other perturbations' effects and predicts the held-out knockdowns through *structure + linear refit*: its two
best regulators per target, weights refit by least squares in log expression on the seen knockdowns (both orientations of the
score matrix tried, the one that reproduces the seen data better kept), the response to a knockdown as the model's fixed
point with the median on-target effect of the seen knockdowns as the dose. Reference rows: the true network through the same
refit, the simulator's noise-free means, a target-gene-only prior, the random and hubness-only controls. The scaled scores are
also written into the per-algorithm rows (`vcc_overall`, `vcc_<member>`). Two options: **cells per perturbation** (the
benchmark's 40, or up to the challenge's 400, which gives the DE members their power; runs with other counts than 40 are shown
but not added to the stored statistics) and **bystander genes** (unregulated genes added to the scored panel so that a
knockdown is a small share of the library, as on a real panel; off by default, they cost roughly a tenfold scoring time).

## Reading the tables on simulated panels

The challenge's scale is designed for 300 perturbations and 18,533 genes. On a panel of a few perturbations and a few dozen
genes the direction and set members are noisy and the two anchors can sit within noise of each other. The scorer therefore
reports, per panel, which members have a resolved scale: a member is left unscaled (raw value shown, excluded from the
overall, the overall marked "k of 6") when the replicate is not separated from the baseline by more than twice the replicate's
spread over its five splits, or, for MSE, when the reference's corrected off-target effect does not exceed the sampling
corrections it subtracts. Compare rows within one table only; the anchor rows show where 0 and 1 sit.

Code: `src/grn_sim/evaluation/vcc2026.py` (scorer), `src/grn_sim/evaluation/vcc_prediction.py` (cells, predictors, scoring of
a prediction set), `grn_sim.benchmarks.vcc_benchmark._vcc_scores_for_held_out` and `grn_sim.evaluation.vcc_benchmark.run_simulation(return_vcc=True)`.
Tests: `tests/test_vcc2026_scorer.py`, `tests/test_vcc2026_integration.py`.
