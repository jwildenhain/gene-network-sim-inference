# To do

State as of 2026-11-12. Source for the algorithm list: `research_notes/Perturb seq design and modelling/`
(`causal_grn_inference.md`, `prediction_models_benchmarks.md`); gap audit in `docs/INFERENCE_GAP_ANALYSIS.md`.

## Where the project stands

- `main` has the Perturb-seq pipeline (phases 1-5), job queue, algorithm registry (19 built-ins incl. `score_vote`),
  paged archive, accessibility pass, CellBox GPU option (PRs #2-#6 merged).
- Branch `feat/inspre-and-interventional-baselines` (pushed, no PR yet, 6 commits ahead of its push):
  inspre, mean-difference / Pearson / GRNBoost-lite / gene-feature baselines, bicycle-lite, DOTEARS-lite, rank consensus,
  partial screens (`ScreenDesign`), per-algorithm docs (`docs/algorithms/`), statistical data generators and the Virtual
  Cell tab benchmark (`grn_sim.statgen`, 1000-simulation corpus, rendered findings), pathway topologies.
- VCC 2026 leaderboard scorer (`evaluation/vcc2026.py`, `vcc_prediction.py`, `docs/VCC2026_SCORING.md`) committed on
  the same branch (2a31295).
- Container `grn-sim-phpiorn` serves :8085 (rebuilt 2026-11-11).
- Gene cap is 2000 and jobs accept `calibration_preset` (e.g. `replogle_like`): the default kinetics oscillate at 2000
  genes, the calibrated preset gives a steady state (generation ~9 min).

## Paper algorithms

Implemented since the first list: inspre, mean-difference, GRNBoost-lite, Pearson, bicycle-lite (no latent variables),
DOTEARS-lite, interventional consensus, gene-feature baseline. Not implemented:

| Priority | Method | Paper | Note |
|---|---|---|---|
| 2 | **LLC** (linear cyclic, latent confounders) | Hyttinen et al. 2012 | Reference linear cyclic SEM; inspre is the modern version. |
| 2 | **bicycle, full** (latent variables, Lyapunov likelihood) | Rohbeck et al., CLeaR 2024 | Only the lite version exists; practical to ~250 genes. |
| 2 | **SCEPTRE** (calibrated ACE) | Barry et al. | `perturbseq/ace.py` uses an NT-calibrated Welch test instead. |
| 3 | **DCD-FG** | Lopez et al., NeurIPS 2022 | Needs torch (optional extra exists). |
| 3 | **DCDI / full DOTEARS** | interventional DAG learners | Lite versions exist. |
| 3 | **GIES / IGSP** | Hauser and Buehlmann 2012; Wang, Squires, Uhler | Slow beyond ~100 genes. |
| 3 | **SparseRC** | CausalBench challenge | Root-cause assumption on a DAG. |
| 4 | **Dictys / RENGE** | Wang 2023; Ishikawa 2023 | Need time-series data, which the archive does not store. |
| skipped | **CLIPR / latent causal diffusions** | arXiv 2601.15341 | No code available (decision 2026-11-04). |

Papers still to read before implementing: LLC, GIES, IGSP, DCDI, NODAGS-Flow, Dictys.

## Perturb-seq proposal (`reports/GRN-Sim Perturb-seq proposal.md`)

- [ ] **Held-out-perturbation benchmark on single-cell screens** (proposal section 3): training mean / additive / linear /
  target-only / random-structure baselines, off-target metrics, stratified by effect size. `evaluation/screen_benchmark.py`
  covers edge recovery (direct and total AUPR); prediction of held-out perturbations is still "not done" in the gap
  analysis. The VCC 2026 scorer (uncommitted) covers part of the metrics; `scripts/holdout_study.py` is still bulk-data only.
- [ ] **2000 genes, all algorithms, G2S in its quick mode.** Not completed: two 2000-gene networks (seed 7, `replogle_like`,
  `iorn-182011`, `iorn-533100`) are in the archive with no algorithm results; the inference job was interrupted by a server
  restart both times (jobs are not resumed). Re-run with `calibration_preset: replogle_like` through the job queue, pinned
  to NUMA nodes 0/2 and niced (see the CPU rule in the notes); expect CellBox (~21 min at 500 genes) and NOTEARS /
  CausalGRN to dominate, and expect some algorithms to be too slow at 2000 genes.
- [ ] Realism targets still missed: mediated share 0.79 at 500 genes (target 0.90), phenotype share 0.20; Virtual Cell tab:
  top-5% share 0.23-0.24, nonlinear phenotype 0.19, mechanistic density 0.10 at 100 genes.

## Housekeeping

- [ ] Open the PR for `feat/inspre-and-interventional-baselines`.
- [ ] Cut a release (everything is under `[Unreleased]`).
- [ ] Real-data check of the Hill refit and transfer predictors on Norman 2019 / Replogle 2022.
- [x] Archive payload at 2000 genes: predictions are stored sparsely above 60 genes (`evaluation/prediction_codec.py`,
  12 MB -> under 1 MB per algorithm). Still large per algorithm: the three SVG graphs (~25 MB at 2000 genes) and the
  comparison graph (~3 MB); one run's simulation payload is 200 MB (`msmt_text`/`pert_text` text copies of the data are
  82 MB of that). Next: render large-network SVGs on demand.
- [ ] Per-algorithm overhead: a trivial algorithm (`by_chance`) costs 8 s at 500 genes, 7 s of it in
  `layered_layout` (pure-Python crossing minimisation, called once per algorithm); 73 s per job at 2000 genes.
- [x] Pool deadlock guard (`inference/_parallel.run_chunks`): workers with no CPU use for 300 s are killed and the
  chunks finished in the parent; workers dump their stacks to `/tmp/grn_sim_worker_<pid>.trace`. Root cause of the
  one observed hang (500-gene G2S, forked worker in futex_wait for an hour) is still unknown: stderr-lock repro failed.
- [x] Job list no longer drops a finished job that queued jobs depend on; jobs store each algorithm as it finishes.
- [ ] Resume jobs interrupted by a restart (they come back as "interrupted"; the last 10 are kept and roll off).
- [ ] 40 archived datasets have no algorithm rows (28 small leftovers, 12 profiling), 10 have 6 of 19, 5 lack only
  `score_vote` (`grn-sim-registry-and-archive-rerun` notes).
