# To do

State as of 2026-11-12 (evening). Source for the algorithm list: `research_notes/Perturb seq design and modelling/`
(`causal_grn_inference.md`, `prediction_models_benchmarks.md`); gap audit in `docs/INFERENCE_GAP_ANALYSIS.md`.

## 1. Algorithm bugs (top priority: speed, memory leakage, accuracy)

Evidence: the scale run (`replogle_like`, seed 7, 500 / 1000 / 2000 genes; archived runs `iorn-838581`, `iorn-592020`,
`iorn-846813`; about 2,100 true edges at 500 genes). Timings are job times in the 48 GB / 8-core container.

### Memory (leaks and scaling)
- [ ] **cpa_lite**: container memory went from 16 GB to 40.5 GB when its 1000-gene job started (about 24 GB), so about
  100 GB is expected at 2000 genes (over the 48 GB cap: an OOM kill takes the server and every in-memory network).
  Find the allocation (dense n x n tensors per batch?) and bound it; refuse above a limit until fixed. Its 2000-gene job
  is skipped for now.
- [ ] Check every algorithm for memory that is not released after `fit` (hold references to X, U, per-gene models, torch
  tensors): run each at 500 genes in a loop and watch resident memory; the container sat at 13.9 -> 16.9 GB across the
  500/1000-gene jobs before `cpa_lite`.
- [ ] A per-algorithm memory estimate that skips or warns before a run, so no algorithm can take the server down.
- [x] The in-memory run store is bounded by bytes (`GRN_SIM_RUN_STORE_GB`, default 8). Check whether anything else keeps
  per-run copies (`SIMULATION_HISTORY`, `CURRENT_STATE`, job summaries).

### Speed
- [ ] **cpa_lite**: 31 s at 500 genes, over 80 minutes at 1000 (worse than cubic).
- [ ] **cellbox**: about 21 min at 500 genes (GPU option exists, `GRN_SIM_DEVICE`); hours expected at 2000.
- [ ] **grnboost**: 41 s at 500, 197 s at 1000, 1475 s at 2000 genes.
- [ ] **G2S** quick mode: about 65 min at 500 genes in the run (that included a hang); the tie-break budget may be the cost.
- [ ] **G2S pool deadlock**: one forked worker hung for an hour at 500 genes. Guarded (`run_chunks` kills idle pools after
  300 s), root cause unknown: the stderr-lock hypothesis did not reproduce; workers dump stacks to
  `/tmp/grn_sim_worker_<pid>.trace` after 600 s.
- [ ] Time every algorithm at 500 / 1000 / 2000 genes (the run gives most of it) and list the ones above O(n^2.5).

### Accuracy (results that look wrong at 500 genes, sensitivity / specificity)
- [ ] **inspre predicts nothing at every size** (sensitivity 0.0, specificity 1.0, 1254 errors = the number of true
  edges). Check thresholding, the ACE input and the spectral-radius condition at N >= 500.
- [ ] **ridge_cv almost empty** (0.059 / 1.0) although `k_interactions` comes from the truth: check the top-k selection and
  the self-term handling at 500 genes.
- [ ] **G2S quick (0.121) is far below S2G (0.347) and NIR (0.271)** though G2S should be at least as good: is the budgeted
  tie-break (`exact_budget_flops`) costing accuracy, or is the quick path wrong on this data?
- [ ] **cellbox 0.047, cpa_lite 0.028, arachne 0.044, grnboost 0.063, netdeconv 0.086, dmdgrn 0.141**: separate weak methods
  from bugs, against the controls (gene-feature 0.009, by-chance 0.003); check input scaling and orientation first.
- [ ] 40 archived datasets have no algorithm rows (28 small leftovers, 12 profiling), 10 have 6 of 19, 5 lack only
  `score_vote`.

## 2. Scale run (in progress)

- [ ] **1000 genes**: `cpa_lite` still running (about 80 min); `notears`, `cellbox` not run. 21 of 24 stored.
- [ ] **2000 genes**: 8 of 24 stored; to run: genie3, inspre, ridge_cv, nir, s2g, g2s, vote, score_vote, sfinn, dmdgrn,
  arachne, causalgrn, deeprig, notears, cellbox. `cpa_lite` skipped (memory).
- [ ] **After `cpa_lite` ends**: stop the container, back up the archive, run `scripts/compact_archive.py --apply --backup`
  (snapshot test: 8.56 GB -> 625 MB), start image `grn-sim-phpiorn:next` (rollback tag `before-scale-2026-11-12`), re-queue
  the remaining algorithms on the archived runs (jobs can run on archived networks since 66667c7). Re-run results for
  algorithms fixed under section 1 afterwards.

## 3. Perturb-seq proposal (`reports/GRN-Sim Perturb-seq proposal.md`)

- [ ] **Held-out-perturbation benchmark on single-cell screens** (proposal section 3): training mean / additive / linear /
  target-only / random-structure baselines, off-target metrics (DEG-weighted MSE, Pearson-delta, direction accuracy, rank),
  stratified by effect size. `evaluation/screen_benchmark.py` covers edge recovery only; `scripts/holdout_study.py` is
  still bulk-data only; the VCC 2026 scorer covers part of the metrics.
- [ ] Realism targets still missed: mediated share 0.79 at 500 genes (target 0.90), phenotype share 0.20; Virtual Cell tab:
  top-5% share 0.23-0.24, nonlinear phenotype 0.19, mechanistic density 0.10 at 100 genes.

## 4. Paper algorithms not yet implemented

Implemented so far: inspre, mean-difference, GRNBoost-lite, Pearson, bicycle-lite (no latent variables), DOTEARS-lite,
interventional consensus, gene-feature baseline (plus the 19 earlier ones).

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

## 5. Speed and storage of the platform (done 2026-11-12, details in CHANGELOG)

- [x] Sparse stored predictions, no per-algorithm SVGs above 100 genes, compressed payload columns, empty legacy columns,
  `compact_archive.py` (not yet run on the real archive, see section 2); run store bounded by bytes; kinetics kernel
  grouped products (500-gene generation 51 s -> 28 s); pool stall guard; per-algorithm job storage with error isolation;
  job list keeps jobs others depend on; jobs on archived runs.
- [ ] Load time of a 2000-gene run (4.2 s) is JSON parsing of X, U, trajectories as nested lists: store them as binary
  arrays. (orjson was measured: 0.5 s of 2.8 s.)
- [ ] Network generation at 500 genes (28 s): `_terms` power evaluation (about 10 s) and the ODE solver loop.

## 6. Housekeeping

- [ ] Push the last commit (`66667c7`, local only) and open the PR for `feat/inspre-and-interventional-baselines`.
- [ ] Cut a release (everything is under `[Unreleased]`).
- [ ] Real-data check of the Hill refit and transfer predictors on Norman 2019 / Replogle 2022.
- [ ] Resume interrupted jobs automatically (they come back as "interrupted"; a job on an archived run can be re-queued).
