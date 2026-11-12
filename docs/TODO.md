# To do

State as of 2026-11-12 (evening). Source for the algorithm list: `research_notes/Perturb seq design and modelling/`
(`causal_grn_inference.md`, `prediction_models_benchmarks.md`); gap audit in `docs/INFERENCE_GAP_ANALYSIS.md`.

## 1. Algorithm bugs (top priority: speed, memory leakage, accuracy)

Evidence: the scale run (`replogle_like`, seed 7, 500 / 1000 / 2000 genes; archived runs `iorn-838581`, `iorn-592020`,
`iorn-846813`) and `scripts/audit_algorithms.py` (every algorithm in its own process: time, peak memory, accuracy; run
at 200 genes, 2026-11-12). Results of runs made before these fixes stay in the archive as they were; re-run to compare.

### Fixed (2026-11-12)
- [x] **inspre returned an empty graph on Hill-simulator data** (F1 0.00, AUPR 0.005 at 500 genes): in the regulator-input
  perturbation the perturbed gene itself barely responds (median own response -5e-5), and inspre divided each response by
  it. `InspreAlgorithm(perturbation_mode="regulator")` applies it to the response of X + U (own effect ~ u); the registry
  uses it on Hill data. 500 genes: F1 0.255, AUPR 0.135; 200 genes: AUPR 0.264 (the best of the 24).
- [x] **Top-K was gated by a magnitude threshold, or K was not wired at all** (ridge_cv kept 90 of 600 edges; genie3 and
  cpa_lite had no K and predicted 18,487 and 4,902 edges, cellbox 40). K regulators per target now whenever K is known
  (genie3, cpa_lite, ridge_cv, cellbox; thresholds apply only without K). 200 genes F1: ridge_cv 0.172 -> 0.334,
  genie3 0.038 -> 0.102 (AUPR unchanged, it is threshold-free).
- [x] **NOTEARS emptied the graph on some networks** (17-45 predicted edges at 100-300 genes with the L1 penalty 0.02):
  default `lambda1` 0.0005; mean AUPR over 6 networks 0.054 -> 0.15 (100/seed 11: 0.06 -> 0.37).
- [x] **cpa_lite "memory leak" and 105-minute job at 1000 genes** (24 GB): not the model (a 4 MB linear autoencoder) but
  the result processing: with no K it predicted a dense graph and three 9-state SVG layouts were rendered for it. No graph is
  rendered above 100 genes now (generation, inference, archive load, studio benchmark, parameter sweep) and K bounds the
  edges: 1000 genes 7.3 s, 1.7 GB peak.
- [x] Per-algorithm job storage with error isolation; stall guard for forked pools; run store bounded by bytes.

### Still open
- [ ] **cellbox is at chance level** (AUPR 0.030-0.035 vs 0.026 chance at 100 genes) whatever the epochs (70 to 1500) or
  penalty; 500 MB extra memory at 200 genes; about 21 min at 500 genes. Check the regulator-input adaptation (the model
  adds u to the activation, the simulator adds it to the regulator concentration) and the GPU path.
- [ ] **cpa_lite is at chance level** too (AUPR 0.018 vs 0.019): a linear autoencoder without structure; decide whether to
  keep it (it still needs 500 MB of torch).
- [ ] **G2S (quick) is below S2G and NIR** (200 genes F1 0.157 vs 0.156 and 0.310; 500 genes sensitivity 0.121 vs 0.347):
  check the budgeted tie-break and the `loo` rule against the exact one on this data.
- [ ] **arachne 0.07, netdeconv 0.10, grnboost 0.087, dmdgrn 0.12, pearson 0.08 F1 at 200 genes**: weak or wrong? Compare
  with the controls (gene-feature 0.015, by-chance 0.025) and check orientation / scaling.
- [ ] **G2S pool deadlock** (one forked worker hung for an hour at 500 genes): guarded by `run_chunks` (kills idle pools
  after 300 s), root cause unknown; workers dump stacks to `/tmp/grn_sim_worker_<pid>.trace` after 600 s.
- [ ] Time and memory of every algorithm at 500 / 1000 / 2000 genes with the fixed code (the audit script takes
  `--genes`); a per-algorithm memory estimate that skips or warns before a run.
- [ ] 40 archived datasets have no algorithm rows (28 small leftovers, 12 profiling), 10 have 6 of 19, 5 lack only
  `score_vote`.

## 2. Scale run (in progress)

- [ ] **1000 genes**: 21 of 24 stored (`cpa_lite` was stuck 105 min in the old image, `notears`, `cellbox` not run).
- [ ] **2000 genes**: 8 of 24 stored; to run: genie3, inspre, ridge_cv, nir, s2g, g2s, vote, score_vote, sfinn, dmdgrn,
  arachne, causalgrn, deeprig, notears, cellbox. `cpa_lite` is fine now (see section 1).
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
