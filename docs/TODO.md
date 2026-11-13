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
- [x] **ridge_cv at 2000 genes took over 95 minutes** (one sklearn RidgeCV per target: about N^4 work). All targets are now
  solved together from one inverse per fold and alpha (rank-1 correction for the left-out regressor): the same coefficients,
  alphas and adjacency to 1e-8 (`fast=False` keeps the per-target path), about N^3: 500 genes 28 s -> 3.3 s, 1000 genes
  280 s -> 22 s, 2000 genes > 95 min -> 156 s. (At 2000 genes its AUPR is 0.004: one condition per gene leaves the ridge
  with as many regressors as samples; check how the other algorithms behave there once the run has them.)
- [x] **cellbox was at chance on Hill data because its regulator route fed it the wrong relation** (`regulator_input_fit`
  gave the network X + U as the state and -X as the additive term, which does not match x = phi(Wx + u)/alpha). It now fits
  X ~ phi(W (X + U)) / alpha directly (`regulator_direct`): AUPR 0.061 -> 0.441 (100 genes), 0.015 -> 0.293 (200 genes),
  F1 0.43 / 0.37, the best algorithm in these audits, and about 50x faster (0.1-0.6 s against 8-10 s; 70 epochs are as good
  as 1000). Still 510 MB of torch per process.
- [x] **cpa_lite was at chance because a 4-dimensional bottleneck cannot hold a sparse N x N network and U was a latent shift
  that does not match the Hill relation.** Regulator mode is now the closed form of a rank-d linear autoencoder on the
  effective concentration (principal component regression on the SVD of X + U; d = 0.75 N up to 500 genes, N/2 above):
  200 genes F1 0.025 -> 0.294 (AUPR 0.018 -> 0.227), 1000 genes AUPR 0.127, 2000 genes AUPR 0.084 in 4 s (a gradient fit
  gave 0.001 at 2000 genes: it had not converged), no torch (it needed 510 MB). Additive mode unchanged. (The 1000-gene
  24 GB / 105 min job was the dense-graph rendering, fixed earlier.)
- [x] **ridge_cv at 2000 genes took over 95 minutes** (one sklearn RidgeCV per target: about N^4 work). All targets are now
  solved together from one inverse per fold and alpha (rank-1 correction for the left-out regressor): the same coefficients,
  alphas and adjacency to 1e-8 (`fast=False` keeps the per-target path), about N^3: 500 genes 28 s -> 3.3 s, 1000 genes
  280 s -> 22 s, 2000 genes > 95 min -> 156 s. (At 2000 genes its AUPR is 0.004: one condition per gene leaves the ridge
  with as many regressors as samples; check how the other algorithms behave there once the run has them.)
- [x] **cellbox was at chance on Hill data because its regulator route fed it the wrong relation** (`regulator_input_fit`
  gave the network X + U as the state and -X as the additive term, which does not match x = phi(Wx + u)/alpha). It now fits
  X ~ phi(W (X + U)) / alpha directly (`regulator_direct`): AUPR 0.061 -> 0.441 (100 genes), 0.015 -> 0.293 (200 genes),
  F1 0.43 / 0.37, the best algorithm in these audits, and about 50x faster (0.1-0.6 s against 8-10 s; 70 epochs are as good
  as 1000). Still 510 MB of torch per process.
- [x] **cpa_lite was at chance because a 4-dimensional bottleneck cannot hold a sparse N x N network and U was a latent shift
  that does not match the Hill relation.** Regulator mode fits X ~ Decoder(Encoder(X + U)) with latent size >= N / 2 (plateau
  from about N / 2), closed-form reduced-rank fallback without torch: 200 genes F1 0.025 -> 0.319, AUPR 0.018 -> 0.224;
  100 genes AUPR 0.065 -> 0.35. Additive mode unchanged. (The 1000-gene 24 GB / 105 min job was the dense-graph rendering,
  fixed earlier.)
- [x] **genie3 and grnboost were scored transposed on Hill data** (declared A[target, regulator], which is right for X alone). On
  X + U (what they now get) a child's response predicts its parent's perturbation, and the output matches T[regulator,
  target] as returned on 6 of 6 networks (GENIE3 AUPR 0.25-0.44 as returned against 0.04-0.13 transposed; the 200-gene audit
  showed 0.035). Declared for additive data only. A check of every algorithm's scored against transposed F1 found no other
  orientation error; pearson and arachne are symmetric (their F1 depends only on whether the K cut-off is taken per
  regulator or per target, which is a property of this generator's degree distribution, left as it is).
- [x] **vote and score_vote repeated the G2S (and NIR) fit** that the g2s / nir jobs of the same network had just run (825 s
  of G2S three times at 1000 genes): a small result cache (`inference/_cache.py`, keyed on the data hash, settings and
  arguments) shares them; an ensemble's reported run time still includes what its members got from the cache; vote now uses
  the same G2S settings as the g2s entry so they match.
- [x] **G2S checked: not a bug.** Its F1 is mid-table (500 genes 0.130, 1000 genes 0.161; AUPR 0.05-0.07); the earlier
  "far below S2G" compared sensitivity alone, and S2G reached it by predicting 6,657 / 21,861 edges. On this data the AIC
  optimum is exactly 2 regulators for every target (the target `U` row is a single spike), so the cap (`k`, `2k`, none), the
  drop rule (`loo` / `exact`) and the regulator formulation (X+U) change nothing. Real 500-gene time is 12 s: the 3906 s in
  the archive was the pool hang. **S2G had no edge cap while every other algorithm gets `k`**; they were designed to choose
  their own size (MDL / AIC) but on this data they do not (S2G 13-22 edges per gene, G2S exactly 2), so by decision
  (2026-11-12) they are given `k` when it is known, as the others are: S2G F1 0.109 -> 0.192 (500 genes), 0.082 -> 0.219
  (1000 genes), AUPR unchanged. This uses the truth's edge density; documented in `docs/SCALE_BENCHMARK.md`; compare with
  AUPR when edge counts differ. (Committed; not yet in the container image.)
- [x] **arachne, netdeconv, pearson, grnboost, genie3, causalgrn ran on X alone, but the Hill regulator-input perturbation
  is invisible in X** (the perturbed gene does not respond; the data satisfy X = cF(X+U)). Given X + U (new
  `EffectiveConcentration` wrapper, registry, Hill data only) AUPR rises 2-3x on calibrated 100-200 gene networks
  (arachne 0.03 -> 0.09, netdeconv 0.06 -> 0.15, pearson 0.04 -> 0.12, genie3 0.04 -> 0.12, causalgrn 0.14 -> 0.22) and on
  default-parameter 40-gene networks (6 seeds, 4-6 wins each). Not wrapped, no gain: sfinn, deeprig, atlasshift, mean_difference,
  notears (+14%), the controls. Committed; not yet in the container image.
- [x] **dmdgrn had two bugs.** (1) On Hill regulator-input data it inverted the additive relation A X = -U instead of
  X = cF(X + U): it now fits on (X + U, -X) like NIR / FAST / RidgeCV (`perturbation_mode="regulator"`; 200 genes AUPR
  0.036 -> 0.241, F1 0.12 -> 0.30, level with NIR and SFINN). (2) Its output orientation was undeclared: on additive
  (Gardner) data the raw output is A[target, regulator] and scored 2-3x better transposed (like S2G and NIR); declared now.
- [ ] **Archived additive-data runs scored dmdgrn with the wrong orientation.** `scripts/rescore_archive_orientation.py --only
  dmdgrn` re-orients it in already rescored runs (additive data models only: Hill-kinetics runs mix regulator-input and
  additive fits, so their old convention is not recoverable; runs dated before 2026-11-13; `orientation_fixes` makes it
  run once). Tested on copies of real archive rows. Dry run on the real archive: 14 additive runs would change
  (2 Gardner, 12 physical), best algorithm changed in 0 runs. **Not applied**: the studio container was writing to the
  archive; apply with `--apply` (it backs up first) when nothing is.
- [x] **Pool deadlock: root cause found and fixed (2026-11-13).** Reproduced with the real G2S job on a scratch server
  while a poller hit the API 20 times a second (it hung on the third run). The hung workers' own stack dumps end in
  `multiprocessing.util._flush_std_streams`: a request-handler thread writing its log line held a stream's buffer lock at the
  instant of the fork, the child inherited it locked, and the worker hangs for ever at its exit flush after finishing its
  work, while the parent waits in `shutdown(wait=True)` (so the stall guard, which watches for missing results, never fired).
  Fix: workers replace stdout / stderr with fresh objects first thing (`_parallel.reset_std_streams`, in both pool
  initializers) and the parent kills a worker that has not exited 30 s after its work (`EXIT_GRACE_SECONDS`). Tests: a
  stream whose flush blocks in children does not hang the pool (0.1 s with the fix, 12 s without) and a worker that
  will not exit is killed. Verified on the real job: the polling loop that hung on run 3 before ran 30 + 12 repetitions of
  the 500-gene G2S job (with the guard messages at zero after a false alarm in the first version of the exit wait was fixed).
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
