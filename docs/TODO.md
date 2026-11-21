# To do

State as of 2026-11-21. Scope decision: nothing above 500 genes is planned; items that needed larger runs were dropped. Source for the algorithm list: `research_notes/Perturb seq design and modelling/`
(`causal_grn_inference.md`, `prediction_models_benchmarks.md`); gap audit in `docs/INFERENCE_GAP_ANALYSIS.md`.

## 1. Algorithm bugs (top priority: speed, memory leakage, accuracy)

Evidence: the scale run (`replogle_like`, seed 7, 500 / 1000 / 2000 genes; archived runs `iorn-838581`, `iorn-592020`,
`iorn-846813`) and `scripts/audit_algorithms.py` (every algorithm in its own process: time, peak memory, accuracy; run
at 200 genes, 2026-11-12). Results of runs made before these fixes stay in the archive as they were; re-run to compare.

### Fixed (2026-11-21, release 1.0.5: review of all 25 algorithms)
- [x] **FAST ranked low because of the identity in its first estimate.** `A_0 = (x - u) X^+ = I + A`, so each row's top-K spent a slot on the
  diagonal (a self-edge the truth does not have; the scorer counts it as a false positive). The studio now ranks by `|A_0 - I|` without the
  diagonal (`FastSVD(rank_without_identity=True)`; the class default stays MATLAB-identical). Hill data, 50 / 100 genes, F1 0.35 / 0.28 ->
  0.45 / 0.35 (NIR 0.45 / 0.37). Still weak on the linear engine (0.16 / 0.05 against NIR 0.27 / 0.15): the unregularised truncated-SVD inverse
  of a noisy X, larger damping did not help. `docs/results/fast_diagnosis.md`.
- [x] **GRNBoost-lite looked at 10% of the regulators per split**, GRNBoost2's setting for thousands of genes (1 of 15 at 16 genes). Now about
  30 candidates per split (`clip(30 / regulators, 0.1, 1)`; unchanged from 300 genes up): F1 0.18 -> 0.25 at 50 genes, 0.13 -> 0.20 at 100.
- [x] **CellBox was at chance at 100 genes** (F1 0.03 on the 100-gene audit network, 0.15 over 4 seeds, against 0.25 at 70 and 200): the size-scaled
  penalty started above 100 genes, leaving the full penalty on 60-100. The reference is now 55 genes (F1 0.20 over 4 seeds at 100, 0.06 on the audit network, still poor there; unchanged at 70, 200,
  250, 500). Stored CellBox rows were recomputed for 56-266 genes; the 500- and 1000-gene rows keep the old penalty (measured neutral there).
- [x] **`llc_mean` registered** (LLC mean-shift equations): 25 algorithms. Calibrated screens and partial screens: top; the studio's Hill data: mid-field.
- [x] **Square-only algorithms** (s2g, g2s, vote, score_vote, netdeconv) raise "needs a square screen ..." with the remedy (`PartialAdapter`) instead of a
  matmul shape error on a partial screen or replicates. `tests/test_algorithm_edge_cases.py` runs every registered algorithm on tiny networks,
  a constant gene, a control column and duplicated conditions (finite n x n output on both data models).
- Reviewed and left as they are:
  - **ARACNE** returns about 20 edges at 100 genes. The cause is not the DPI (switching it off still gives 21): the preprocessor's shrunk
    correlation matrix is near zero for almost every pair (median |r| 0.0005), the adaptive MI cut (mean + 0.15 sd of a heavy-tailed
    distribution) lets 19 pairs through, and K per gene cannot be filled. Without the cut the DPI reduces the dense MI graph to a tree-like
    skeleton (about N - 1 edges, recall at most 1/K), and F1 hardly moves (0.16 -> 0.14 at 50 genes, 0.10 -> 0.12 at 100, 4 seeds), so the
    code was left alone.
  - **G2S** is the weakest regression variant by design (leave-one-out shortcut, documented); **NOTEARS** decline with more perturbed genes is
    still unexplained; **CellBox** stays well below NIR at 100 genes (a nonlinear fit of a mostly linear response).

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
  default `lambda1` 0.0005; mean AUPR over 6 networks 0.054 -> 0.15 (100/seed 11: 0.06 -> 0.37). **Superseded in 1.0.3:** a fixed value cannot suit both data types (0.0005 overflowed `Tr(exp(W o W))` on the statistical screens at 200 genes: NaN weights, empty graph); the default is now data-relative (`lambda1=None`: fractions 0.01 ... 0.8 of lambda_max, smallest first, first finite non-empty fit with at most 12 edges per gene; 1.0.3 started at 0.1 and over-shrank mechanistic networks, fixed in 1.0.4). Tab 4 mean AUPR 0.191 -> 0.277 (rank 16 -> 9).
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
  AUPR when edge counts differ. (In the 1.0.x images.)
- [x] **arachne, netdeconv, pearson, grnboost, genie3, causalgrn ran on X alone, but the Hill regulator-input perturbation
  is invisible in X** (the perturbed gene does not respond; the data satisfy X = cF(X+U)). Given X + U (new
  `EffectiveConcentration` wrapper, registry, Hill data only) AUPR rises 2-3x on calibrated 100-200 gene networks
  (arachne 0.03 -> 0.09, netdeconv 0.06 -> 0.15, pearson 0.04 -> 0.12, genie3 0.04 -> 0.12, causalgrn 0.14 -> 0.22) and on
  default-parameter 40-gene networks (6 seeds, 4-6 wins each). Not wrapped, no gain: sfinn, deeprig, atlasshift, mean_difference,
  notears (+14%), the controls. In the 1.0.x images.
- [x] **dmdgrn had two bugs.** (1) On Hill regulator-input data it inverted the additive relation A X = -U instead of
  X = cF(X + U): it now fits on (X + U, -X) like NIR / FAST / RidgeCV (`perturbation_mode="regulator"`; 200 genes AUPR
  0.036 -> 0.241, F1 0.12 -> 0.30, level with NIR and SFINN). (2) Its output orientation was undeclared: on additive
  (Gardner) data the raw output is A[target, regulator] and scored 2-3x better transposed (like S2G and NIR); declared now.
- [x] **Archived additive-data runs scored dmdgrn with the wrong orientation** (applied to the archive on 2026-11-13, 14 runs). `scripts/rescore_archive_orientation.py --only
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
- [x] Time and memory of every algorithm at 500 genes with the 1.0.4 code (2026-11-20): `docs/results/timings_500/` (modular audit and the three statistical generators; all 24 algorithms finish, slowest grnboost about 64 s, peak 2.0 GB). Tab 4 whole-run totals re-measured, limits unchanged. Still open: a per-algorithm memory estimate that skips or warns before a run.
- [x] Incomplete archived runs: none left (516 of 516 runs have all 25 algorithms after the 1.0.5 rerun, `scripts/clean_archive.py --verify`; it was 512 of 512 with 24).
- [x] **Partial screens across sizes and fractions (2026-11-20):** 100 and 250 genes at 6/20/50/75% perturbed, with imputation of the missing perturbations from source contexts (`docs/results/partial_screen_grid.md`). 500 genes not run. Open: NOTEARS falls as more genes are perturbed (cause unknown); the imputation experiment covers six methods and synthetic sources only.

## 2. Scale run (in progress)

- [x] **Audit follow-ups (2026-11-13):** 100 and 250-gene audits stored in the archive and as `docs/results/audit_100_250.*`; inference now
  runs on X and U rounded to the archive's 4 decimals so archived runs re-analyse reproducibly; tab 5 rebuilt from the archived sweep
  networks and stored in the `overall_stats` table; `.h5ad` downloads zipped; empty-search row, job error text, `$TMP_` labels,
  404 for missing `.svg`/`.dot`, phone-width Virtual Cell form.
- [x] **Server start-up archived runs with few algorithms** (fixed in 1.0.0: seeded examples are completed with the remaining algorithms by queued jobs).
  Was: Starting a container with no flat/oscillating example in the history seeds
  four small example runs with 6 algorithms each, and each start archives the studio's start-up run with no inference at all (6 such
  runs appeared on 2026-11-13 and were completed with `scripts/fill_archive_algorithms.py`; the archive is 512 of 512 complete).
  Fix: seed with every registered algorithm, and do not archive the start-up run until it has results.
- [x] **Archived runs scored before the rounding change** were all re-scored with the current code on 2026-11-16 (500 runs, 0 failures; backup `data/backups/rgn_db.before-rerun-2026-11-16.sqlite`). The NOTEARS column predates the 1.0.3 penalty change and is re-scored separately (see below).
- [ ] **CellBox / GRNBoost differ slightly between the host and the container** (scikit-learn 1.6 vs 1.9, NumPy 1.23 vs 2.4, torch):
  compare scores within one environment, or pin the host environment to the image's versions.
- [ ] **`.h5ad` export size:** 140 MB (zipped smaller) for 250 genes with 400 cells per condition; consider sparse counts and a size
  warning before the export starts.
- [x] **Archive cleaned and complete (2026-11-13):** 504 runs of at most 1000 genes, every run with all 24 algorithms under one key
  (`scripts/clean_archive.py --verify`); 2000-gene runs removed (numbers kept in `docs/results/scale_benchmark.json`); DMD-GRN re-scored;
  500 and 1000 gene benchmark networks re-run uniformly; per-algorithm tables in `docs/results/archive_leaderboard.md`.
- [x] **All 22 algorithms stored at 500, 1000 and 2000 genes** (2026-11-13); table in `docs/SCALE_BENCHMARK.md`, rendered from
  `docs/results/scale_benchmark.json`. G2S at 2000 genes: 71 min with numba and 8 workers (quick mode, see section 1).
- [x] Archive compacted (the live archive is about 0.8 GB; the pre-compaction copy `data/rgn_db.pre-profiling-cleanup-2026-11-04.sqlite` is a 2.3 GB backup that can be deleted when no longer wanted).

## 3. Perturb-seq proposal (`reports/GRN-Sim Perturb-seq proposal.md`)

- [x] **Held-out-perturbation benchmark on single-cell screens** (proposal section 3; 2026-11-19): `evaluation/heldout_screen.py`
  (control / training mean / random response / co-expression neighbours / ridge-linear baselines; Pearson-delta, DEG-weighted
  MSE skill, direction accuracy, top-k overlap; silent vs active strata), `scripts/heldout_screen_benchmark.py`, results and findings
  in `docs/results/heldout_screen_*.md`. Not done: the mechanistic (Hill) simulator screens, published predictors (GEARS, scGPT,
  CPA), and the double-perturbation split.
- [ ] **Statistical generators have no shared response.** On real data (Norman 2019) the training mean predicts a held-out
  perturbation with correlation 0.48; on every simulated screen it is about 0.00 (`docs/results/heldout_screen_findings.md`).
  Add a shared / state component to the generators and calibrate it to that number.
- [ ] Realism targets still missed: mediated share 0.79 at 500 genes (target 0.90), phenotype share 0.20; Virtual Cell tab:
  top-5% share 0.23-0.24, nonlinear phenotype 0.19, mechanistic density 0.10 at 100 genes.

## 4. Paper algorithms not yet implemented

Implemented so far: inspre, mean-difference, GRNBoost-lite, Pearson, bicycle-lite (no latent variables), DOTEARS-lite,
interventional consensus, gene-feature baseline (plus the 19 earlier ones); LLC mean-shift joined the registry in 1.0.5 (25 algorithms).

| Priority | Method | Paper | Note |
|---|---|---|---|
| 2 | **LLC** (linear cyclic, latent confounders) | Hyttinen et al. 2012 | **Registered 2026-11-21 as `llc_mean`** (mean-shift part, `inference/llc.py`): best on calibrated screens and partial screens, mid-field on the studio's Hill data (`results/paper_algorithms.md`). Covariance part not done. |
| 2 | **bicycle, full** (latent variables, Lyapunov likelihood) | Rohbeck et al., CLeaR 2024 | Only the lite version exists; practical to ~250 genes. |
| 2 | **SCEPTRE** (calibrated ACE) | Barry et al. | Tried (simplified conditional resampling, `test="sceptre"`): equals the Welch test within noise. Full NB-regression SCEPTRE not done. |
| 3 | **DCD-FG** | Lopez et al., NeurIPS 2022 | Tried (linear-Gaussian lite, `experimental/dcdfg.py`): 0.06-0.25 AUPR, 31 s; does not beat the baselines; fits cells, so it cannot be registered. |
| 3 | **DCDI / full DOTEARS** | interventional DAG learners | Not implemented: same family as the lite versions (DOTEARS-lite, DCD-FG-lite). |
| 3 | **GIES / IGSP** | Hauser and Buehlmann 2012; Wang, Squires, Uhler | Not implemented (slow beyond ~100 genes, binary output, no library available). |
| 3 | **SparseRC** | CausalBench challenge | Not implemented (needs root-cause assumption and observational data). |
| 4 | **Dictys / RENGE** | Wang 2023; Ishikawa 2023 | Need time-series data, which the archive does not store. |
| skipped | **CLIPR / latent causal diffusions** | arXiv 2601.15341 | No code available (decision 2026-11-04). |

Papers still to read before implementing: LLC, GIES, IGSP, DCDI, NODAGS-Flow, Dictys.

## 5. Speed and storage of the platform (done 2026-11-12, details in CHANGELOG)

- [x] Sparse stored predictions, no per-algorithm SVGs above 100 genes, compressed payload columns, empty legacy columns,
  `compact_archive.py` (not yet run on the real archive, see section 2); run store bounded by bytes; kinetics kernel
  grouped products (500-gene generation 51 s -> 28 s); pool stall guard; per-algorithm job storage with error isolation;
  job list keeps jobs others depend on; jobs on archived runs.
- [ ] Network generation at 500 genes (28 s): `_terms` power evaluation (about 10 s) and the ODE solver loop.

## 6. Housekeeping

- [x] Branches merged into `main` and released as 1.0.0 (2026-11-13).
- [x] Real-data held-out check on Norman 2019 and Replogle 2022 (K562 essential pseudobulk, downloaded 2026-11-20, 80 MB, git-ignored): `scripts/real_data_heldout.py [--dataset replogle]`, `docs/results/real_data_heldout*.md`; the training mean beats the structure-using predictors on both. Not done: the Hill refit on real data.
- [x] Interrupted inference jobs on archived networks are re-queued after a restart (generating jobs still come back as failed: re-running them would make a second network).
