# Changelog

All notable changes. The format follows [Keep a Changelog](https://keepachangelog.com/); the project uses semantic
versioning once it leaves 0.x.


## Unreleased - Virtual Cell tab: tuning round and measured limits

- Second calibration round (effect scale x noise x weight spread, `calibrate_statgen.py --spreads`): linear SEM and NB copula
  now meet every shown realism target; the Boolean model's effect tail is fixed; the nonlinear model stays short on phenotype
  share (its saturating links cap propagation) and its noise levels hardly change difficulty - both stated in the generated
  calibration document.
- Edge weights capped at |w| <= 1: the spectral bound did not limit single weights on acyclic pathways, where a heavy-tailed
  weight of 9 drove log expression to ~140 and crashed the measurement; regression test over every pathway and generator.
- Cross-atlas size limits are measured (`docs/results/vcc_cross_atlas_timings.json`, `tests/test_vcc_limits.py`): the
  largest size whose slowest whole run fits 600 s and whose double-perturbation screen fits 180 s. Mechanistic 150 genes (was
  32), Boolean 150, linear SEM / NB copula / nonlinear SCM 200 with every pair screened; Boolean screens its 64 most-connected
  genes; Hill refits at every size.
- Simulated source contexts renamed "Source context A/B/C" and the target-context menu shows the rewired share only (the
  cell-line names suggested real data).

## Unreleased - Virtual Cell tab: correctness pass

- Perturbation dependencies tested for every generator (`tests/test_perturbation_dependencies.py`): a perturbation moves exactly
  the genes downstream of it, double perturbations interact only downstream of both partners, at most 5% of independent gene
  pairs look dependent in the cells. Fixed on the way: the Boolean model's Monte Carlo truth leaked effects of up to 0.08 log2
  onto independent genes (now common random numbers: exactly 0); total-count normalisation made 9-14% of independent pairs look
  dependent, so effects are now estimated with median-ratio size factors (`estimate_ace(normalisation="median_ratio")`, opt-in;
  used by the Virtual Cell benchmark).
- Algorithm admission test (`tests/test_vcc_algorithm_admission.py`): every benchmarked algorithm must beat the hubness-only
  control by 1.5x on noise-free data. Dropped: CPA-Lite and GRNBoost on effect profiles; GRNBoost now runs on single cells.
- CellBox: penalties scaled by (100 / N)^2 and at least 150 epochs above 100 genes (its summed L1/L2 penalty grew with N^2 and
  shrank every weight at 200 genes: AUPR 0.08 -> 0.32-0.49); unchanged at or below 100 genes.
- Cross-atlas comparison on statistical data (`generator=` in `run_vcc_comparative_benchmark` and the API) up to 200 genes;
  above 32 genes it runs in a spawned process (forking the algorithms' process pools from the threaded server deadlocked), the
  page polls (HTTP 202 with the current stage). Double-perturbation screen over the 32 most-connected genes and no Hill refits
  above 32 genes, both stated on the page. Default perturbation CRISPRi, default data linear SEM.
- Findings and calibration documents are generated from the data (`grn_sim.evaluation.vcc_findings`, `grn_sim.statgen.report`)
  and tests fail when the documents and the data disagree. Loading bar names every pending request.

## Unreleased - Virtual Cell tab: statistical data generators

- New package `grn_sim.statgen`: data from statistical models instead of the panel-1 Hill kinetics - linear SEM, nonlinear
  additive-noise SCM, negative-binomial Gaussian copula, probabilistic (logistic) Boolean network - on the same graphs
  (`preset_graph` reproduces a mechanistic preset's graph for the same seed) with the shared guide-dose and measurement layers.
  Tested not to touch the kinetics.
- Noise levels low / medium / high (0.5 / 1 / 2 x calibrated noise) for every generator, mechanistic included; calibrated against
  the genome-scale Perturb-seq realism targets (`scripts/calibrate_statgen.py`, `scripts/verify_statgen_noise.py`,
  `docs/results/statgen_calibration.md`).
- Corpus: 1000 simulations (100 settings x 10 seeds: 5 generators x noise x perturbation type, pathways, 50-200 genes, double
  knockouts, partial screens, activation) x 23 algorithms (all registered algorithms except AtlasShift and by_chance).
  Ranking with bootstrap 95% intervals of the mean and of the rank. Output orientation of the registered algorithms on screen
  data fixed once on separate seeds (`TRANSPOSED_ON_SCREENS`).
- Tab 4: generator and noise filters, comparison by generator, live simulations with a generator choice, "all five on the same
  graph" comparison and the panel-1 network as the graph (mechanistic vs statistical data on one network).
- `run_screen(cell_options=...)` overrides the cell model; benchmark mechanistic screens use the preset's 0.92 guide efficiency.

## Unreleased - Virtual Cell tab: pathways and screen statistics

- Tab 4 now states what it is: experiments simulated from statistics (Perturb-seq-like screens generated from the panel-1 simulator),
  not a comparison with a mechanistic model. The AtlasShift/mechanistic header box is gone (the cross-atlas section stays, up to 32 genes).
- New pathway architectures (`cascade`, `feedforward`, `lineage`, `regulon` in `core/topology.py`, presets in `perturbseq/calibration.py`,
  also selectable in panel 1): signalling cascade, feed-forward-loop network, lineage master regulators, overlapping regulons.
- Screens of up to 200 genes from the tab (`/api/vcc/screen_simulate`, fast algorithms only); results are saved to the user's
  data folder and join the statistics.
- New gene-selection rules for partial screens: `expressed` and `hubs` (oracle), next to `effective` and `random`.
- Benchmark of 290 simulations (29 settings x 10 seeds: pathway, size, CRISPRi / knockout / activation, double knockouts, selection)
  with a paired random-ranking and hubness-only control per simulation, Wilcoxon tests, verdicts and filters
  (`evaluation/vcc_benchmark.py`, `scripts/run_vcc_benchmark.py`, `/api/vcc/screen_benchmark`, `docs/results/vcc_screen_benchmark.md`).
- S2G and G2S in the benchmark run in their regression form (`matlab_mode=False`), which accepts partial screens and double knockouts.

## Unreleased - partial screens

- Documentation: one page per algorithm in `docs/algorithms/` (generated from `inference/descriptions.py`, checked by a test), a docs index
  (`docs/README.md`), result files moved to `docs/results/`.
- Partial Perturb-seq screens: `ScreenDesign`, `run_screen(n_perturbed=, perturbed_selection=, genes=)`, `grn-sim perturbseq --perturbed N
  --selection effective|random`, `?perturbed=&selection=` on the studio API, panel fields. Effective-gene selection from the simulated
  network, shortfall reporting, own statistics categories (design, identifiability, coverage, blocks), selection-biased realism
  statistics marked not comparable. `adata.uns["screen_design"]`.
- Algorithms for partial screens: `PartialAdapter`, `registry.build_partial`, `partial` capability in `registry.describe()`; inspre,
  SFINN, AtlasShift and CellBox accept fewer perturbation columns than genes.
- New: inspre, mean difference, Pearson, GRNBoost-lite, gene-level features, bicycle-lite, DOTEARS-lite, calibrated ACE estimates,
  consensus, direct/total-effect screen benchmark, prior-quality experiment (docs/INFERENCE_GAP_ANALYSIS.md).
- Fixed: RidgeCV in additive mode now includes the own-gene term (it scored at the random level); CellBox rescales its inputs
  (AUPR 0.03-0.06 -> 0.45-0.60 on 100-gene screens); the cell simulator no longer raises the BLAS thread count (forked process pools
  could deadlock afterwards).

## [Unreleased]

### Changed
- **Algorithm fixes found by `scripts/audit_algorithms.py`** (one process per algorithm: accuracy, time, peak memory).
  inspre applies to the response of X + U on Hill-simulator data (it returned an empty graph; 500 genes F1 0 -> 0.255);
  genie3, cpa_lite, cellbox and ridge_cv keep K regulators per target whenever K is known (ridge_cv F1 0.17 -> 0.33,
  genie3 0.04 -> 0.10 at 200 genes; thresholds apply only without K); CellBox fits X ~ phi(W (X + U)) / alpha on Hill data instead of feeding the additive model the wrong relation (AUPR 0.015 -> 0.293 at 200 genes, 50x faster); DMD-GRN fits the regulator-input relation on Hill data (200 genes AUPR 0.036 -> 0.241) and declares its additive-data output orientation (it was scored transposed); arachne, netdeconv, pearson, grnboost, genie3 and causalgrn receive the effective concentration X + U on Hill data (the perturbation is invisible in X; AUPR 2-3x on calibrated networks); S2G is capped at K on Hill data like the others (it predicted 13-22 edges per gene: F1 0.11 -> 0.19 at 500 genes, AUPR unchanged; documented in docs/SCALE_BENCHMARK.md); S2G is capped at K on Hill data like the others (it predicted 13-22 edges per gene: F1 0.11 -> 0.19 at 500 genes, AUPR unchanged); NOTEARS default `lambda1` 0.02 -> 0.0005 (it
  shrank some networks to 17-45 edges). No graph layout or SVG is computed above 100 genes anywhere (CPA-Lite at 1000
  genes: 105 minutes and 24 GB -> 7 s and 1.7 GB).
- **Archive size and speed at 500-2000 genes.** Predicted matrices are stored as coordinate lists above 60 genes; the
  per-algorithm comparison SVGs are not rendered or stored above 100 genes (the page draws graphs of more than 25 nodes as
  modules from the stored comparison graph, and a trivial algorithm now takes 0.9 s instead of 8 s at 500 genes); payload
  columns over 1 M characters are stored zlib-compressed and read back transparently; the PHP-era columns nothing reads are
  left empty above 100 genes; the archive summary no longer reads every run's payload (4.6 s -> 0.43 s).
  `scripts/compact_archive.py` converts an existing archive (dry run by default, `--apply` needs a backup path and verifies
  each run: 8.56 GB -> 625 MB on the real one).
- The in-memory run store is bounded by bytes (`GRN_SIM_RUN_STORE_GB`, default 8) and keeps runs queued jobs need; a pool of
  forked workers that makes no progress for 5 minutes is killed and its chunks are finished in the calling process;
  background inference jobs store each algorithm's result as it finishes and an algorithm that fails no longer discards
  the others; the job list keeps a finished job while queued jobs depend on it; the kinetics kernel uses grouped products
  (bit-identical, about 1.8x faster network generation at 500 genes).
- **CellBox can run on a GPU.** `CellBoxNeuralODE(device="auto"|"cpu"|"cuda")` (default `auto`, or the `GRN_SIM_DEVICE`
  environment variable): `auto` uses a usable CUDA GPU for networks of at least 300 genes and the CPU otherwise, `cuda`
  fails with a clear error when there is no GPU, and an `auto` run whose GPU step fails (for example out of memory) finishes on
  the CPU and says so in `diagnostics["gpu_fallback"]`. The fit reports `diagnostics["device"]`. Measured on an RTX 5080:
  0.4 s instead of 35-40 s for 2000 genes and 0.2 s instead of 2 s for 500 genes, with the same network (weights are drawn on
  the CPU, so a seeded run starts identically on either device). The Docker image installs the CPU build of PyTorch, so it stays on
  the CPU unless you build it with a CUDA PyTorch and run it with `--gpus all`.
- **Archive tab is paged.** It no longer downloads the whole history (3.3 MB, 3.6 s for 340 runs): `/api/history` accepts
  `set`, `page`, `page_size`, `sort`, `order` and `q` and returns one page of slim rows (about 20 KB), ranked, sorted and
  searched on the server, so sorting a column orders the whole archive, not only the visible page. The tab has a pager, a page
  size and a run-id search. Without those parameters `/api/history` answers as before. The ranked sets and the by-type summary
  are cached against the archive file's version.

### Fixed
- **Archive: the algorithm comparison follows the active filter.** It always averaged the whole archive whatever the curated
  view or id search showed; it now covers exactly the runs the table lists (`/api/archive/algorithm_comparison?set=&q=`) and
  says which ones ("Averages over the 15 runs in the current list: ..."). The category cards above it still describe the
  whole archive.
- **Archive: the id search says what an id is.** The box shows an example (`iorn-0059`) and a hint (`iorn-005930` studio runs,
  `iorn-osc...` / `iorn-flt...` presets, `sweep-...` parameter sweeps; any part, case-insensitive), and an empty result
  names the search text and, in a curated view, points to "All runs" for searching the whole archive.
- **Server hang under concurrent requests.** OpenBLAS's thread pool and `fork()` deadlocked the process pools (G2S, S2G,
  GENIE3, ...) when several request threads computed at once: six concurrent G2S fits hung in one run of three. The package
  now sets `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` at import (before NumPy loads; your own value wins), the
  Dockerfile sets them too, and the BLAS controller / Numba kernel are warmed up before the first request
  (`grn_sim.inference._native`). Fits are also about 3x faster on one thread.
- **Stale results drawn over newer ones.** A slow earlier run, archive selection or benchmark load no longer overwrites what
  the user asked for later (newest request wins), and its busy button is released only when the last request ends.
- `/api/infer` with an empty or unknown algorithm list no longer wipes the run's stored results (400); `/api/simulate`
  rejects unknown topology / engine / preset names; the Studio no longer starts a run with a blank topology; the real-data
  banner follows the run that was just made; the archive detail menu used the retired key `ridge`.

### Accessibility
- axe-core (WCAG 2.0-2.2 A/AA and best-practice rules) reports no violations on any tab in the light, dark and phone layouts,
  down from 1,562 nodes: text colours now use darker "ink" tokens on the light theme (the accent colours remain for fills),
  the dark-theme indigo fill holds 4.5:1 under white text, targets are at least 24 px (40 px on touch screens), scrollable
  regions can be focused, the tab bar is a real `tablist` with roving focus and arrow / Home / End keys, and there is a skip link
  and a visible focus ring.
- Every chart has a text alternative (a summary plus its numbers for screen readers); clickable rows and badges work with
  Enter / Space; tables have names and header scopes; the theme button names the action it will take; reduced-motion is respected.
- Not covered: the network graphs' nodes are focusable but edges are not keyboard-operable, chart hover tooltips have no keyboard
  equivalent (the numbers are in the text alternative), and no screen reader was used for testing.

### Added
- **Algorithm registry** (`grn_sim.inference.registry`): one `register(AlgorithmSpec(...))` call makes an algorithm known to
  the API, the job queue, the studio, the benchmark studio, the Virtual Cell Challenge, the parameter sweep, scoring
  orientation and the page (`/api/algorithms`).
- **Score Vote** (`score_vote`): S2G + G2S (+ NIR) combined through per-edge evidence scores instead of a presence/sign vote.
- A loading bar with elapsed seconds for slow requests; F1 columns in the run, archive and Virtual Cell Challenge tables;
  F1 charts for all algorithms across the size, connectivity and noise sweeps; the runtime chart lists every algorithm.
- S2G, GENIE3, CausalGRN, NIR and RidgeCV run per-gene work on a process pool (identical results, 5-17x faster);
  CausalGRN's partial-correlation loop is vectorised; SFINN is now a deterministic scoring rule (the earlier random,
  untrained network scored at chance level).

### Added
- **Background job queue** (`grn_sim.web.jobs`, tab 6 "Job Queue"): queue network generation and the inference
  algorithms as jobs that run one at a time while the studio stays usable. A job generates a network, runs
  algorithms on one (its own, a queued job's - "generate now, infer afterwards" - or an in-memory run), or both;
  batches (`count`, seeds rise by one), live progress per algorithm, cancel, logs, toasts and a badge on the tab.
  The studio's open network is never replaced by a job. API: `GET/POST /api/jobs`, `GET /api/jobs/<id>`,
  `POST /api/jobs/<id>/cancel`, `POST /api/jobs/clear`. While a job holds the engine, interactive simulate / infer
  answer `409` with the job's name instead of hanging. The last ten finished jobs are kept in `jobs.json` beside the
  archive and survive a restart (jobs still queued or running then come back as interrupted); the runs they create are
  archived.
- Archive filters: "All runs, newest first" and "Modular topology".
- **Perturb-seq screens** (`grn_sim.perturbseq`, see `docs/PERTURBSEQ.md`): knockdown / knockout / double
  perturbation screens with ground truth, per-cell doses and escapers, network-propagated cell noise (LNA or SDE),
  droplet measurement artefacts and guide calling, realism statistics against genome-scale Perturb-seq, anchored
  kinetics and the `replogle_like` preset; `grn-sim perturbseq` CLI; studio panel with h5ad export and realism report
  (`/api/perturbseq/realism`); `modular` topology; precise steady-state solver; large-network studio mode.
- **Cross-context transfer predictors** for the Virtual Cell Challenge tab (`grn_sim.benchmarks.cross_context`): a plain source-atlas average, a calibrated AtlasShift-style ensemble, a TransPert-style consensus + global-scale predictor and a pseudo-bulk ridge regression, a third response method in the studio (`transfer_models`, `transfer_predictions`, rewiring series `calibrated_r` / `transpert_r` / `pseudobulk_r`) and a reproducible paired study (`scripts/transfer_study.py`, `docs/CROSS_CONTEXT_TRANSFER.md`). Finding: none beats the plain source average on held-out data.
- **Held-out perturbation prediction** (`grn_sim.benchmarks.hill_response`): structure -> Hill refit -> steady state for
  every algorithm, a model-matched Hill forward search, a mean-profile null, a true-structure oracle and a target-gene
  prior. Research notes: `docs/HOLDOUT_RESEARCH.md`; reproducible paired study: `scripts/holdout_study.py`.
- **CRISPRi-style knockdown generator** (`PerturbationSimulator(perturbation_mode="knockdown")`) alongside the
  default regulator-input perturbation.
- **Virtual Cell Challenge tab**: perturbation selector, held-out table for all algorithms and both response methods,
  off-target scores, Hill rewiring series; `GET /api/benchmark/vcc_atlasshift?perturbation=input|knockdown`.
- `grn_sim.paths`: one place that decides where bundled data lives and where the package writes (`GRN_SIM_HOME`,
  `RGN_DB_PATH`).
- `py.typed`, `CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, GitHub Actions workflow (lint, types, tests on Python
  3.9-3.12, wheel build and clean-environment install).
- Extras: `torch` (PyTorch backend of CellBox / CPA-Lite), `browser` (Playwright), `dev` (pytest, ruff, mypy, build, twine).

### Changed
- **The archive's "best method" is the highest F1, not the fewest total errors.** Fewest errors is won by an algorithm
  that predicts almost nothing (an empty graph has only as many errors as the network has edges): 28 archived runs
  named the by-chance baseline best and 22 named one that recovered nothing. New runs carry `best_basis: "f1"`; older
  entries are re-derived on load from their stored per-algorithm summary (the database is not rewritten); the summary
  cards name the top algorithm by mean F1. NOTEARS also no longer returns an empty graph on the linear-Jacobian and
  CellBox engines (expression spanning thousands).
- **G2S follows gen2spec.m exactly and is fast.** `GeneralToSpecific(drop_rule="exact")` (the new default) removes
  the column whose removal leaves the smallest pinv least-squares residual for every basis size, as MATLAB's
  `dropanother` does; it reproduces the stored 60-gene MATLAB run (G2S and the S2G/G2S vote) bit for bit. The previous
  LOO-score shortcut for bases wider than `fast_loo_threshold` columns stays available as `drop_rule="loo"`; it gives
  the same result on full-column-rank X but not on rank-deficient X, where candidates inside a linear dependency tie.
  The elimination now runs on O(k^2) inverse-Gram-factor downdates with certified decisions (the reference pinv
  residual is computed only for tied candidates) and spreads target genes over a process pool
  (`backend="auto"|"numpy"|"processes"|"reference"`, `n_jobs`); 500-gene networks take seconds instead of hours.
  On rank-deficient X with many dependent columns, reproducing MATLAB's tie-breaking costs hours per gene; past a
  per-gene work budget (`exact_budget_flops`, default 2e9) such steps keep the lowest-index column among those with
  the smallest error increase (same fit at that step) and are counted in `diagnostics["tie_breaks"]`
  (`diagnostics["matlab_exact"]` is False then). `exact_budget_flops=None` keeps every step MATLAB-exact.
- **The package is installable on its own.** The web page and the read-only datasets (about 2.5 MB) now ship inside
  the wheel at `grn_sim/web/static/` and `grn_sim/data/`; they used to be found by walking up from the source file,
  which only worked in a clone. The files moved from `data/` to `src/grn_sim/data/` (history is kept).
- The simulation archive, downloads and regenerated exports use the user data folder; in a clone that is still `data/`.
- Dependencies: `matplotlib` and `h5py` removed (never imported; `anndata` brings `h5py`).
- Tab titles shortened to "2. Real-World Benchmarks" and "4. Virtual Cell Challenge".
- The Docker image installs the package non-editable, no longer bakes `data/` into the image (mount it at
  `/app/data`) and runs the tests against the installed copy; `docker-compose.yml` matches how the container is run.
- Tests: those that need a running studio carry the `live_server` marker and are deselected by default; no test
  contains a machine-specific path; in-process tests use a temporary archive.
- `examples/demo.py` replaces the top-level `demo.py`.
- Lint (ruff) and type (mypy) clean; unused imports, dead variables and mutable default arguments removed.

### Fixed
- **UI/UX audit fixes.** `/api/simulate` answers `400` naming the field for a bad seed / size / noise instead of a
  `500`; seed `0` is a real seed (it became 42) and the studio reports every failed run instead of staying silent;
  clearing all algorithms no longer runs six anyway. Archived benchmark-sweep runs (about half the archive) showed
  `undefined` edge counts and had no edge list. A Perturb-seq screen of an oscillating network ran for minutes: it
  now says it has no steady state, and the steady-state solver stops relaxing once it recognises a limit cycle
  (4x faster there). The inference-algorithm menu disables algorithms that were not run; diverged-run amplitudes
  print as `>1000%`; the statistics banner quotes the live numbers; LaTeX in labels shows as text; new tabs open at
  the top; no horizontal overflow on phones; unlabeled controls have accessible names.
- **Studio networks were shown with every edge reversed.** The simulator's adjacency is T[regulator, target], but the
  node inspector (in/out degree, activators/inhibitors), edge parameters, gene profiles (server and the page's
  `buildClientGeneProfile`) and the graph DOT/SVG/JSON read it row-wise as regulators. The Hill constants shown for an
  edge were already those of the right edge; only its ends were swapped. The E. coli SOS ground truth stays in its
  MATLAB A[target, regulator] form for scoring and is now simulated and drawn transposed
  (`benchmark_datasets.regulator_target_matrix`); `estimate_biophysical_parameters` returns Hill constants indexed
  [regulator, target] and set from the regulator's level. Runs archived before the fix are repaired when they are
  opened (marker: `true_matrix_orientation`).
- **Some inference scores compared a prediction with the transposed network.** The algorithms do not share one output
  orientation (`grn_sim.inference.orientation`, measured on networks with known topology): GENIE3, CausalGRN and
  DeepRIG return A[target, regulator] by construction, and the Gardner-style algorithms (S2G, G2S, vote, NIR, FAST,
  CellBox) do so on additive data. Predictions are now brought into the orientation of the ground truth before
  scoring in the studio, the Real-World Benchmarks tab, the parameter sweep and the Virtual Cell Challenge edge
  scores. Effect: on the DREAM4 / Norman (16 TF) / Replogle sets the Gardner algorithms were scored against the
  transpose (DREAM4 G2S 0% -> 100% of edges found); on Hill-simulated runs CausalGRN and DeepRIG move from the
  bottom of the leaderboard to the top four. The thesis algorithms on the E. coli SOS set are unchanged.
  `scripts/rescore_archive_orientation.py` rescored the archived runs (marker: `pred_orientation`; AUPR is cleared
  for transposed algorithms because runs do not store the continuous estimate), and
  `grn_sim/data/overall_statistics_sweep.json` was regenerated.
- The linear-Jacobian and CellBox neural-ODE engines simulated the transposed network (they read the adjacency as
  A[target, regulator]); they now simulate T[regulator, target] like the Hill engine.
- The real Norman 2019 ground truth (`norman_2019_real_16tf`, `norman_2019_real_105gene`) is stored as
  A[target, regulator] like the SOS set and is now treated so (`TARGET_REGULATOR_DATASETS`).
- The synthetic Norman (16 TF) and Replogle sets now follow DREAM4's sign convention (B X = -U), which the studio's
  perturbation sign handling for the additive algorithms assumes.
- Archived sweep runs could not be opened (the page ignored a response without `"status": "success"`); a stored run
  without an expression matrix raised a `NameError`; archived runs report their stored classification and note when
  the trajectories would classify differently.
- A hard-coded personal backup path in `sos_pathway.py` was removed.
- `scripts/download_data.py --all --dry-run` failed on a fresh checkout.
- CPA-Lite's test assumed PyTorch; both backends are covered now.
- The studio no longer computes the Virtual Cell Challenge benchmark on every page load; it is computed when its tab
  is first opened.

## [0.1.0]
Initial reverse-engineered release of the thesis simulator, inference algorithms and the phpIoRN web studio.
