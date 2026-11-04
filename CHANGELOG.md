# Changelog

All notable changes. The format follows [Keep a Changelog](https://keepachangelog.com/); the project uses semantic
versioning once it leaves 0.x.

## [Unreleased]

### Changed
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
