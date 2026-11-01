# Changelog

All notable changes. The format follows [Keep a Changelog](https://keepachangelog.com/); the project uses semantic
versioning once it leaves 0.x.

## [Unreleased]

### Added
- **Background job queue** (`grn_sim.web.jobs`, tab 6 "Job Queue"): queue network generation and the inference
  algorithms as jobs that run one at a time while the studio stays usable. A job generates a network, runs
  algorithms on one (its own, a queued job's - "generate now, infer afterwards" - or an in-memory run), or both;
  batches (`count`, seeds rise by one), live progress per algorithm, cancel, logs, toasts and a badge on the tab.
  The studio's open network is never replaced by a job. API: `GET/POST /api/jobs`, `GET /api/jobs/<id>`,
  `POST /api/jobs/<id>/cancel`, `POST /api/jobs/clear`. While a job holds the engine, interactive simulate / infer
  answer `409` with the job's name instead of hanging. Jobs live in memory; the runs they create are archived.
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
