# Changelog

All notable changes. The format follows [Keep a Changelog](https://keepachangelog.com/); the project uses semantic
versioning once it leaves 0.x.

## [Unreleased]

### Added
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
