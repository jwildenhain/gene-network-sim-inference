# Architecture

GRN-Sim has three jobs: **simulate** perturbation experiments on gene regulatory networks, **infer** the network back
from the data with 24 registered algorithms ([`docs/algorithms/`](algorithms/README.md)), and **evaluate** the result (against the truth, or on real benchmarks). A web studio
(`grn-sim web`) wraps all three. This page is the map; the mathematics is in [`SPECIFICATION.md`](../SPECIFICATION.md) and every simulation option is listed in [`SIMULATION_OPTIONS.md`](SIMULATION_OPTIONS.md).

## Layout

```text
src/grn_sim/
├── paths.py            where bundled data lives and where the package writes (see "Files and paths")
├── core/               GeneNetwork (topology + kinetic parameters), topology generators (random, scale-free, hierarchical)
├── simulator/          Hill kinetics, ODE solver, PerturbationSimulator, alternative engines, Perturb-seq counts, response classes
├── inference/          BaseInferenceAlgorithm + 24 registered algorithms (registry.py: one `register` call each), the partial-screen
│                       adapter (partial.py), inspre, baselines, and the experimental bicycle-lite / DOTEARS-lite / consensus;
│                       descriptions.py holds the one-paragraph description of each (docs/algorithms/ is generated from it)
├── perturbseq/         Perturb-seq generator: conditions, cells, measurement, realism, calibration; design.py (full or partial
│                       screens, effective-gene selection), partial_stats.py (their statistics), ace.py (effect estimates + FDR)
├── evaluation/         confusion matrices, coverage / FDR / power, 9-state classification (BenchmarkMetrics); screen_benchmark.py
│                       (direct vs total-effect truth), partial_screen.py (per-block scoring), prior_experiment.py, heldout_prediction.py;
│                       vcc2026.py (the Virtual Cell Challenge 2026 scorer, a port of cell-eval2's vcc2026 profile) and
│                       vcc_prediction.py (predicted cells, the structure + linear-refit predictor, scoring of a prediction set)
├── benchmarks/         SOS pathway, dataset registry + provenance, Virtual Cell Challenge benchmark, held-out prediction, sweeps
├── visualization/      graph export (DOT / SVG / JSON) and the built-in layered layout used without Graphviz
├── agents/             small multi-agent simulate -> infer -> evaluate loop (CLI: grn-sim multi-agent)
├── web/                HTTP server, SQLite archive, and the single-page studio (static/index.html)
├── data/               read-only datasets shipped in the wheel (MATLAB thesis archives, SOS data, Norman benchmarks)
└── cli.py              `grn-sim` entry point: sos-benchmark, multi-agent, web
```
Dependencies point downwards: `core` knows nothing else, `simulator` uses `core`, `inference` uses neither, `evaluation`
uses numpy only, `benchmarks` and `web` use everything.

## The data flow in five lines

```python
net = GeneNetwork.create(n=12, k_av=1.8, net_type=TopologyType.POWER_LAW, seed=1)   # truth: net.topology.adjacency
exp = PerturbationSimulator(perturbation_factor=2.0, measurement_noise=0.01).run(net, seed=1)   # exp.X, exp.U, exp.true_matrix
result = SimpleToGeneral().fit(exp.X, exp.U)                                        # result.A_est, result.adjacency_pred
metrics = evaluate_inference(exp.true_matrix, result.adjacency_pred)                # coverage, FDR, power, F1, ...
```

### Conventions that are easy to get wrong
- `true_matrix[regulator, target]` is in {-1, 0, +1}. `adjacency_pred` uses the same convention.
- `X[i, j]` is the relative response of gene *i* when gene *j* is perturbed, `x / x_reference - 1`; `U[i, j]` is the
  perturbation applied to gene *i* in experiment *j* (one gene per experiment, so both are square).
- The Gardner-style algorithms solve `A X = -U`. A few expect `-U` as input (`FIT_INPUT_SIGN` in
  `benchmarks/vcc_benchmark.py`: NIR, FAST, RidgeCV). The simulator's default perturbation is added to the *regulator
  input* of the Hill kinetics, `x + u` (`perturbation_mode="regulator"` in the algorithms fits that model class);
  `PerturbationSimulator(perturbation_mode="knockdown")` instead scales the perturbed gene's production by `1 - e`.
- Held-out prediction: the linear response `-pinv(A)` is exactly zero for a gene whose experiment was held out; the Hill
  refit in `benchmarks/hill_response.py` does not have that limitation (`docs/HOLDOUT_RESEARCH.md`).

## Files and paths

`grn_sim.paths` is the only place that decides where files are:

| Kind | Function | Where |
|---|---|---|
| read-only datasets | `bundled_path()`, `resolve_data_file()` | `grn_sim/data/` inside the package |
| everything written | `user_data_dir()` | `$GRN_SIM_HOME`, else `<clone>/data`, else the per-user data folder |
| simulation archive | `archive_db_path()` | `$RGN_DB_PATH`, else `<user data>/rgn_db.sqlite` |
| raw downloads / exports | `downloads_dir()`, `exports_dir()` | under the user data folder |

A user's copy of a data file takes precedence over the bundled one, so a rebuilt benchmark overrides the shipped one.

## The web studio

`web/server.py` is a standard-library `ThreadingMixIn` HTTP server (no framework, no authentication, loopback by
default). `GRNWebHandler.do_GET` / `do_POST` route the `/api/...` endpoints; `static/index.html` is the whole front end
(vanilla JS, canvas charts, no build step). State lives in three places:

- `CURRENT_STATE`: the run shown in the studio, guarded by `STATE_LOCK`;
- `SIMULATION_RUN_STORE`: recent runs kept in memory;
- the SQLite archive (`web/database.py`, tables `nw` and `alg`), which survives restarts.

Expensive results (benchmark studios, the Virtual Cell Challenge benchmark) are cached per parameter set in
`BENCHMARK_CACHE`. Responses are serialised before headers are sent, so a failure is a clean HTTP 500.

## Extending

- **A new inference algorithm:** subclass `BaseInferenceAlgorithm`, return an `InferenceResult` from `fit(X, U)`, export it
  in `inference/__init__.py`, and add one `register(AlgorithmSpec(...))` in `inference/registry.py` (set its output orientation
  and its `partial` capability: `native`, or `adapter` when it needs `PartialAdapter`). Add its description to
  `inference/descriptions.py` and run `python scripts/generate_algorithm_docs.py` (`tests/test_algorithm_docs.py` fails until
  both exist). The studio, job queue, sweep and scoring read the registry; test the contract (shapes, determinism, orientation).
- **A new screen design:** extend `perturbseq/design.py` (`ScreenDesign`, `select_perturbed`), give it statistics in
  `perturbseq/partial_stats.py`-style categories, a scoring function in `evaluation/`, and its options in the CLI, the server's
  `_parse_perturb_seq_extras` and the Perturb-seq panel.
- **A new benchmark dataset:** add it to `benchmarks/benchmark_datasets.py` and `provenance.py`, ship the file under
  `src/grn_sim/data/`, and add a provenance entry (source, license, processing).
- **A new endpoint or UI panel:** route in `web/server.py`, render in `static/index.html`, and cover both: a server test
  (`tests/test_web_*.py`) and a browser test (`tests/test_playwright_*.py`, in-process server from `conftest.py`).

## Known structure debt

- `web/server.py` (about 2,400 lines) and `static/index.html` (about 8,700 lines) are single files; the natural split is
  by tab (studio, real-world benchmarks, archive, Virtual Cell Challenge, statistics). Tests reach into
  `server.CURRENT_STATE` and friends, so a split needs the shared state moved to its own module first.
- `benchmarks/vcc_benchmark.py` still keeps its own display order and labels next to the registry (`_BUILTIN_ALGORITHMS`,
  `EXTRA_LABELS`); the registry is the source of truth for everything else.
