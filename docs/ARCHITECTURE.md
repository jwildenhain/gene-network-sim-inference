# Architecture

GRN-Sim has three jobs: **simulate** perturbation experiments on gene regulatory networks, **infer** the network back
from the data with 18 algorithms, and **evaluate** the result (against the truth, or on real benchmarks). A web studio
(`grn-sim web`) wraps all three. This page is the map; the mathematics is in [`SPECIFICATION.md`](../SPECIFICATION.md).

## Layout

```text
src/grn_sim/
├── paths.py            where bundled data lives and where the package writes (see "Files and paths")
├── core/               GeneNetwork (topology + kinetic parameters), topology generators (random, scale-free, hierarchical)
├── simulator/          Hill kinetics, ODE solver, PerturbationSimulator, alternative engines, Perturb-seq counts, response classes
├── inference/          BaseInferenceAlgorithm + 18 algorithms (S2G, G2S, NIR, FAST, Vote, CellBox, GENIE3, NOTEARS, AtlasShift, ...)
├── evaluation/         confusion matrices, coverage / FDR / power, 9-state classification (BenchmarkMetrics)
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
  in `inference/__init__.py`, add it to `benchmarks/vcc_benchmark.py` (`ALGORITHMS`, `_build_runners`) and to the
  algorithm lists in `web/server.py`, and test the contract (shapes, determinism, no use of masked columns).
- **A new benchmark dataset:** add it to `benchmarks/benchmark_datasets.py` and `provenance.py`, ship the file under
  `src/grn_sim/data/`, and add a provenance entry (source, license, processing).
- **A new endpoint or UI panel:** route in `web/server.py`, render in `static/index.html`, and cover both: a server test
  (`tests/test_web_*.py`) and a browser test (`tests/test_playwright_*.py`, in-process server from `conftest.py`).

## Known structure debt

- `web/server.py` (about 2,400 lines) and `static/index.html` (about 8,700 lines) are single files; the natural split is
  by tab (studio, real-world benchmarks, archive, Virtual Cell Challenge, statistics). Tests reach into
  `server.CURRENT_STATE` and friends, so a split needs the shared state moved to its own module first.
- The list of algorithms is spelled out in four places (server, VCC benchmark, parameter sweep, CLI); a registry
  (`key -> class, label, defaults`) would remove the duplication.
