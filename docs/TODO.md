# To do

Open work, newest decisions first. Source for the algorithm list: `research_notes/Perturb seq design and modelling/`
(`causal_grn_inference.md`, `prediction_models_benchmarks.md`). Already implemented (18): S2G, G2S, Vote, NIR, FastSVD,
ByChance, AtlasShift, CellBox, GENIE3, NetworkDeconvolution, RidgeCV, CPA-Lite, ARACNe, CausalGRN, DMD-GRN, DeepRIG,
SFINN, NOTEARS.

## Paper algorithms not yet implemented

Ordered by fit to Perturb-seq knockout data (steady-state response matrix) and implementation cost.

| Priority | Method | Paper | Why / note |
|---|---|---|---|
| 1 | **inspre** (inverse sparse regression) | Brown, Morris, Lappalainen, Knowles; bioRxiv 2023, Nat Commun | Sparse inverse of the average-causal-effect matrix, handles cycles; fits our total-effect data directly. ADMM; seconds at scale. |
| 1 | **MeanDifference / BetterBoost** | CausalBench challenge winners (arXiv 2308.15395) | Control-vs-perturbed mean difference (+ KS p-value on a GRNBoost score). Simple, and the research notes say it is competitive with elaborate methods: needed as a baseline. |
| 2 | **LLC** (linear cyclic with latent confounders) | Hyttinen et al. 2012 | Classic linear cyclic SEM with interventions; reference for inspre. |
| 2 | **bicycle** | Rohbeck et al., CLeaR 2024 | OU steady state + Lyapunov covariance; uses single-cell covariance, so it needs the Phase 2 cell data. |
| 2 | **CLIPR / latent causal diffusions** | arXiv 2601.15341 | Linearised direct effects from perturbation responses; identifiable under linear drift. |
| 3 | **DCD-FG** (factor graphs) | Lopez et al., NeurIPS 2022 | Scales to hundreds of interventions; needs torch (already an optional extra). |
| 3 | **DOTEARS / DCDI** | Interventional DAG learners | NOTEARS family with intervention-aware likelihood; ours (NOTEARS) is observational. |
| 3 | **GIES / IGSP** | Hauser and Buehlmann 2012; Wang, Squires, Uhler | Score / constraint based interventional DAGs; slow beyond ~100 genes. |
| 3 | **SparseRC** | CausalBench challenge | Root-cause assumption on a DAG. |
| 4 | **Dictys / RENGE** | Wang et al. 2023; Ishikawa et al. 2023 | Need time-series data, which the simulator can produce but the archive does not yet store. |

Research gaps listed in the notes that need a paper read before implementation: LLC, GIES, IGSP, DCDI, NODAGS-Flow, Dictys.

## Perturb-seq proposal (`reports/GRN-Sim Perturb-seq proposal.md`)

- [ ] **Benchmark harness on single-cell screens** (proposal section 3): held-out-perturbation splits, baselines (training
  mean, additive, linear, target-only, random structure), off-target metrics (DEG-weighted MSE, Pearson-delta, direction
  accuracy, rank), stratified by true effect size. `scripts/holdout_study.py` still runs on bulk data.
- [ ] **2000-gene run through the job queue** with all 18 algorithms; G2S in its quick (budgeted tie-break) mode. Results
  go to the simulation archive.
- [ ] Realism targets still missed at 500 genes: mediated share 0.79 (target 0.90), phenotype share 0.20.

## Housekeeping

- [ ] Cut a release (everything is under `[Unreleased]`).
- [ ] Two pre-existing mypy errors (`inference/ensemble.py`, `web/server.py`): recheck.
- [ ] Real-data check of the Hill refit and the transfer predictors on Norman 2019 / Replogle 2022.
- [ ] Resume jobs interrupted by a restart (today they come back as "interrupted"; the last 10 are kept and roll off).
