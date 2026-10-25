# Held-Out Perturbation Prediction: Research Notes

Why the Virtual Cell Challenge tab reported a held-out correlation of exactly 0, what actually limits prediction of a
perturbation nobody has observed, and what the simulator itself had been getting wrong about knockdowns.
Everything below is reproducible with `scripts/holdout_study.py` (commands at the end).

## Summary

1. **The exact zero was an artifact of the linear predictor.** In the Gardner model `A X = -U` the row of `A` that
   belongs to gene *j* is only identified by the experiment that perturbs *j*. Hold that experiment out and the row is
   zero, so the predicted response is exactly zero for S2G, G2S and Vote. It says nothing about whether held-out
   prediction is possible.
2. **With the true regulators, it is possible.** Refitting Hill kinetics on the seen experiments and solving for the
   steady state predicts unseen perturbations with r = 0.475 at 1% noise and 0.888 without noise (24 networks, 4 of 16
   perturbations held out).
3. **The bottleneck is the inferred structure.** Across all 17 algorithms, the regulators they find give a Hill refit
   that is not distinguishable from a *randomly chosen* structure (`by_chance`, +0.074 ± 0.052 above the null); only
   NOTEARS is nominally above the null (+0.106 ± 0.044, one of 17 comparisons) at 1% noise. A structure search
   matched to the refit model class (Hill forward selection) is the only approach that beats the null clearly
   (+0.151 ± 0.072), but it recovers only about half of the true edges.
4. **The simulator's "perturbation" was not a knockdown.** It adds +1 to the *regulator input* of the perturbed gene;
   the gene's own transcript barely moves. A CRISPRi-like mode (production scaled by 1 - efficiency) is now available.
   In that mode the perturbed gene is ~77% of every response profile, which **inflates correlation over all genes**: a
   baseline that only knows *which* gene was hit scores r = 0.79, more than any algorithm. Off-target scoring is needed.

## 1. Why held-out was exactly zero

`vcc_benchmark.py` masks the held-out experiments (`X = U = 0`) and refits each model; the response to perturbing gene *j*
is `R u_j` with `R = ±pinv(A_est)`. For the Gardner-type algorithms the equation of gene *j* only contains information
through the experiment where *j* is perturbed, so `A_est[j, :]` is all zero after masking and `R u_j = 0`. This was
verified directly (zero row), and it is structural: no amount of regularisation tuning removes it for S2G/G2S/Vote.
Regularised models (RidgeCV, DMD-GRN) keep small positive correlations but are not meaningfully better than chance.

A second, independent problem is **model-class mismatch**. The simulator applies a perturbation to the *effective
regulator concentration* `x + u` of Hill kinetics. Even a linear regulator-input model `X = F (X + U)` with the *true*
support reaches only r ≈ 0.55 in-sample, and an attempt to express regulator-input perturbations through the additive
model (`Z = X + U`, `V = -X`) failed for S2G/G2S because a gene can then explain itself through its own term.

## 2. The mitigation: structure → Hill refit → steady state

Implemented in `src/grn_sim/benchmarks/hill_response.py`, usable with the output of *any* algorithm:

1. **Structure.** Keep the top-q (q = 2) regulators of every target from `|A_est|`. Algorithms disagree on edge orientation,
   so both the matrix and its transpose are tried and the one whose linear regulator-input refit explains the *training*
   data better is kept; signs come from that refit.
2. **Kinetics.** Per target, `x_k = c_k Π_l h(x_l + u_l; K_l, m_l)` with the simulator's own activator/inhibitor forms;
   `log c_k` in closed form, `K ∈ [0.05, 20]`, `m ∈ [0.5, 8]` by bounded least squares on the seen experiments and the
   control only.
3. **Prediction.** Fixed point `x = g(x + u)` by damped iteration; response `x / x_ctrl - 1`.

Held-out columns are never used by any fit (a unit test replaces them by garbage and checks the predictions do not change).
`hill_forward_selection` searches the structure with the same model class: for each target, regulators (activator or
inhibitor) are added greedily while the Hill-form BIC improves by more than 2 (at most 3 regulators, 6 correlation-screened
candidates per step).

## 3. Evidence: paired study over 24 networks

Single networks are dominated by noise. With seeds 1–6 the *null* alone ranged from 0.01 to 0.40, and an early
conclusion drawn from 4–6 networks ("Hill refit doubles the null") turned out to be noise: on the same networks the null
was as large. All comparisons below are therefore **paired** (same network, same held-out genes), 24 random scale-free
networks per cell, N = 16 genes, k_av = 1.8, 15% of edges rewired, 4 perturbations held out unless stated. Entries are
held-out Pearson r averaged over the held-out perturbations; "− null" is the paired difference ± standard error and
"wins" counts networks where the predictor beats the null.

### Activation data (the simulator's default perturbation)

| Predictor | r, 1% noise | − null | wins | r, no noise | − null | wins |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| mean-profile null | +0.015 | — | — | +0.014 | — | — |
| **oracle**: true structure + Hill | **+0.475** | +0.459 ± 0.065 | 23/24 | **+0.888** | +0.874 ± 0.073 | 24/24 |
| **Hill forward search** | +0.167 | +0.151 ± 0.072 | 17/24 | +0.272 | +0.258 ± 0.069 | 20/24 |
| GENIE3 structure + Hill | +0.075 | +0.060 ± 0.065 | 17/24 | +0.191 | +0.177 ± 0.086 | 17/24 |
| NIR structure + Hill | +0.047 | +0.031 ± 0.059 | 10/24 | +0.165 | +0.151 ± 0.068 | 16/24 |
| G2S structure + Hill | +0.054 | +0.039 ± 0.034 | 10/24 | +0.079 | +0.064 ± 0.066 | 13/24 |
| Vote structure + Hill | +0.041 | +0.025 ± 0.030 | 11/24 | +0.051 | +0.037 ± 0.069 | 10/24 |

* Hill forward search edge recovery: precision 0.58 / recall 0.49 at 1% noise, 0.75 / 0.65 without noise.
* Holding out only 2 perturbations (14 of 16 seen, 1% noise): null −0.011, oracle +0.459 (+0.469 ± 0.082), forward search
  +0.153 (+0.164 ± 0.092), algorithms +0.03 … +0.07. More training experiments do not rescue the algorithms' structures.
* **Every algorithm, 1% noise, off-target r minus null** (the committed script with `--algorithms` set to all 17 keys;
  activation data → knockdown data): netdeconv +0.113 ± 0.059 → +0.036; causalgrn +0.112 ± 0.068 → +0.007;
  notears +0.106 ± 0.044 → −0.022; genie3 +0.091 ± 0.061 → −0.038; **by_chance (random structure) +0.074 ± 0.052 →
  −0.011**; cellbox +0.055 → +0.026; cpa_lite +0.059 → +0.063 ± 0.042; fast +0.053 → −0.034; arachne +0.044 → −0.057;
  nir +0.045 → +0.026; sfinn +0.038 → −0.038; deeprig +0.035 → −0.036; dmdgrn +0.031 → −0.006; g2s +0.030 → −0.008;
  vote +0.023 → −0.012; ridge_cv +0.016 → +0.012; s2g +0.009 → −0.027 (standard errors 0.03–0.07 throughout). None
  differs from the random structure by more than one standard error, so the Hill refit with an algorithm's regulators
  is no better than with random ones; the part that helps is refitting kinetics on the seen experiments.
* Off-target scoring (perturbed gene removed) changes nothing in this mode: its own transcript is only ~14% of the
  profile norm.

### CRISPRi knockdown data (`perturbation=knockdown`, efficiency 0.8)

| Predictor | all genes | − null | off-target only | − null | wins |
| :--- | :---: | :---: | :---: | :---: | :---: |
| mean-profile null | +0.006 | — | +0.047 | — | — |
| **target-gene prior** (−0.8 at the hit gene, nothing else; no model) | **+0.790** | +0.783 ± 0.056 | 0.000 | −0.047 ± 0.047 | 9/24 |
| oracle: true structure + Hill | +0.905 | +0.898 ± 0.047 | **+0.330** | +0.283 ± 0.070 | 21/24 |
| Hill forward search | +0.724 | +0.717 ± 0.055 | +0.160 | +0.113 ± 0.069 | 16/24 |
| G2S structure + Hill | +0.682 | +0.676 ± 0.053 | +0.039 | −0.008 ± 0.037 | 17/24 |
| NIR structure + Hill | +0.680 | +0.673 ± 0.052 | +0.073 | +0.026 ± 0.036 | 16/24 |
| Vote structure + Hill | +0.679 | +0.673 ± 0.045 | +0.035 | −0.012 ± 0.033 | 14/24 |
| GENIE3 structure + Hill | +0.654 | +0.648 ± 0.051 | +0.010 | −0.038 ± 0.040 | 12/24 |

Without noise (knockdown, off-target): oracle +0.489 (+0.616 ± 0.087), forward search +0.167 (+0.293 ± 0.081, 21/24),
GENIE3 +0.116 ± 0.057, NIR +0.108 ± 0.047, G2S +0.033 ± 0.045, Vote −0.013 ± 0.035 versus the null. (The null itself is
negative here, −0.126, because the mean profile of the seen perturbations anticorrelates with the held-out
off-target responses; that is why only the paired differences are interpreted.)

### What did not help

* Pooling the three source atlases or a consensus structure across algorithms as the Hill structure.
* The linear regulator-input refit as a predictor (ceiling r ≈ 0.55 even with the true support).
* A larger number of training experiments (see the 2-held-out run).
* The structures of the 17 algorithms individually (list above): only the search matched to the model class helped.

## 4. What had been overlooked in the data generation

Audit of `PerturbationSimulator` / `kinetics.py` (checked in code, 16-gene example):

| Property | Simulator (`input`, default) | CRISPRi data |
| :--- | :--- | :--- |
| What is perturbed | +1 added to the gene's regulator input (an activity boost, closer to CRISPRa/overexpression) | the gene's own transcription is repressed |
| The gene's own transcript | barely changes (10 of 16 genes move by less than 0.1; the largest, 1.07, is feedback) | falls by roughly 80% or more |
| Size | fixed +1, i.e. 0.27–2.1 × the gene's own steady state | a fraction of the gene's own level |
| Documentation | `kinetics.py` docstring says *u* is added to production; the code adds it to the regulator concentration | |
| Cells | one noisy steady state per perturbation | ~1000 single cells per perturbation, sparse counts |

Real dataset facts from Arc's description of the Virtual Cell Challenge data: CRISPRi with dual guides, more than 80%
knockdown in 83% of cells, about 1000 cells and more than 50,000 UMIs per cell, perturbations selected for sufficient
basal expression and strong repression, 300 genes split 150/50/100 into training/validation/test
([Behind the data](https://arcinstitute.org/news/behind-the-data-virtual-cell-challenge)). The 2025 challenge held out
*perturbations* within one cell line; the 2026 challenge asks for zero-shot prediction in *unseen cell contexts*
(six cell lines, CRISPRi, no challenge-specific training set,
[VCC 2026](https://arcinstitute.org/news/virtual-cell-challenge-2026)). Tab 4 benchmarks held-out perturbations inside a
rewired context, i.e. the 2025 task plus a context-shift sweep; it is not a zero-shot cross-context benchmark.

### Existing generators that model knockdowns

| Generator | Knockdown model | Notes |
| :--- | :--- | :--- |
| GeneNetWeaver (DREAM3/4/5) | knockdown halves the transcription rate constant of one gene, knockout sets it to zero ([DREAM4 description](https://bioconductor.statistik.tu-dortmund.de/packages/3.13/data/experiment/vignettes/DREAM4/inst/doc/DREAM4_InSilico_Description.pdf)) | same mechanism as the new mode, with a fixed 50% |
| GeneSPIDER2 | steady-state linear model `Y = (I − A)⁻¹ P` with a perturbation design matrix, partial to complete knockdown; counts via negative binomial and expression-dependent dropout; compared with K562/Calu-3 CRISPRi Perturb-seq, where real effects were weaker ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11409065/)) | MATLAB, GPLv3; a linear generator favours linear inference methods |
| SERGIO | Hill-type SDE, knockout by setting a regulator to zero; outlier/library-size/dropout noise ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7530147)) | Python; no partial knockdown in the text read |
| dyngen, BoolODE | reported to support perturbations (knockdown, interventions/overexpression) | not verified or run here |

None of these tools was run for this report; the design of the new mode follows GeneNetWeaver's production-rate
mechanism with an adjustable efficiency.

### The new mode and the trap it exposes

`PerturbationSimulator(perturbation_mode="knockdown", knockdown_efficiency=0.8)` multiplies the production of the
perturbed gene by `1 − e` (`production_scale` in `evaluate_derivatives` / `ODESolver`), records `U[j, j] = −e`, and
leaves the default mode untouched. `HillResponseModel(mode="knockdown")` and `hill_forward_selection(mode="knockdown")`
fit `x_k / (1 + u_k) = c_k Π h(x_l)`. The own gene then dominates each profile (77% of the norm), which is why

* r over all genes jumps for *every* predictor (0.65–0.72 for algorithm structures), yet
* a model-free baseline using only the design (`self_prior`) scores 0.79, and
* off-target correlation, the downstream regulatory response, behaves as in the activation mode: oracle ≫ search >
  algorithms ≈ null.

The web tab reports `offtarget_r`, a `self_prior` row, a paired "vs null" column, and labels a model *above null* only if
its own off-target r exceeds 0.1.

## 5. Limitations and open questions

* One topology family (scale-free, N = 16), one rewiring level, fixed efficiency 0.8 without spread; no single-cell counts
  or dropout, no cell-to-cell heterogeneity. Real CRISPRi effects are weaker than simulated ones (GeneSPIDER2 saw this).
* The Hill refit uses the simulator's own functional form; the oracle number is an upper bound that real data would not reach.
* The forward search is greedy with a BIC threshold chosen on a handful of networks, not tuned systematically.
* No real-data check yet: the Norman 2019 and Replogle 2022 benchmarks bundled with the repository are the natural next
  test of the Hill refit (activation data), a zero-shot cross-context benchmark the natural next test of AtlasShift.
* Structure quality is the lever: better structure inference for held-out prediction should be evaluated *with the paired
  design above*, not on single networks.

## 6. Reproduce

```bash
# paired study (24 networks, 16 genes, 4 held out); ~4-7 minutes each
python scripts/holdout_study.py --mode input     --noise 0.01
python scripts/holdout_study.py --mode input     --noise 0.0
python scripts/holdout_study.py --mode knockdown --noise 0.01
python scripts/holdout_study.py --mode knockdown --noise 0.0
python scripts/holdout_study.py --mode input     --noise 0.01 --held-out 2
python scripts/holdout_study.py --mode knockdown --json knockdown.json   # also writes per-network scores

# the same predictors for one network, in the web studio: tab 4, "Perturbation" and "Response Method" selectors
python -m grn_sim.web.server 8080
```

Runtime of the web benchmark with the Hill methods: about 18 s for 12 genes and about 90 s for 32 genes (the
epistasis screen is the largest single cost); results are cached per parameter set.

Tests: `tests/test_hill_response.py`, `tests/test_knockdown_mode.py`, `tests/test_holdout_study_script.py`,
`tests/test_playwright_vcc_ui.py`.
