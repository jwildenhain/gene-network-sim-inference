# Virtual Cell screen benchmark: findings (2026-11-10)

From the 1000-simulation corpus (`vcc_screen_benchmark.md`, data in `src/grn_sim/data/benchmarks/vcc_screen_benchmark.jsonl`).
All numbers are mean direct-edge AUPR over 10 seeds (chance about 0.025 at 100 genes). The pooled tables of the report mix
factors (for example 100 double knockouts only occur at 200 genes), so the statements below use matched settings: modular
graph, CRISPRi, medium noise, all genes perturbed, unless the line says otherwise; "all generators" pools the five generators
with 10 seeds each.

## Which algorithms are appropriate

* 20 of the 21 algorithms (controls excluded) beat both the random ranking and the hubness-only control over the whole corpus.
* **CPA-Lite** is not better than random on four of the five generators (marginal on mechanistic data): not appropriate for
  network inference from screens.
* **GRNBoost** (tree importances on effect profiles) is only marginal on Boolean and nonlinear data and on full screens
  (0.04-0.05): it ranks hub genes rather than pairs.
* Over the whole corpus SFINN ranks first in every bootstrap draw (0.354, rank interval 1-1); CellBox, Ridge CV and S2G follow
  and cannot be told apart (0.338-0.345, ranks 2-4), then mean difference and CausalGRN (0.32, ranks 5-6). The order depends
  on the data generator (next section).

## Mechanistic versus statistical data (same graphs, same seeds)

| algorithm | mechanistic | linear SEM | NB copula | nonlinear SCM | Boolean |
|---|---|---|---|---|---|
| S2G | 0.471 | 0.314 | 0.317 | 0.242 | 0.345 |
| CellBox | 0.445 | 0.336 | 0.337 | 0.246 | 0.360 |
| Ridge CV | 0.382 | 0.346 | 0.353 | 0.269 | 0.374 |
| SFINN | 0.340 | 0.372 | 0.378 | 0.290 | 0.392 |
| mean difference | 0.332 | 0.324 | 0.331 | 0.253 | 0.353 |

(all 200 simulations per generator pooled.) S2G and CellBox gain most from mechanistic data (0.60 / 0.55 vs 0.36 / 0.40 on the
modular graph with linear-SEM data): the Hill steady state matches their model of the data. SFINN and the interventional
baselines are steadier across generators. Linear SEM and NB copula rank the algorithms almost identically (same mean model;
the noise structure barely matters); the nonlinear SCM is the hardest statistical generator. On the uncalibrated thesis
kinetics (scale-free graph) mechanistic data are much harder for every method (0.10-0.26) than linear-SEM data on the same
graphs (0.21-0.40).

## Screen design

* **Double knockouts** help the methods that use them as extra experiments: at 200 genes Ridge CV 0.263 -> 0.382 and S2G
  0.337 -> 0.377 with 100 doubles; at 100 genes with 50 doubles the gain is small (Ridge 0.416 -> 0.440, S2G 0.395 -> 0.408).
  Methods on single knockouts only do not gain.
* **Partial screens** (50% of genes perturbed): S2G and Ridge CV lose most (0.395 / 0.416 -> 0.16-0.29), because they must
  extend to genes that were never perturbed; mean difference, inspre, CausalGRN and SFINN are unaffected or better (the
  scored rows are only the perturbed genes). Choosing *effective* genes beats choosing at random for every method.
* **CRISPRi and knockout** give the same accuracy; **activation** (thesis input / CRISPRa) halves it (S2G 0.395 -> 0.170).
* **Size**: accuracy falls from 50 to 200 genes for every method; **CellBox collapses at 200 genes** (0.41 -> 0.09) with its
  default training budget - a scaling problem to fix before using it on larger networks.
* **Noise**: low -> high noise costs 0.07-0.14 AUPR for the top methods; the order of the top group is stable.
