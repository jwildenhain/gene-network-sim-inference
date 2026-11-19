# Held-out perturbation prediction: what the numbers say

Tables: [simulated screens](heldout_screen_benchmark.md) (three statistical generators, 100 and 200 genes, 5 seeds each) and
[Norman 2019, real data](real_data_heldout.md). Method and metrics: `grn_sim.evaluation.heldout_screen`. All figures below are for the
perturbations that moved at least one gene ("active").

1. **On the simulated screens no predictor does much.** Predicting a held-out perturbation from the others gives a correlation with
   the true response of 0.016-0.059 for the best predictor (`linear`), 0.006-0.051 for `coexpression`, and 0.00 for the training mean
   and for a random response. Sign accuracy on the moved genes is 0.51-0.58 for `linear` (0.5 is chance), and the share of the
   truly moved genes among the k largest predicted effects is 0.02-0.08 (chance is about the same).
2. **The error-based score is at best zero.** `deg_mse_skill` (1 means perfect, 0 is "predict no effect") is -0.015 to +0.014 for
   `linear` and -0.04 to -0.12 for the training mean: averaging other perturbations' responses is worse than predicting nothing,
   because in these generators a perturbation moves its own targets and little else. A response model has to know the
   structure to do better than the control, which is the point of the benchmark.
3. **The structure-using predictors rank above the structure-free ones on every generator and size** (`linear` above
   `coexpression` above the mean) but the gap shrinks with the network size and is smallest for `nonlinear_scm`.
4. **Real data behave differently.** On Norman 2019 (104 single CRISPRa perturbations, K562) the training mean reaches a correlation of
   0.48 and a sign accuracy of 0.81, far above `coexpression` (0.41) and `linear` (0.13): real perturbations share a large common
   response (a state shift, the guide / activation programme, technical effects), and a mean captures it. The statistical generators
   have no such shared component, which is why the simulated numbers in point 1 sit near zero. This is a realism gap of the generators,
   not a result about real biology: a screen simulator that is meant to rank virtual-cell models should reproduce a shared response of
   this size before its held-out scores are taken at face value.
5. **Caveats.** The simulated truth is noise-free total effects; the real-data "truth" is itself an estimate from about 290 cells, so the
   real-data scores are lower bounds and only comparable between predictors. The five predictors are baselines, not published methods
   (no GEARS, scGPT or CPA), and no mechanistic (Hill) simulator screens are included: this answers "how hard is held-out prediction on
   the statistical generators", not "which virtual cell model is best".
