rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 100 genes, 75 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.025  B 0.028  C 0.021  D 0.017)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.42 | 0.36 | 0.03 | 0.03 | 0.26 |
| inspre | imputed | 0.40 | 0.45 | 0.48 | 0.48 | 0.42 |
| inspre | sources | 0.59 | 0.63 | 0.64 | 0.65 | 0.60 |
| inspre | oracle | 0.43 | 0.47 | 0.44 | 0.39 | 0.42 |
| mean_difference | partial | 0.27 | 0.37 | 0.03 | 0.03 | 0.24 |
| mean_difference | imputed | 0.27 | 0.37 | 0.39 | 0.43 | 0.30 |
| mean_difference | sources | 0.37 | 0.44 | 0.39 | 0.43 | 0.38 |
| mean_difference | oracle | 0.28 | 0.39 | 0.33 | 0.42 | 0.31 |
| nir | partial | 0.38 | 0.40 | 0.06 | 0.08 | 0.19 |
| nir | imputed | 0.38 | 0.39 | 0.35 | 0.46 | 0.37 |
| nir | sources | 0.51 | 0.55 | 0.47 | 0.50 | 0.50 |
| nir | oracle | 0.36 | 0.42 | 0.31 | 0.36 | 0.36 |
| s2g | partial | 0.46 | 0.57 | 0.03 | 0.03 | 0.39 |
| s2g | imputed | 0.47 | 0.64 | 0.63 | 0.67 | 0.52 |
| s2g | sources | 0.57 | 0.71 | 0.61 | 0.66 | 0.59 |
| s2g | oracle | 0.47 | 0.60 | 0.51 | 0.48 | 0.48 |
| ridge_cv | partial | 0.39 | 0.45 | 0.03 | 0.03 | 0.27 |
| ridge_cv | imputed | 0.33 | 0.43 | 0.41 | 0.48 | 0.34 |
| ridge_cv | sources | 0.54 | 0.67 | 0.61 | 0.59 | 0.56 |
| ridge_cv | oracle | 0.36 | 0.43 | 0.44 | 0.43 | 0.38 |
| cellbox | partial | 0.60 | 0.43 | 0.10 | 0.10 | 0.40 |
| cellbox | imputed | 0.59 | 0.66 | 0.64 | 0.60 | 0.61 |
| cellbox | sources | 0.71 | 0.78 | 0.78 | 0.76 | 0.73 |
| cellbox | oracle | 0.61 | 0.68 | 0.65 | 0.57 | 0.62 |

### 100 genes, 75 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.025  B 0.027  C 0.020  D 0.018)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.41 | 0.51 | 0.02 | 0.04 | 0.30 |
| inspre | imputed | 0.35 | 0.46 | 0.31 | 0.40 | 0.34 |
| inspre | sources | 0.31 | 0.39 | 0.34 | 0.43 | 0.32 |
| inspre | oracle | 0.37 | 0.45 | 0.46 | 0.37 | 0.39 |
| mean_difference | partial | 0.41 | 0.51 | 0.02 | 0.04 | 0.34 |
| mean_difference | imputed | 0.41 | 0.51 | 0.22 | 0.35 | 0.33 |
| mean_difference | sources | 0.21 | 0.29 | 0.22 | 0.35 | 0.22 |
| mean_difference | oracle | 0.39 | 0.49 | 0.40 | 0.47 | 0.40 |
| nir | partial | 0.39 | 0.43 | 0.04 | 0.05 | 0.16 |
| nir | imputed | 0.31 | 0.39 | 0.23 | 0.32 | 0.29 |
| nir | sources | 0.27 | 0.38 | 0.24 | 0.33 | 0.28 |
| nir | oracle | 0.32 | 0.40 | 0.37 | 0.40 | 0.34 |
| s2g | partial | 0.36 | 0.47 | 0.02 | 0.04 | 0.31 |
| s2g | imputed | 0.31 | 0.58 | 0.30 | 0.37 | 0.33 |
| s2g | sources | 0.30 | 0.41 | 0.33 | 0.45 | 0.32 |
| s2g | oracle | 0.32 | 0.51 | 0.32 | 0.45 | 0.34 |
| ridge_cv | partial | 0.38 | 0.31 | 0.02 | 0.04 | 0.25 |
| ridge_cv | imputed | 0.26 | 0.37 | 0.20 | 0.21 | 0.23 |
| ridge_cv | sources | 0.28 | 0.39 | 0.33 | 0.43 | 0.30 |
| ridge_cv | oracle | 0.35 | 0.40 | 0.36 | 0.37 | 0.35 |
| cellbox | partial | 0.53 | 0.37 | 0.05 | 0.06 | 0.31 |
| cellbox | imputed | 0.49 | 0.59 | 0.33 | 0.46 | 0.45 |
| cellbox | sources | 0.37 | 0.50 | 0.42 | 0.47 | 0.40 |
| cellbox | oracle | 0.48 | 0.57 | 0.54 | 0.43 | 0.50 |
