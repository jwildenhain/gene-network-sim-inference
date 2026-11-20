rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 100 genes, 50 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.021  B 0.024  C 0.024  D 0.027)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.41 | 0.22 | 0.03 | 0.03 | 0.13 |
| inspre | imputed | 0.45 | 0.50 | 0.55 | 0.50 | 0.49 |
| inspre | sources | 0.57 | 0.65 | 0.64 | 0.58 | 0.60 |
| inspre | oracle | 0.47 | 0.52 | 0.46 | 0.36 | 0.42 |
| mean_difference | partial | 0.28 | 0.31 | 0.03 | 0.03 | 0.15 |
| mean_difference | imputed | 0.28 | 0.31 | 0.42 | 0.40 | 0.32 |
| mean_difference | sources | 0.37 | 0.40 | 0.42 | 0.40 | 0.38 |
| mean_difference | oracle | 0.33 | 0.30 | 0.37 | 0.28 | 0.31 |
| nir | partial | 0.43 | 0.40 | 0.10 | 0.06 | 0.14 |
| nir | imputed | 0.43 | 0.42 | 0.50 | 0.42 | 0.43 |
| nir | sources | 0.51 | 0.50 | 0.56 | 0.47 | 0.50 |
| nir | oracle | 0.37 | 0.38 | 0.43 | 0.31 | 0.36 |
| s2g | partial | 0.33 | 0.39 | 0.03 | 0.03 | 0.18 |
| s2g | imputed | 0.43 | 0.62 | 0.63 | 0.65 | 0.54 |
| s2g | sources | 0.56 | 0.72 | 0.59 | 0.68 | 0.59 |
| s2g | oracle | 0.56 | 0.45 | 0.62 | 0.44 | 0.48 |
| ridge_cv | partial | 0.40 | 0.39 | 0.03 | 0.03 | 0.16 |
| ridge_cv | imputed | 0.54 | 0.53 | 0.57 | 0.50 | 0.51 |
| ridge_cv | sources | 0.58 | 0.61 | 0.58 | 0.55 | 0.56 |
| ridge_cv | oracle | 0.42 | 0.38 | 0.43 | 0.39 | 0.38 |
| cellbox | partial | 0.62 | 0.51 | 0.09 | 0.06 | 0.25 |
| cellbox | imputed | 0.67 | 0.66 | 0.74 | 0.65 | 0.67 |
| cellbox | sources | 0.74 | 0.75 | 0.78 | 0.71 | 0.74 |
| cellbox | oracle | 0.65 | 0.64 | 0.65 | 0.55 | 0.61 |

### 100 genes, 50 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.019  B 0.026  C 0.026  D 0.025)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.51 | 0.42 | 0.03 | 0.03 | 0.22 |
| inspre | imputed | 0.46 | 0.43 | 0.36 | 0.28 | 0.35 |
| inspre | sources | 0.31 | 0.40 | 0.36 | 0.29 | 0.32 |
| inspre | oracle | 0.42 | 0.36 | 0.44 | 0.41 | 0.39 |
| mean_difference | partial | 0.49 | 0.44 | 0.03 | 0.03 | 0.23 |
| mean_difference | imputed | 0.49 | 0.44 | 0.25 | 0.22 | 0.28 |
| mean_difference | sources | 0.22 | 0.26 | 0.25 | 0.22 | 0.22 |
| mean_difference | oracle | 0.45 | 0.40 | 0.41 | 0.40 | 0.40 |
| nir | partial | 0.46 | 0.39 | 0.07 | 0.06 | 0.12 |
| nir | imputed | 0.42 | 0.37 | 0.31 | 0.26 | 0.32 |
| nir | sources | 0.27 | 0.32 | 0.30 | 0.26 | 0.28 |
| nir | oracle | 0.40 | 0.31 | 0.36 | 0.34 | 0.34 |
| s2g | partial | 0.38 | 0.44 | 0.03 | 0.03 | 0.20 |
| s2g | imputed | 0.30 | 0.45 | 0.37 | 0.30 | 0.30 |
| s2g | sources | 0.31 | 0.40 | 0.32 | 0.35 | 0.32 |
| s2g | oracle | 0.36 | 0.36 | 0.35 | 0.38 | 0.34 |
| ridge_cv | partial | 0.52 | 0.25 | 0.03 | 0.03 | 0.16 |
| ridge_cv | imputed | 0.37 | 0.31 | 0.27 | 0.24 | 0.27 |
| ridge_cv | sources | 0.31 | 0.34 | 0.33 | 0.29 | 0.30 |
| ridge_cv | oracle | 0.42 | 0.37 | 0.36 | 0.33 | 0.35 |
| cellbox | partial | 0.55 | 0.43 | 0.05 | 0.03 | 0.20 |
| cellbox | imputed | 0.56 | 0.53 | 0.41 | 0.34 | 0.43 |
| cellbox | sources | 0.39 | 0.46 | 0.43 | 0.38 | 0.40 |
| cellbox | oracle | 0.52 | 0.47 | 0.55 | 0.50 | 0.50 |
