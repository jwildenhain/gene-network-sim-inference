rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 250 genes, 50 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.009  B 0.008  C 0.009  D 0.010)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.55 | 0.23 | 0.01 | 0.01 | 0.05 |
| inspre | imputed | 0.56 | 0.45 | 0.55 | 0.56 | 0.53 |
| inspre | sources | 0.66 | 0.50 | 0.60 | 0.61 | 0.58 |
| inspre | oracle | 0.32 | 0.28 | 0.33 | 0.34 | 0.32 |
| mean_difference | partial | 0.62 | 0.36 | 0.01 | 0.01 | 0.08 |
| mean_difference | imputed | 0.62 | 0.36 | 0.62 | 0.56 | 0.52 |
| mean_difference | sources | 0.68 | 0.49 | 0.62 | 0.56 | 0.56 |
| mean_difference | oracle | 0.62 | 0.36 | 0.52 | 0.43 | 0.42 |
| nir | partial | 0.56 | 0.49 | 0.02 | 0.02 | 0.04 |
| nir | imputed | 0.56 | 0.39 | 0.47 | 0.48 | 0.46 |
| nir | sources | 0.59 | 0.37 | 0.46 | 0.50 | 0.47 |
| nir | oracle | 0.39 | 0.30 | 0.33 | 0.37 | 0.35 |
| s2g | partial | 0.39 | 0.31 | 0.01 | 0.01 | 0.06 |
| s2g | imputed | 0.33 | 0.52 | 0.70 | 0.68 | 0.62 |
| s2g | sources | 0.72 | 0.63 | 0.69 | 0.71 | 0.69 |
| s2g | oracle | 0.57 | 0.47 | 0.46 | 0.56 | 0.53 |
| ridge_cv | partial | 0.50 | 0.15 | 0.01 | 0.01 | 0.03 |
| ridge_cv | imputed | 0.49 | 0.38 | 0.60 | 0.57 | 0.52 |
| ridge_cv | sources | 0.67 | 0.56 | 0.68 | 0.66 | 0.63 |
| ridge_cv | oracle | 0.53 | 0.39 | 0.42 | 0.38 | 0.38 |
| cellbox | partial | 0.53 | 0.54 | 0.03 | 0.02 | 0.09 |
| cellbox | imputed | 0.85 | 0.64 | 0.82 | 0.82 | 0.78 |
| cellbox | sources | 0.91 | 0.78 | 0.81 | 0.84 | 0.83 |
| cellbox | oracle | 0.63 | 0.51 | 0.63 | 0.64 | 0.61 |

### 250 genes, 50 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.010  B 0.008  C 0.010  D 0.010)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.35 | 0.17 | 0.01 | 0.01 | 0.04 |
| inspre | imputed | 0.26 | 0.21 | 0.27 | 0.29 | 0.27 |
| inspre | sources | 0.41 | 0.22 | 0.33 | 0.32 | 0.30 |
| inspre | oracle | 0.22 | 0.14 | 0.19 | 0.20 | 0.18 |
| mean_difference | partial | 0.48 | 0.36 | 0.01 | 0.01 | 0.07 |
| mean_difference | imputed | 0.48 | 0.36 | 0.31 | 0.30 | 0.30 |
| mean_difference | sources | 0.40 | 0.25 | 0.31 | 0.30 | 0.29 |
| mean_difference | oracle | 0.43 | 0.36 | 0.42 | 0.42 | 0.41 |
| nir | partial | 0.39 | 0.36 | 0.01 | 0.01 | 0.02 |
| nir | imputed | 0.25 | 0.18 | 0.26 | 0.26 | 0.25 |
| nir | sources | 0.37 | 0.18 | 0.26 | 0.26 | 0.24 |
| nir | oracle | 0.18 | 0.14 | 0.24 | 0.25 | 0.21 |
| s2g | partial | 0.24 | 0.19 | 0.01 | 0.01 | 0.04 |
| s2g | imputed | 0.23 | 0.31 | 0.34 | 0.33 | 0.30 |
| s2g | sources | 0.45 | 0.30 | 0.35 | 0.36 | 0.35 |
| s2g | oracle | 0.36 | 0.30 | 0.31 | 0.37 | 0.34 |
| ridge_cv | partial | 0.32 | 0.11 | 0.01 | 0.01 | 0.03 |
| ridge_cv | imputed | 0.35 | 0.16 | 0.26 | 0.28 | 0.23 |
| ridge_cv | sources | 0.43 | 0.28 | 0.35 | 0.34 | 0.32 |
| ridge_cv | oracle | 0.24 | 0.18 | 0.18 | 0.17 | 0.16 |
| cellbox | partial | 0.38 | 0.43 | 0.01 | 0.01 | 0.05 |
| cellbox | imputed | 0.51 | 0.37 | 0.41 | 0.41 | 0.40 |
| cellbox | sources | 0.55 | 0.37 | 0.43 | 0.43 | 0.42 |
| cellbox | oracle | 0.45 | 0.35 | 0.42 | 0.46 | 0.43 |
