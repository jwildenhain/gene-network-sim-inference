rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 100 genes, 20 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.029  B 0.030  C 0.024  D 0.022)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.57 | 0.28 | 0.03 | 0.03 | 0.09 |
| inspre | imputed | 0.67 | 0.50 | 0.58 | 0.55 | 0.54 |
| inspre | sources | 0.83 | 0.63 | 0.64 | 0.58 | 0.60 |
| inspre | oracle | 0.51 | 0.43 | 0.43 | 0.45 | 0.42 |
| mean_difference | partial | 0.45 | 0.28 | 0.03 | 0.03 | 0.09 |
| mean_difference | imputed | 0.45 | 0.28 | 0.56 | 0.39 | 0.34 |
| mean_difference | sources | 0.56 | 0.35 | 0.56 | 0.39 | 0.38 |
| mean_difference | oracle | 0.47 | 0.30 | 0.38 | 0.32 | 0.31 |
| nir | partial | 0.46 | 0.34 | 0.06 | 0.05 | 0.07 |
| nir | imputed | 0.39 | 0.43 | 0.55 | 0.47 | 0.46 |
| nir | sources | 0.55 | 0.55 | 0.56 | 0.49 | 0.50 |
| nir | oracle | 0.26 | 0.35 | 0.40 | 0.39 | 0.36 |
| s2g | partial | 0.52 | 0.38 | 0.03 | 0.03 | 0.12 |
| s2g | imputed | 0.58 | 0.57 | 0.74 | 0.56 | 0.55 |
| s2g | sources | 0.81 | 0.59 | 0.82 | 0.56 | 0.59 |
| s2g | oracle | 0.79 | 0.53 | 0.65 | 0.45 | 0.48 |
| ridge_cv | partial | 0.55 | 0.23 | 0.03 | 0.03 | 0.08 |
| ridge_cv | imputed | 0.59 | 0.42 | 0.54 | 0.47 | 0.43 |
| ridge_cv | sources | 0.82 | 0.58 | 0.70 | 0.55 | 0.56 |
| ridge_cv | oracle | 0.56 | 0.38 | 0.43 | 0.39 | 0.38 |
| cellbox | partial | 0.45 | 0.47 | 0.05 | 0.05 | 0.12 |
| cellbox | imputed | 0.67 | 0.59 | 0.77 | 0.72 | 0.69 |
| cellbox | sources | 0.81 | 0.71 | 0.76 | 0.73 | 0.72 |
| cellbox | oracle | 0.69 | 0.58 | 0.64 | 0.64 | 0.62 |

### 100 genes, 20 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.032  B 0.029  C 0.024  D 0.022)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.47 | 0.41 | 0.03 | 0.03 | 0.12 |
| inspre | imputed | 0.54 | 0.42 | 0.40 | 0.32 | 0.34 |
| inspre | sources | 0.44 | 0.32 | 0.38 | 0.32 | 0.32 |
| inspre | oracle | 0.43 | 0.44 | 0.42 | 0.38 | 0.39 |
| mean_difference | partial | 0.43 | 0.40 | 0.03 | 0.03 | 0.12 |
| mean_difference | imputed | 0.43 | 0.40 | 0.35 | 0.23 | 0.26 |
| mean_difference | sources | 0.35 | 0.19 | 0.35 | 0.23 | 0.22 |
| mean_difference | oracle | 0.42 | 0.39 | 0.56 | 0.39 | 0.40 |
| nir | partial | 0.30 | 0.42 | 0.06 | 0.03 | 0.05 |
| nir | imputed | 0.37 | 0.37 | 0.38 | 0.28 | 0.30 |
| nir | sources | 0.31 | 0.30 | 0.34 | 0.26 | 0.28 |
| nir | oracle | 0.39 | 0.37 | 0.37 | 0.34 | 0.34 |
| s2g | partial | 0.40 | 0.40 | 0.03 | 0.03 | 0.12 |
| s2g | imputed | 0.49 | 0.42 | 0.44 | 0.29 | 0.32 |
| s2g | sources | 0.45 | 0.30 | 0.50 | 0.31 | 0.32 |
| s2g | oracle | 0.56 | 0.35 | 0.64 | 0.29 | 0.34 |
| ridge_cv | partial | 0.51 | 0.23 | 0.03 | 0.03 | 0.08 |
| ridge_cv | imputed | 0.49 | 0.31 | 0.39 | 0.26 | 0.27 |
| ridge_cv | sources | 0.47 | 0.28 | 0.44 | 0.30 | 0.30 |
| ridge_cv | oracle | 0.56 | 0.43 | 0.43 | 0.31 | 0.35 |
| cellbox | partial | 0.44 | 0.43 | 0.02 | 0.03 | 0.10 |
| cellbox | imputed | 0.57 | 0.48 | 0.43 | 0.39 | 0.41 |
| cellbox | sources | 0.43 | 0.39 | 0.48 | 0.40 | 0.40 |
| cellbox | oracle | 0.61 | 0.53 | 0.59 | 0.48 | 0.51 |
