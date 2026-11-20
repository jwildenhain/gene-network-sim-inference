rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 250 genes, 15 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.019  B 0.011  C 0.010  D 0.010)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.22 | 0.44 | 0.01 | 0.01 | 0.04 |
| inspre | imputed | 0.22 | 0.64 | 0.59 | 0.59 | 0.59 |
| inspre | sources | 0.30 | 0.67 | 0.61 | 0.59 | 0.58 |
| inspre | oracle | 0.27 | 0.38 | 0.29 | 0.32 | 0.32 |
| mean_difference | partial | 0.28 | 0.49 | 0.01 | 0.01 | 0.04 |
| mean_difference | imputed | 0.28 | 0.49 | 0.58 | 0.56 | 0.55 |
| mean_difference | sources | 0.32 | 0.62 | 0.58 | 0.56 | 0.56 |
| mean_difference | oracle | 0.23 | 0.45 | 0.50 | 0.43 | 0.42 |
| nir | partial | 0.15 | 0.42 | 0.01 | 0.01 | 0.01 |
| nir | imputed | 0.27 | 0.53 | 0.44 | 0.48 | 0.48 |
| nir | sources | 0.27 | 0.56 | 0.47 | 0.47 | 0.47 |
| nir | oracle | 0.33 | 0.41 | 0.30 | 0.35 | 0.35 |
| s2g | partial | 0.22 | 0.44 | 0.01 | 0.01 | 0.04 |
| s2g | imputed | 0.24 | 0.55 | 0.76 | 0.67 | 0.66 |
| s2g | sources | 0.53 | 0.75 | 0.80 | 0.68 | 0.69 |
| s2g | oracle | 0.42 | 0.63 | 0.63 | 0.51 | 0.53 |
| ridge_cv | partial | 0.22 | 0.32 | 0.01 | 0.01 | 0.03 |
| ridge_cv | imputed | 0.24 | 0.57 | 0.55 | 0.61 | 0.59 |
| ridge_cv | sources | 0.54 | 0.82 | 0.59 | 0.63 | 0.63 |
| ridge_cv | oracle | 0.37 | 0.59 | 0.39 | 0.38 | 0.38 |
| cellbox | partial | 0.21 | 0.52 | 0.03 | 0.01 | 0.04 |
| cellbox | imputed | 0.34 | 0.75 | 0.81 | 0.83 | 0.81 |
| cellbox | sources | 0.49 | 0.88 | 0.84 | 0.83 | 0.83 |
| cellbox | oracle | 0.50 | 0.67 | 0.60 | 0.62 | 0.62 |

### 250 genes, 15 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.014  B 0.011  C 0.010  D 0.010)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.62 | 0.39 | 0.01 | 0.01 | 0.04 |
| inspre | imputed | 0.24 | 0.39 | 0.28 | 0.30 | 0.30 |
| inspre | sources | 0.22 | 0.36 | 0.34 | 0.30 | 0.30 |
| inspre | oracle | 0.23 | 0.28 | 0.17 | 0.18 | 0.18 |
| mean_difference | partial | 0.64 | 0.47 | 0.01 | 0.01 | 0.04 |
| mean_difference | imputed | 0.64 | 0.47 | 0.31 | 0.29 | 0.30 |
| mean_difference | sources | 0.18 | 0.35 | 0.31 | 0.29 | 0.29 |
| mean_difference | oracle | 0.61 | 0.49 | 0.42 | 0.41 | 0.41 |
| nir | partial | 0.54 | 0.36 | 0.01 | 0.01 | 0.01 |
| nir | imputed | 0.29 | 0.27 | 0.23 | 0.24 | 0.24 |
| nir | sources | 0.16 | 0.31 | 0.26 | 0.24 | 0.24 |
| nir | oracle | 0.18 | 0.30 | 0.23 | 0.21 | 0.21 |
| s2g | partial | 0.33 | 0.34 | 0.01 | 0.01 | 0.03 |
| s2g | imputed | 0.25 | 0.45 | 0.37 | 0.34 | 0.34 |
| s2g | sources | 0.32 | 0.42 | 0.41 | 0.34 | 0.35 |
| s2g | oracle | 0.62 | 0.43 | 0.43 | 0.33 | 0.34 |
| ridge_cv | partial | 0.46 | 0.30 | 0.01 | 0.01 | 0.03 |
| ridge_cv | imputed | 0.42 | 0.30 | 0.29 | 0.31 | 0.30 |
| ridge_cv | sources | 0.45 | 0.44 | 0.31 | 0.32 | 0.32 |
| ridge_cv | oracle | 0.35 | 0.32 | 0.22 | 0.17 | 0.16 |
| cellbox | partial | 0.13 | 0.31 | 0.01 | 0.01 | 0.02 |
| cellbox | imputed | 0.40 | 0.50 | 0.42 | 0.41 | 0.41 |
| cellbox | sources | 0.37 | 0.47 | 0.44 | 0.42 | 0.42 |
| cellbox | oracle | 0.76 | 0.46 | 0.46 | 0.42 | 0.43 |
