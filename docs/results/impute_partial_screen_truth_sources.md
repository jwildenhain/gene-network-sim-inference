rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 150 genes, 30 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.009  B 0.009  C 0.018  D 0.017)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.37 | 0.37 | 0.02 | 0.02 | 0.08 |
| inspre | imputed | 0.41 | 0.51 | 0.78 | 0.79 | 0.73 |
| inspre | sources | 0.88 | 0.82 | 0.84 | 0.83 | 0.83 |
| inspre | oracle | 0.35 | 0.27 | 0.32 | 0.29 | 0.28 |
| mean_difference | partial | 0.36 | 0.40 | 0.02 | 0.02 | 0.08 |
| mean_difference | imputed | 0.36 | 0.40 | 0.57 | 0.60 | 0.55 |
| mean_difference | sources | 0.63 | 0.59 | 0.57 | 0.60 | 0.58 |
| mean_difference | oracle | 0.35 | 0.34 | 0.36 | 0.37 | 0.34 |
| nir | partial | 0.35 | 0.35 | 0.02 | 0.02 | 0.04 |
| nir | imputed | 0.42 | 0.45 | 0.66 | 0.61 | 0.58 |
| nir | sources | 0.63 | 0.62 | 0.63 | 0.66 | 0.65 |
| nir | oracle | 0.35 | 0.30 | 0.41 | 0.34 | 0.34 |
| s2g | partial | 0.28 | 0.36 | 0.02 | 0.02 | 0.08 |
| s2g | imputed | 0.39 | 0.63 | 0.76 | 0.83 | 0.73 |
| s2g | sources | 0.86 | 0.82 | 0.86 | 0.85 | 0.85 |
| s2g | oracle | 0.44 | 0.43 | 0.47 | 0.46 | 0.45 |
| ridge_cv | partial | 0.42 | 0.20 | 0.02 | 0.02 | 0.05 |
| ridge_cv | imputed | 0.45 | 0.55 | 0.68 | 0.70 | 0.63 |
| ridge_cv | sources | 0.85 | 0.80 | 0.84 | 0.81 | 0.81 |
| ridge_cv | oracle | 0.48 | 0.52 | 0.41 | 0.39 | 0.41 |
| cellbox | partial | 0.52 | 0.45 | 0.03 | 0.03 | 0.10 |
| cellbox | imputed | 0.41 | 0.42 | 0.61 | 0.78 | 0.68 |
| cellbox | sources | 0.90 | 0.78 | 0.78 | 0.80 | 0.80 |
| cellbox | oracle | 0.44 | 0.25 | 0.31 | 0.26 | 0.26 |

### 150 genes, 30 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.009  B 0.009  C 0.017  D 0.017)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.22 | 0.25 | 0.02 | 0.02 | 0.05 |
| inspre | imputed | 0.19 | 0.33 | 0.36 | 0.40 | 0.37 |
| inspre | sources | 0.47 | 0.45 | 0.42 | 0.42 | 0.42 |
| inspre | oracle | 0.18 | 0.14 | 0.19 | 0.21 | 0.17 |
| mean_difference | partial | 0.18 | 0.31 | 0.02 | 0.02 | 0.06 |
| mean_difference | imputed | 0.18 | 0.31 | 0.29 | 0.31 | 0.29 |
| mean_difference | sources | 0.37 | 0.33 | 0.29 | 0.31 | 0.30 |
| mean_difference | oracle | 0.25 | 0.29 | 0.29 | 0.33 | 0.30 |
| nir | partial | 0.20 | 0.28 | 0.02 | 0.02 | 0.03 |
| nir | imputed | 0.17 | 0.26 | 0.30 | 0.30 | 0.29 |
| nir | sources | 0.39 | 0.35 | 0.32 | 0.32 | 0.32 |
| nir | oracle | 0.28 | 0.17 | 0.25 | 0.23 | 0.21 |
| s2g | partial | 0.10 | 0.14 | 0.02 | 0.02 | 0.04 |
| s2g | imputed | 0.06 | 0.40 | 0.35 | 0.41 | 0.33 |
| s2g | sources | 0.47 | 0.45 | 0.44 | 0.43 | 0.43 |
| s2g | oracle | 0.24 | 0.32 | 0.22 | 0.34 | 0.29 |
| ridge_cv | partial | 0.12 | 0.17 | 0.02 | 0.02 | 0.03 |
| ridge_cv | imputed | 0.07 | 0.13 | 0.27 | 0.31 | 0.21 |
| ridge_cv | sources | 0.47 | 0.45 | 0.43 | 0.42 | 0.42 |
| ridge_cv | oracle | 0.15 | 0.16 | 0.17 | 0.16 | 0.15 |
| cellbox | partial | 0.29 | 0.36 | 0.02 | 0.02 | 0.06 |
| cellbox | imputed | 0.17 | 0.24 | 0.32 | 0.39 | 0.35 |
| cellbox | sources | 0.49 | 0.42 | 0.38 | 0.40 | 0.40 |
| cellbox | oracle | 0.30 | 0.15 | 0.25 | 0.21 | 0.19 |
