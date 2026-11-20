rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 100 genes, 6 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.000  B 0.011  C 0.032  D 0.025)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.14 | 0.19 | 0.05 | 0.02 | 0.03 |
| inspre | imputed | 0.31 | 0.61 | 0.64 | 0.60 | 0.59 |
| inspre | sources | 0.44 | 0.60 | 0.65 | 0.61 | 0.60 |
| inspre | oracle | 0.38 | 0.50 | 0.52 | 0.42 | 0.42 |
| mean_difference | partial | 0.29 | 0.29 | 0.05 | 0.02 | 0.04 |
| mean_difference | imputed | 0.29 | 0.29 | 0.37 | 0.40 | 0.37 |
| mean_difference | sources | 0.20 | 0.33 | 0.37 | 0.40 | 0.38 |
| mean_difference | oracle | 0.24 | 0.32 | 0.37 | 0.31 | 0.31 |
| nir | partial | 0.24 | 0.16 | 0.04 | 0.03 | 0.03 |
| nir | imputed | 0.19 | 0.51 | 0.48 | 0.50 | 0.49 |
| nir | sources | 0.41 | 0.50 | 0.52 | 0.51 | 0.50 |
| nir | oracle | 0.08 | 0.47 | 0.31 | 0.36 | 0.36 |
| s2g | partial | 0.20 | 0.20 | 0.05 | 0.02 | 0.03 |
| s2g | imputed | 0.46 | 0.59 | 0.66 | 0.62 | 0.60 |
| s2g | sources | 0.53 | 0.66 | 0.67 | 0.61 | 0.59 |
| s2g | oracle | 0.50 | 0.56 | 0.52 | 0.50 | 0.48 |
| ridge_cv | partial | 0.67 | 0.17 | 0.05 | 0.02 | 0.04 |
| ridge_cv | imputed | 0.57 | 0.67 | 0.69 | 0.68 | 0.67 |
| ridge_cv | sources | 0.46 | 0.73 | 0.58 | 0.56 | 0.56 |
| ridge_cv | oracle | 0.53 | 0.62 | 0.53 | 0.36 | 0.38 |
| cellbox | partial | 0.35 | 0.37 | 0.06 | 0.03 | 0.05 |
| cellbox | imputed | 0.45 | 0.70 | 0.71 | 0.74 | 0.73 |
| cellbox | sources | 0.58 | 0.75 | 0.76 | 0.73 | 0.73 |
| cellbox | oracle | 0.46 | 0.73 | 0.62 | 0.61 | 0.61 |

### 100 genes, 6 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.000  B 0.011  C 0.034  D 0.024)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.28 | 0.32 | 0.05 | 0.03 | 0.05 |
| inspre | imputed | 0.17 | 0.43 | 0.42 | 0.33 | 0.33 |
| inspre | sources | 0.11 | 0.34 | 0.44 | 0.32 | 0.32 |
| inspre | oracle | 0.12 | 0.51 | 0.46 | 0.39 | 0.39 |
| mean_difference | partial | 0.23 | 0.42 | 0.05 | 0.03 | 0.06 |
| mean_difference | imputed | 0.23 | 0.42 | 0.27 | 0.23 | 0.23 |
| mean_difference | sources | 0.10 | 0.20 | 0.27 | 0.23 | 0.22 |
| mean_difference | oracle | 0.33 | 0.46 | 0.43 | 0.40 | 0.40 |
| nir | partial | 0.04 | 0.19 | 0.07 | 0.02 | 0.03 |
| nir | imputed | 0.21 | 0.43 | 0.30 | 0.27 | 0.27 |
| nir | sources | 0.20 | 0.32 | 0.34 | 0.28 | 0.28 |
| nir | oracle | 0.04 | 0.49 | 0.45 | 0.33 | 0.34 |
| s2g | partial | 0.07 | 0.30 | 0.05 | 0.03 | 0.04 |
| s2g | imputed | 0.08 | 0.44 | 0.51 | 0.33 | 0.33 |
| s2g | sources | 0.14 | 0.41 | 0.45 | 0.33 | 0.32 |
| s2g | oracle | 0.09 | 0.42 | 0.54 | 0.34 | 0.34 |
| ridge_cv | partial | 0.28 | 0.15 | 0.05 | 0.03 | 0.04 |
| ridge_cv | imputed | 0.05 | 0.35 | 0.39 | 0.28 | 0.28 |
| ridge_cv | sources | 0.09 | 0.40 | 0.40 | 0.30 | 0.30 |
| ridge_cv | oracle | 0.11 | 0.50 | 0.33 | 0.35 | 0.35 |
| cellbox | partial | 0.18 | 0.40 | 0.04 | 0.02 | 0.05 |
| cellbox | imputed | 0.14 | 0.48 | 0.52 | 0.40 | 0.40 |
| cellbox | sources | 0.14 | 0.40 | 0.51 | 0.40 | 0.40 |
| cellbox | oracle | 0.17 | 0.56 | 0.60 | 0.50 | 0.50 |
