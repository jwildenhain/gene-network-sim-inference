rewire 0.0 seed 1 done
rewire 0.0 seed 2 done
rewire 0.0 seed 3 done
rewire 0.3 seed 1 done
rewire 0.3 seed 2 done
rewire 0.3 seed 3 done

### 250 genes, 125 perturbed in the target, 6 methods, rewire 0.0, 3 seeds
(random AUPR: A 0.010  B 0.010  C 0.010  D 0.009)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.39 | 0.39 | 0.01 | 0.01 | 0.16 |
| inspre | imputed | 0.39 | 0.43 | 0.43 | 0.44 | 0.41 |
| inspre | sources | 0.60 | 0.59 | 0.58 | 0.58 | 0.58 |
| inspre | oracle | 0.28 | 0.29 | 0.36 | 0.36 | 0.32 |
| mean_difference | partial | 0.42 | 0.41 | 0.01 | 0.01 | 0.20 |
| mean_difference | imputed | 0.42 | 0.41 | 0.57 | 0.58 | 0.46 |
| mean_difference | sources | 0.58 | 0.56 | 0.57 | 0.58 | 0.56 |
| mean_difference | oracle | 0.41 | 0.44 | 0.41 | 0.48 | 0.42 |
| nir | partial | 0.45 | 0.45 | 0.05 | 0.05 | 0.13 |
| nir | imputed | 0.33 | 0.41 | 0.41 | 0.41 | 0.39 |
| nir | sources | 0.44 | 0.49 | 0.50 | 0.46 | 0.47 |
| nir | oracle | 0.33 | 0.34 | 0.37 | 0.38 | 0.35 |
| s2g | partial | 0.44 | 0.51 | 0.01 | 0.01 | 0.23 |
| s2g | imputed | 0.45 | 0.61 | 0.69 | 0.62 | 0.57 |
| s2g | sources | 0.73 | 0.69 | 0.70 | 0.66 | 0.69 |
| s2g | oracle | 0.53 | 0.51 | 0.54 | 0.54 | 0.53 |
| ridge_cv | partial | 0.48 | 0.25 | 0.01 | 0.01 | 0.15 |
| ridge_cv | imputed | 0.33 | 0.35 | 0.43 | 0.48 | 0.38 |
| ridge_cv | sources | 0.66 | 0.64 | 0.62 | 0.65 | 0.63 |
| ridge_cv | oracle | 0.37 | 0.37 | 0.39 | 0.45 | 0.38 |
| cellbox | partial | 0.64 | 0.61 | 0.05 | 0.04 | 0.23 |
| cellbox | imputed | 0.72 | 0.72 | 0.74 | 0.74 | 0.72 |
| cellbox | sources | 0.86 | 0.81 | 0.80 | 0.82 | 0.82 |
| cellbox | oracle | 0.63 | 0.60 | 0.63 | 0.64 | 0.62 |

### 250 genes, 125 perturbed in the target, 6 methods, rewire 0.3, 3 seeds
(random AUPR: A 0.011  B 0.009  C 0.010  D 0.009)
| method | data | A P->P | B P->Q | C Q->P | D Q->Q | all pairs |
|---|---|---|---|---|---|---|
| inspre | partial | 0.27 | 0.39 | 0.01 | 0.01 | 0.14 |
| inspre | imputed | 0.23 | 0.24 | 0.22 | 0.25 | 0.23 |
| inspre | sources | 0.32 | 0.31 | 0.31 | 0.29 | 0.30 |
| inspre | oracle | 0.18 | 0.19 | 0.19 | 0.20 | 0.18 |
| mean_difference | partial | 0.44 | 0.41 | 0.01 | 0.01 | 0.20 |
| mean_difference | imputed | 0.44 | 0.41 | 0.31 | 0.30 | 0.33 |
| mean_difference | sources | 0.30 | 0.29 | 0.31 | 0.30 | 0.29 |
| mean_difference | oracle | 0.42 | 0.40 | 0.44 | 0.41 | 0.41 |
| nir | partial | 0.39 | 0.41 | 0.02 | 0.02 | 0.07 |
| nir | imputed | 0.21 | 0.22 | 0.21 | 0.25 | 0.22 |
| nir | sources | 0.24 | 0.25 | 0.26 | 0.23 | 0.24 |
| nir | oracle | 0.22 | 0.23 | 0.23 | 0.24 | 0.21 |
| s2g | partial | 0.31 | 0.37 | 0.01 | 0.01 | 0.16 |
| s2g | imputed | 0.25 | 0.39 | 0.33 | 0.29 | 0.29 |
| s2g | sources | 0.36 | 0.35 | 0.37 | 0.34 | 0.35 |
| s2g | oracle | 0.35 | 0.33 | 0.36 | 0.35 | 0.34 |
| ridge_cv | partial | 0.30 | 0.20 | 0.01 | 0.01 | 0.09 |
| ridge_cv | imputed | 0.21 | 0.21 | 0.18 | 0.19 | 0.19 |
| ridge_cv | sources | 0.32 | 0.33 | 0.33 | 0.33 | 0.32 |
| ridge_cv | oracle | 0.19 | 0.21 | 0.14 | 0.16 | 0.16 |
| cellbox | partial | 0.41 | 0.44 | 0.03 | 0.02 | 0.14 |
| cellbox | imputed | 0.48 | 0.47 | 0.40 | 0.38 | 0.42 |
| cellbox | sources | 0.44 | 0.41 | 0.43 | 0.44 | 0.42 |
| cellbox | oracle | 0.44 | 0.41 | 0.43 | 0.42 | 0.42 |
