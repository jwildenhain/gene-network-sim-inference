### 500 genes, 30 perturbed, 3 seeds  (random: A 0.003, B 0.0056)
| method | mode | AUPR A (P->P) | AUPR B (P->unperturbed) | balanced | scores B? | s | status |
|---|---|---|---|---|---|---|---|
| mean_difference | native | 0.410 ± 0.181 | 0.350 ± 0.187 | 0.380 ± 0.184 | yes | 0.0 | 3/3 ok |
| inspre | native | 0.531 ± 0.277 | 0.224 ± 0.144 | 0.377 ± 0.157 | yes | 0.0 | 3/3 ok |
| pearson | native | 0.173 ± 0.038 | 0.508 ± 0.244 | 0.341 ± 0.137 | yes | 0.0 | 3/3 ok |
| causalgrn | native | 0.184 ± 0.052 | 0.457 ± 0.091 | 0.320 ± 0.019 | yes | 0.4 | 3/3 ok |
| ridge_cv | adapted | 0.364 ± 0.121 | 0.198 ± 0.073 | 0.281 ± 0.043 | yes | 20.0 | 3/3 ok |
| nir | native | 0.265 ± 0.062 | 0.292 ± 0.071 | 0.278 ± 0.058 | yes | 1.5 | 3/3 ok |
| arachne | native | 0.012 ± 0.002 | 0.453 ± 0.237 | 0.232 ± 0.119 | yes | 3.0 | 3/3 ok |
| g2s | adapted | 0.263 ± 0.291 | 0.194 ± 0.133 | 0.228 ± 0.211 | yes | 2.6 | 3/3 ok |
| nir | adapted | 0.244 ± 0.127 | 0.208 ± 0.167 | 0.226 ± 0.105 | yes | 0.1 | 3/3 ok |
| dmdgrn | native | 0.023 ± 0.011 | 0.429 ± 0.145 | 0.226 ± 0.072 | yes | 0.1 | 3/3 ok |
| score_vote | adapted | 0.295 ± 0.278 | 0.143 ± 0.138 | 0.219 ± 0.208 | yes | 0.5 | 3/3 ok |
| vote | adapted | 0.265 ± 0.289 | 0.140 ± 0.137 | 0.203 ± 0.213 | yes | 0.2 | 3/3 ok |
| s2g | adapted | 0.218 ± 0.208 | 0.140 ± 0.107 | 0.179 ± 0.156 | yes | 1.6 | 3/3 ok |
| fast | native | 0.303 ± 0.274 | 0.006 ± 0.002 | 0.154 ± 0.138 | no | 0.1 | 3/3 ok |
| ridge_cv | native | 0.260 ± 0.171 | 0.006 ± 0.002 | 0.133 ± 0.086 | no | 5.2 | 3/3 ok |
| genie3 | native | 0.007 ± 0.002 | 0.235 ± 0.104 | 0.121 ± 0.051 | yes | 4.1 | 3/3 ok |
| deeprig | native | 0.022 ± 0.021 | 0.164 ± 0.113 | 0.093 ± 0.061 | yes | 1.0 | 3/3 ok |
| cpa_lite | native | 0.082 ± 0.066 | 0.093 ± 0.052 | 0.088 ± 0.058 | yes | 0.0 | 3/3 ok |
| netdeconv | adapted | 0.142 ± 0.143 | 0.007 ± 0.001 | 0.074 ± 0.072 | yes | 0.0 | 3/3 ok |
| notears | native | 0.026 ± 0.021 | 0.095 ± 0.044 | 0.061 ± 0.015 | yes | 9.3 | 3/3 ok |
| fast | adapted | 0.072 ± 0.046 | 0.045 ± 0.046 | 0.058 ± 0.046 | yes | 0.0 | 3/3 ok |
| gene_features | native | 0.042 ± 0.030 | 0.018 ± 0.009 | 0.030 ± 0.014 | yes | 0.0 | 3/3 ok |
| dmdgrn | adapted | 0.035 ± 0.039 | 0.006 ± 0.001 | 0.021 ± 0.019 | yes | 0.0 | 3/3 ok |
| grnboost | native | 0.009 ± 0.006 | 0.030 ± 0.002 | 0.019 ± 0.004 | yes | 7.1 | 3/3 ok |
| cellbox | adapted | 0.012 ± 0.002 | 0.006 ± 0.002 | 0.009 ± 0.002 | no | 1.8 | 3/3 ok |
| sfinn | native | 0.012 ± 0.002 | 0.006 ± 0.002 | 0.009 ± 0.002 | no | 0.0 | 3/3 ok |
| by_chance | native | 0.011 ± 0.002 | 0.006 ± 0.002 | 0.008 ± 0.001 | yes | 0.0 | 3/3 ok |
| s2g | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| g2s | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| score_vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| cellbox | native | - | - | - | - | - | error: MemoryError((250500, 250500), dtype('float64')) |
| netdeconv | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |

### 500 genes, 100 perturbed, 3 seeds  (random: A 0.004, B 0.0050)
| method | mode | AUPR A (P->P) | AUPR B (P->unperturbed) | balanced | scores B? | s | status |
|---|---|---|---|---|---|---|---|
| nir | native | 0.366 ± 0.088 | 0.327 ± 0.093 | 0.347 ± 0.083 | yes | 2.9 | 3/3 ok |
| mean_difference | native | 0.334 ± 0.179 | 0.303 ± 0.190 | 0.319 ± 0.177 | yes | 0.0 | 3/3 ok |
| causalgrn | native | 0.182 ± 0.013 | 0.453 ± 0.181 | 0.317 ± 0.093 | yes | 0.5 | 3/3 ok |
| dmdgrn | native | 0.131 ± 0.016 | 0.490 ± 0.057 | 0.310 ± 0.035 | yes | 0.1 | 3/3 ok |
| inspre | native | 0.278 ± 0.127 | 0.283 ± 0.177 | 0.280 ± 0.145 | yes | 0.1 | 3/3 ok |
| pearson | native | 0.170 ± 0.060 | 0.351 ± 0.211 | 0.260 ± 0.133 | yes | 0.0 | 3/3 ok |
| s2g | adapted | 0.267 ± 0.143 | 0.242 ± 0.116 | 0.255 ± 0.129 | yes | 1.1 | 3/3 ok |
| arachne | native | 0.131 ± 0.067 | 0.372 ± 0.183 | 0.251 ± 0.124 | yes | 2.7 | 3/3 ok |
| nir | adapted | 0.221 ± 0.164 | 0.241 ± 0.151 | 0.231 ± 0.142 | yes | 0.3 | 3/3 ok |
| ridge_cv | native | 0.426 ± 0.068 | 0.005 ± 0.001 | 0.215 ± 0.034 | no | 6.9 | 3/3 ok |
| ridge_cv | adapted | 0.242 ± 0.224 | 0.170 ± 0.050 | 0.206 ± 0.113 | yes | 19.1 | 3/3 ok |
| score_vote | adapted | 0.183 ± 0.135 | 0.178 ± 0.190 | 0.181 ± 0.163 | yes | 0.9 | 3/3 ok |
| genie3 | native | 0.036 ± 0.007 | 0.309 ± 0.175 | 0.173 ± 0.091 | yes | 5.4 | 3/3 ok |
| vote | adapted | 0.111 ± 0.100 | 0.183 ± 0.181 | 0.147 ± 0.140 | yes | 0.6 | 3/3 ok |
| fast | native | 0.261 ± 0.050 | 0.005 ± 0.001 | 0.133 ± 0.025 | no | 0.1 | 3/3 ok |
| g2s | adapted | 0.077 ± 0.078 | 0.144 ± 0.077 | 0.110 ± 0.071 | yes | 32.0 | 3/3 ok |
| netdeconv | adapted | 0.144 ± 0.085 | 0.009 ± 0.005 | 0.077 ± 0.043 | yes | 0.0 | 3/3 ok |
| deeprig | native | 0.043 ± 0.023 | 0.100 ± 0.070 | 0.071 ± 0.046 | yes | 1.1 | 3/3 ok |
| grnboost | native | 0.036 ± 0.013 | 0.069 ± 0.028 | 0.052 ± 0.007 | yes | 12.2 | 3/3 ok |
| fast | adapted | 0.061 ± 0.043 | 0.012 ± 0.002 | 0.036 ± 0.021 | yes | 0.0 | 3/3 ok |
| dmdgrn | adapted | 0.063 ± 0.021 | 0.009 ± 0.003 | 0.036 ± 0.012 | yes | 0.0 | 3/3 ok |
| cpa_lite | native | 0.045 ± 0.032 | 0.027 ± 0.018 | 0.036 ± 0.025 | yes | 0.0 | 3/3 ok |
| notears | native | 0.013 ± 0.012 | 0.007 ± 0.003 | 0.010 ± 0.006 | yes | 1.2 | 3/3 ok |
| gene_features | native | 0.010 ± 0.003 | 0.009 ± 0.002 | 0.009 ± 0.002 | yes | 0.0 | 3/3 ok |
| by_chance | native | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.000 | yes | 0.0 | 3/3 ok |
| cellbox | adapted | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.000 | no | 95.1 | 3/3 ok |
| sfinn | native | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.000 | no | 0.0 | 3/3 ok |
| s2g | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| g2s | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| score_vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| cellbox | native | - | - | - | - | - | error: MemoryError((250500, 250500), dtype('float64')) |
| netdeconv | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
