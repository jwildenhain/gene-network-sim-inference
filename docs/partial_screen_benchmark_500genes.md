### 500 genes, 30 perturbed, 3 seeds  (random: A 0.003, B 0.0056)
| method | mode | AUPR A (P->P) | AUPR B (P->unperturbed) | balanced | scores B? | s | status |
|---|---|---|---|---|---|---|---|
| cellbox | native | 0.517 ± 0.175 | 0.249 ± 0.152 | 0.383 ± 0.163 | yes | 1.2 | 3/3 ok |
| mean_difference | native | 0.410 ± 0.181 | 0.350 ± 0.187 | 0.380 ± 0.184 | yes | 0.0 | 3/3 ok |
| inspre | native | 0.531 ± 0.277 | 0.224 ± 0.144 | 0.377 ± 0.157 | yes | 0.0 | 3/3 ok |
| pearson | native | 0.194 ± 0.021 | 0.508 ± 0.244 | 0.351 ± 0.122 | yes | 0.0 | 3/3 ok |
| causalgrn | native | 0.184 ± 0.052 | 0.457 ± 0.091 | 0.320 ± 0.019 | yes | 0.6 | 3/3 ok |
| consensus(bicycle_lite+dotears_lite+inspre+mean_difference) | native | 0.363 ± 0.120 | 0.228 ± 0.172 | 0.295 ± 0.144 | yes | 0.0 | 3/3 ok |
| ridge_cv | adapted | 0.364 ± 0.121 | 0.198 ± 0.073 | 0.281 ± 0.043 | yes | 28.1 | 3/3 ok |
| nir | native | 0.265 ± 0.062 | 0.292 ± 0.071 | 0.278 ± 0.058 | yes | 1.2 | 3/3 ok |
| arachne | native | 0.012 ± 0.002 | 0.453 ± 0.237 | 0.232 ± 0.119 | yes | 2.8 | 3/3 ok |
| sfinn | native | 0.266 ± 0.153 | 0.194 ± 0.202 | 0.230 ± 0.156 | yes | 0.0 | 3/3 ok |
| bicycle_lite | native | 0.335 ± 0.220 | 0.122 ± 0.125 | 0.229 ± 0.145 | yes | 0.2 | 3/3 ok |
| g2s | adapted | 0.263 ± 0.291 | 0.194 ± 0.133 | 0.228 ± 0.211 | yes | 2.5 | 3/3 ok |
| nir | adapted | 0.244 ± 0.127 | 0.208 ± 0.167 | 0.226 ± 0.105 | yes | 0.1 | 3/3 ok |
| dmdgrn | native | 0.023 ± 0.011 | 0.429 ± 0.145 | 0.226 ± 0.072 | yes | 0.1 | 3/3 ok |
| dotears_lite | native | 0.315 ± 0.205 | 0.131 ± 0.130 | 0.223 ± 0.137 | yes | 0.3 | 3/3 ok |
| score_vote | adapted | 0.296 ± 0.278 | 0.149 ± 0.149 | 0.222 ± 0.213 | yes | 0.5 | 3/3 ok |
| vote | adapted | 0.266 ± 0.289 | 0.147 ± 0.148 | 0.207 ± 0.218 | yes | 0.2 | 3/3 ok |
| cellbox | adapted | 0.342 ± 0.146 | 0.008 ± 0.001 | 0.175 ± 0.074 | yes | 0.1 | 3/3 ok |
| s2g | adapted | 0.208 ± 0.191 | 0.140 ± 0.108 | 0.174 ± 0.149 | yes | 1.6 | 3/3 ok |
| fast | native | 0.303 ± 0.274 | 0.006 ± 0.002 | 0.154 ± 0.138 | no | 0.1 | 3/3 ok |
| atlasshift | native | 0.176 ± 0.142 | 0.123 ± 0.121 | 0.150 ± 0.083 | yes | 0.0 | 3/3 ok |
| ridge_cv | native | 0.260 ± 0.171 | 0.006 ± 0.002 | 0.133 ± 0.086 | no | 4.8 | 3/3 ok |
| genie3 | native | 0.007 ± 0.002 | 0.235 ± 0.104 | 0.121 ± 0.051 | yes | 2.7 | 3/3 ok |
| deeprig | native | 0.022 ± 0.021 | 0.164 ± 0.113 | 0.093 ± 0.061 | yes | 0.7 | 3/3 ok |
| netdeconv | adapted | 0.142 ± 0.143 | 0.007 ± 0.001 | 0.074 ± 0.072 | yes | 0.0 | 3/3 ok |
| notears | native | 0.026 ± 0.021 | 0.095 ± 0.044 | 0.061 ± 0.015 | yes | 9.6 | 3/3 ok |
| fast | adapted | 0.072 ± 0.046 | 0.045 ± 0.046 | 0.058 ± 0.046 | yes | 0.0 | 3/3 ok |
| gene_features | native | 0.042 ± 0.030 | 0.018 ± 0.009 | 0.030 ± 0.014 | yes | 0.0 | 3/3 ok |
| grnboost | native | 0.011 ± 0.010 | 0.036 ± 0.011 | 0.024 ± 0.010 | yes | 3.3 | 3/3 ok |
| dmdgrn | adapted | 0.035 ± 0.039 | 0.006 ± 0.001 | 0.021 ± 0.019 | yes | 0.0 | 3/3 ok |
| by_chance | native | 0.011 ± 0.002 | 0.006 ± 0.002 | 0.008 ± 0.001 | yes | 0.0 | 3/3 ok |
| cpa_lite | native | 0.009 ± 0.006 | 0.008 ± 0.003 | 0.008 ± 0.004 | yes | 0.0 | 3/3 ok |
| s2g | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| g2s | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| score_vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| netdeconv | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |

### 500 genes, 100 perturbed, 3 seeds  (random: A 0.004, B 0.0050)
| method | mode | AUPR A (P->P) | AUPR B (P->unperturbed) | balanced | scores B? | s | status |
|---|---|---|---|---|---|---|---|
| consensus(bicycle_lite+dotears_lite+inspre+mean_difference) | native | 0.480 ± 0.107 | 0.268 ± 0.160 | 0.374 ± 0.125 | yes | 0.0 | 3/3 ok |
| nir | native | 0.366 ± 0.088 | 0.327 ± 0.093 | 0.347 ± 0.083 | yes | 2.2 | 3/3 ok |
| mean_difference | native | 0.334 ± 0.179 | 0.303 ± 0.190 | 0.319 ± 0.177 | yes | 0.0 | 3/3 ok |
| causalgrn | native | 0.182 ± 0.013 | 0.453 ± 0.181 | 0.317 ± 0.093 | yes | 0.5 | 3/3 ok |
| dmdgrn | native | 0.131 ± 0.016 | 0.490 ± 0.057 | 0.310 ± 0.035 | yes | 0.1 | 3/3 ok |
| inspre | native | 0.278 ± 0.127 | 0.283 ± 0.177 | 0.280 ± 0.145 | yes | 0.1 | 3/3 ok |
| pearson | native | 0.167 ± 0.063 | 0.351 ± 0.211 | 0.259 ± 0.135 | yes | 0.0 | 3/3 ok |
| s2g | adapted | 0.267 ± 0.143 | 0.242 ± 0.116 | 0.255 ± 0.129 | yes | 1.2 | 3/3 ok |
| arachne | native | 0.131 ± 0.067 | 0.372 ± 0.183 | 0.251 ± 0.124 | yes | 2.6 | 3/3 ok |
| nir | adapted | 0.221 ± 0.164 | 0.241 ± 0.151 | 0.231 ± 0.142 | yes | 0.3 | 3/3 ok |
| cellbox | adapted | 0.448 ± 0.112 | 0.012 ± 0.008 | 0.230 ± 0.055 | yes | 0.1 | 3/3 ok |
| ridge_cv | native | 0.426 ± 0.068 | 0.005 ± 0.001 | 0.215 ± 0.034 | no | 5.8 | 3/3 ok |
| sfinn | native | 0.231 ± 0.234 | 0.195 ± 0.156 | 0.213 ± 0.193 | yes | 0.0 | 3/3 ok |
| ridge_cv | adapted | 0.242 ± 0.224 | 0.170 ± 0.050 | 0.206 ± 0.113 | yes | 25.1 | 3/3 ok |
| dotears_lite | native | 0.210 ± 0.030 | 0.156 ± 0.135 | 0.183 ± 0.082 | yes | 3.2 | 3/3 ok |
| score_vote | adapted | 0.183 ± 0.135 | 0.178 ± 0.190 | 0.181 ± 0.163 | yes | 0.8 | 3/3 ok |
| atlasshift | native | 0.195 ± 0.209 | 0.158 ± 0.139 | 0.177 ± 0.173 | yes | 0.0 | 3/3 ok |
| genie3 | native | 0.036 ± 0.007 | 0.309 ± 0.175 | 0.173 ± 0.091 | yes | 3.3 | 3/3 ok |
| vote | adapted | 0.111 ± 0.100 | 0.183 ± 0.181 | 0.147 ± 0.140 | yes | 0.6 | 3/3 ok |
| bicycle_lite | native | 0.226 ± 0.063 | 0.067 ± 0.033 | 0.147 ± 0.048 | yes | 2.2 | 3/3 ok |
| fast | native | 0.261 ± 0.050 | 0.005 ± 0.001 | 0.133 ± 0.025 | no | 0.1 | 3/3 ok |
| g2s | adapted | 0.077 ± 0.078 | 0.144 ± 0.077 | 0.110 ± 0.071 | yes | 33.7 | 3/3 ok |
| cellbox | native | 0.147 ± 0.096 | 0.031 ± 0.015 | 0.089 ± 0.056 | yes | 0.5 | 3/3 ok |
| netdeconv | adapted | 0.144 ± 0.085 | 0.009 ± 0.005 | 0.077 ± 0.043 | yes | 0.0 | 3/3 ok |
| deeprig | native | 0.043 ± 0.023 | 0.100 ± 0.070 | 0.071 ± 0.046 | yes | 0.7 | 3/3 ok |
| grnboost | native | 0.034 ± 0.015 | 0.068 ± 0.029 | 0.051 ± 0.008 | yes | 5.4 | 3/3 ok |
| fast | adapted | 0.061 ± 0.043 | 0.012 ± 0.002 | 0.036 ± 0.021 | yes | 0.0 | 3/3 ok |
| dmdgrn | adapted | 0.063 ± 0.021 | 0.009 ± 0.003 | 0.036 ± 0.012 | yes | 0.0 | 3/3 ok |
| notears | native | 0.013 ± 0.012 | 0.007 ± 0.003 | 0.010 ± 0.006 | yes | 1.2 | 3/3 ok |
| gene_features | native | 0.010 ± 0.003 | 0.009 ± 0.002 | 0.009 ± 0.002 | yes | 0.0 | 3/3 ok |
| cpa_lite | native | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.006 ± 0.000 | yes | 0.0 | 3/3 ok |
| by_chance | native | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.005 ± 0.000 | yes | 0.0 | 3/3 ok |
| s2g | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| g2s | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| score_vote | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
| netdeconv | native | - | - | - | - | - | error: ValueError('matmul: Input operand 1 has a mismatch in its core dimensi |
