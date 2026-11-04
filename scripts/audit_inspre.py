"""
Audit of the Gardner-family algorithms (NIR, S2G, G2S, RidgeCV) against inspre on a linear cyclic model.

    python scripts/audit_inspre.py

Data: x_c = -delta_i * (I - G)^-T e_i + noise for every gene i perturbed once (square, one column per gene), i.e. the
total-effect matrix of a stable cyclic linear model. The truth is the direct matrix G; scores are AUPR of the
regulator->target ranking (diagonal ignored), mean of 3 networks of 40 genes and in-degree 3.
"""
import warnings

import numpy as np

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.inspre import InspreAlgorithm
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.s2g import SimpleToGeneral


def make(n, k, seed, sigma):
    rng = np.random.default_rng(seed)
    G = (rng.random((n, n)) < k / n) * rng.normal(size=(n, n)) * 0.5
    np.fill_diagonal(G, 0)
    r = max(abs(np.linalg.eigvals(G)))
    G = G * (0.7 / r) if r > 0.7 else G
    T = np.linalg.inv(np.eye(n) - G)
    d = rng.uniform(0.5, 1.0, size=n)
    return G, (-d[:, None] * T).T + rng.normal(scale=sigma, size=(n, n)), -np.diag(d)


def main():
    warnings.filterwarnings("ignore")
    n, k = 40, 3
    algos = {
        "inspre": (lambda: InspreAlgorithm(k_interactions=k), False),
        "nir K=3": (lambda: NIRAlgorithm(k_interactions=k), True),
        "ridge_cv (self excluded)": (lambda: KaggleMultiOutputRidge(k_interactions=k, allow_self_regulation=False), True),
        "ridge_cv (self allowed)": (lambda: KaggleMultiOutputRidge(k_interactions=k, allow_self_regulation=True), True),
        "g2s (loo)": (lambda: GeneralToSpecific(drop_rule="loo"), True),
        "s2g": (lambda: SimpleToGeneral(), True),
    }
    print("noise sd  " + "  ".join(f"{a:>28}" for a in algos))
    for sigma in (0.0, 0.02, 0.05, 0.1):
        res = {a: [] for a in algos}
        for seed in range(3):
            G, X, U = make(n, k, seed, sigma)
            for a, (f, target_regulator) in algos.items():
                A = f().fit(X, U).A_est           # Gardner family: A[target, regulator]
                res[a].append(compute_aupr(G != 0, A.T if target_regulator else A, ignore_diagonal=True))
        print(f"{sigma:8}  " + "  ".join(f"{np.mean(v):28.3f}" for v in res.values()))


if __name__ == "__main__":
    main()
