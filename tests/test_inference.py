"""
Unit tests for reverse engineering inference algorithms.
"""

import numpy as np
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.by_chance import ByChance
from grn_sim.inference.ensemble import VotingEnsemble


def test_linear_recovery_simple():
    # Simple known system: A = [[-1, 0.5], [0, -1]], U = eye(2)
    # A * X = -U => X = -inv(A) * U
    A_true = np.array([
        [-1.0,  0.5],
        [ 0.0, -1.0]
    ])
    U = np.eye(2)
    X = -np.linalg.inv(A_true) @ U

    # S2G
    s2g = SimpleToGeneral()
    res_s2g = s2g.fit(X, U)
    assert res_s2g.A_est.shape == (2, 2)
    assert res_s2g.adjacency_pred.shape == (2, 2)

    # G2S
    g2s = GeneralToSpecific()
    res_g2s = g2s.fit(X, U)
    assert res_g2s.A_est.shape == (2, 2)

    # NIR
    nir = NIRAlgorithm(k_interactions=1)
    res_nir = nir.fit(X, U)
    assert res_nir.A_est.shape == (2, 2)

    # FAST
    fast = FastSVD(k_interactions=1)
    res_fast = fast.fit(X, U)
    assert res_fast.A_est.shape == (2, 2)

    # Ensemble
    ens = VotingEnsemble(s2g=s2g, g2s=g2s)
    res_ens = ens.fit(X, U)
    assert res_ens.adjacency_pred.shape == (2, 2)

    # ByChance
    bc = ByChance(k_interactions=1)
    res_bc = bc.fit(X, U)
    assert res_bc.adjacency_pred.shape == (2, 2)
