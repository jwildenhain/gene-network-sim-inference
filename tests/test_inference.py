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


def test_matlab_equivalence_sos_table_4_1():
    """Verify 100% numerical equivalence with Thesis Table 4.1 (chap4.tex) & Appendix A.4-A.7."""
    from grn_sim.benchmarks.sos_pathway import run_sos_benchmark

    res = run_sos_benchmark()

    # Exact Thesis Table 4.1 (chap4.tex lines 169-175) & Appendix A.4-A.7 3x3 confusion matrices
    expected_cmat = {
        "s2g":  [[6, 8, 17], [3, 4, 11], [2, 5, 25]],
        "g2s":  [[5, 4, 22], [3, 0, 15], [5, 8, 19]],
        "fast": [[16, 4, 11], [6, 3, 9], [13, 3, 16]],
        "NIR":  [[17, 7, 7], [4, 4, 10], [11, 2, 19]],
    }
    expected_metrics = {
        "s2g":  (56.79, 8.64, 42.86, 25.00, 12.35, 46),
        "g2s":  (38.27, 16.05, 24.49, 52.00, 6.17, 57),
        "fast": (55.56, 19.75, 59.18, 35.56, 23.46, 46),
        "NIR":  (62.96, 16.05, 65.31, 28.89, 25.93, 41),
    }

    for alg in ("s2g", "g2s", "fast", "NIR"):
        m = res[alg]
        assert m.confusion_matrix_3x3 == expected_cmat[alg], f"{alg} 3x3 matrix mismatch"
        exp = expected_metrics[alg]
        assert round(m.coverage_pct, 2) == exp[0]
        assert round(m.false_positives_pct, 2) == exp[1]
        assert round(m.inferred_connections_pct, 2) == exp[2]
        assert round(m.fp_per_inferred_pct, 2) == exp[3]
        assert round(m.inferred_sign_pct, 2) == exp[4]
        assert m.total_errors == exp[5]


def test_matlab_equivalence_60_gene_archive():
    """Verify bit-for-bit match against stored 60-gene MATLAB run random_nz60_run5_iot2_sig0.mat."""
    from pathlib import Path
    import scipy.io as sio
    from grn_sim.evaluation.metrics import evaluate_inference

    mat_path = Path(__file__).resolve().parents[1] / "data" / "matlab_benchmarks" / "random_nz60_run5_iot2_sig0.mat"
    m = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    rnd = m["random"]
    X = np.real(rnd.steady_state_measurement).astype(float)
    P = np.real(rnd.perturbation).astype(float)
    T = np.real(rnd.true_matrix).astype(float)

    res_s2g = SimpleToGeneral().fit(X, P)
    cmat_s2g = np.asarray(evaluate_inference(T, res_s2g.adjacency_pred).confusion_matrix_3x3, dtype=int).T.flatten()
    assert np.array_equal(cmat_s2g, np.asarray(rnd.pred_matrix_sg[:, -1], dtype=int))

    res_nir = NIRAlgorithm(k_interactions=3, top_d=5).fit(X, -P)
    cmat_nir = np.asarray(evaluate_inference(T, res_nir.adjacency_pred).confusion_matrix_3x3, dtype=int).T.flatten()
    assert np.array_equal(cmat_nir, np.asarray(rnd.pred_matrix_gardener[:, -1], dtype=int))



