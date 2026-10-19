"""
Unit tests for evaluation metrics.
"""

import numpy as np
from grn_sim.evaluation.metrics import evaluate_inference


def test_perfect_prediction():
    T = np.array([
        [ 1,  0, -1],
        [ 0,  1,  0],
        [-1,  0,  0]
    ])
    P = T.copy()

    m = evaluate_inference(T, P)
    assert m.coverage_pct == 100.0
    assert m.false_positives_pct == 0.0
    assert round(m.inferred_sign_pct, 2) == 44.44  # 4 true signed edges out of n^2=9 entries (Table 4.1 formula)
    assert m.signed_ppv == 1.0
    assert m.total_errors == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1_score == 1.0


def test_sign_error_and_false_positives():
    T = np.array([
        [ 1,  0],
        [ 0, -1]
    ])
    # P has one correct edge with inverted sign, and one false positive
    P = np.array([
        [-1,  1],
        [ 0,  0]
    ])

    m = evaluate_inference(T, P)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.sign_errors == 1
    assert m.coverage_pct == 50.0  # 1 out of 2 true edges detected
    assert m.inferred_sign_pct == 0.0  # 0 out of 2 true edges have correct sign
