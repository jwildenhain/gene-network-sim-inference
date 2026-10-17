"""
Evaluation metrics for gene regulatory network inference.
Directly reproduces the statistics and metrics from thesis Table 4.1, new_stats.m,
and stats_obs_pred.m.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class BenchmarkMetrics:
    """Metrics matching thesis Table 4.1."""
    coverage_pct: float                 # TP / Total True Edges (%)
    false_positives_pct: float          # FP / Total Possible Edges (%)
    inferred_connections_pct: float     # Total Predicted Edges / Total Possible Edges (%)
    fp_per_inferred_pct: float          # FP / Total Predicted Edges (%)
    inferred_sign_pct: float            # Correct Sign TP / Total True Edges (%)
    total_errors: int                   # FP + FN + Sign Mismatches

    # Standard ML metrics
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    sign_errors: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Coverage %": round(self.coverage_pct, 2),
            "False Positives %": round(self.false_positives_pct, 2),
            "Inferred Connections %": round(self.inferred_connections_pct, 2),
            "FP per Inferred %": round(self.fp_per_inferred_pct, 2),
            "Inferred Sign %": round(self.inferred_sign_pct, 2),
            "Total Errors": self.total_errors,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1-Score": round(self.f1_score, 4),
            "Accuracy": round(self.accuracy, 4),
        }


def evaluate_inference(
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    ignore_diagonal: bool = False,
) -> BenchmarkMetrics:
    """
    Computes benchmark statistics comparing predicted adjacency with true network.

    Args:
        true_matrix: (N, N) true signed interaction matrix (-1, 0, 1).
        predicted_matrix: (N, N) inferred signed interaction matrix (-1, 0, 1).
        ignore_diagonal: If True, self-loops are ignored from evaluation.

    Returns:
        BenchmarkMetrics dataclass matching thesis Table 4.1.
    """
    T = true_matrix.copy()
    P = predicted_matrix.copy()
    N = T.shape[0]

    if ignore_diagonal:
        np.fill_diagonal(T, 0)
        np.fill_diagonal(P, 0)
        total_possible = N * (N - 1)
    else:
        total_possible = N * N

    # Binary topology (edge present vs absent)
    true_edges = (T != 0)
    pred_edges = (P != 0)

    total_true = int(np.sum(true_edges))
    total_pred = int(np.sum(pred_edges))

    tp_mask = true_edges & pred_edges
    fp_mask = (~true_edges) & pred_edges
    fn_mask = true_edges & (~pred_edges)
    tn_mask = (~true_edges) & (~pred_edges)

    TP = int(np.sum(tp_mask))
    FP = int(np.sum(fp_mask))
    FN = int(np.sum(fn_mask))
    TN = int(np.sum(tn_mask))

    # Sign verification among TP
    correct_sign_mask = tp_mask & (np.sign(T) == np.sign(P))
    sign_error_mask = tp_mask & (np.sign(T) != np.sign(P))

    correct_sign_count = int(np.sum(correct_sign_mask))
    sign_errors = int(np.sum(sign_error_mask))

    # Metrics matching Table 4.1
    coverage = (TP / total_true * 100.0) if total_true > 0 else 0.0
    fp_pct = (FP / total_possible * 100.0) if total_possible > 0 else 0.0
    inferred_pct = (total_pred / total_possible * 100.0) if total_possible > 0 else 0.0
    fp_per_inferred = (FP / total_pred * 100.0) if total_pred > 0 else 0.0
    inferred_sign = (correct_sign_count / total_true * 100.0) if total_true > 0 else 0.0
    total_errors = FP + FN + sign_errors

    precision = (TP / total_pred) if total_pred > 0 else 0.0
    recall = (TP / total_true) if total_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / total_possible if total_possible > 0 else 0.0

    return BenchmarkMetrics(
        coverage_pct=coverage,
        false_positives_pct=fp_pct,
        inferred_connections_pct=inferred_pct,
        fp_per_inferred_pct=fp_per_inferred,
        inferred_sign_pct=inferred_sign,
        total_errors=total_errors,
        true_positives=TP,
        false_positives=FP,
        true_negatives=TN,
        false_negatives=FN,
        sign_errors=sign_errors,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
    )
