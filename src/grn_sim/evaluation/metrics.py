"""
Evaluation metrics for gene regulatory network inference.
Directly reproduces the statistics and metrics from thesis Table 4.1 (chap2b.tex / chap4.tex),
new_stats.m, and stats_obs_pred.m.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class BenchmarkMetrics:
    """
    Metrics matching Thesis Table 4.1 (chap2b.tex / chap4.tex), new_stats.m, and stats_obs_pred.m.
    """
    # Thesis Table 4.1 metrics (chap2b.tex lines 9-52)
    coverage_pct: float                 # (TP + TN) / N^2 * 100 (%)
    false_positives_pct: float          # FP / N^2 * 100 (%)
    inferred_connections_pct: float     # TP / Total True Edges * 100 (%) [Sensitivity / Power]
    fp_per_inferred_pct: float          # FP / (TP + FP) * 100 (%) [FDR]
    inferred_sign_pct: float            # Signed TP / N^2 * 100 (%)
    total_errors: int                   # FP + FN + Sign Mismatches (Yeung et al. E)

    # Raw counts
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    signed_true_positives: int
    sign_errors: int

    # Standard ML & new_stats.m metrics
    precision: float                    # PPV = TP / (TP + FP)
    recall: float                       # Sensitivity / Power = TP / (TP + FN)
    f1_score: float
    accuracy: float                     # (TP + TN) / N^2
    fpr: float = 0.0                    # FP / (TN + FP)
    fnr: float = 0.0                    # FN / (TP + FN)
    specificity: float = 0.0            # 1 - FPR = TN / (TN + FP)
    fdr: float = 0.0                    # FP / (TP + FP)
    signed_ppv: float = 0.0             # Signed TP / (TP + FP)
    confusion_matrix_3x3: List[List[int]] = field(default_factory=list)
    obs_pred_error_vec: List[float] = field(default_factory=list)
    overall_misclassification_error: float = 0.0

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
            "FPR": round(self.fpr, 4),
            "FNR": round(self.fnr, 4),
            "Specificity": round(self.specificity, 4),
            "FDR": round(self.fdr, 4),
            "Signed PPV": round(self.signed_ppv, 4),
        }

    def new_stats_vector(self) -> np.ndarray:
        """Returns the exact 8-element vector of new_stats.m lines 193-201."""
        return np.array([
            self.fpr,
            self.fnr,
            self.specificity,
            self.recall,
            self.precision,
            self.fdr,
            self.signed_ppv,
            float(self.total_errors),
        ], dtype=float)


def evaluate_inference(
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    ignore_diagonal: bool = False,
) -> BenchmarkMetrics:
    """
    Computes benchmark statistics comparing predicted adjacency with true network
    matching Thesis Table 4.1 (chap2b.tex), new_stats.m, and stats_obs_pred.m.

    Args:
        true_matrix: (N, N) true signed interaction matrix (-1, 0, 1).
        predicted_matrix: (N, N) inferred signed interaction matrix.
        ignore_diagonal: If True, self-loops are ignored from evaluation.

    Returns:
        BenchmarkMetrics dataclass matching thesis Table 4.1 and new_stats.m.
    """
    T = np.sign(np.real(true_matrix)).astype(int)
    P = np.sign(np.real(predicted_matrix)).astype(int)
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

    if ignore_diagonal:
        off_diag = ~np.eye(N, dtype=bool)
        tp_mask = true_edges & pred_edges & off_diag
        fp_mask = (~true_edges) & pred_edges & off_diag
        fn_mask = true_edges & (~pred_edges) & off_diag
        tn_mask = (~true_edges) & (~pred_edges) & off_diag
    else:
        tp_mask = true_edges & pred_edges
        fp_mask = (~true_edges) & pred_edges
        fn_mask = true_edges & (~pred_edges)
        tn_mask = (~true_edges) & (~pred_edges)

    TP = int(np.sum(tp_mask))
    FP = int(np.sum(fp_mask))
    FN = int(np.sum(fn_mask))
    TN = int(np.sum(tn_mask))

    total_true = TP + FN
    total_pred = TP + FP
    total_zero = TN + FP

    # Sign verification among TP (new_stats.m lines 154-160)
    correct_sign_mask = tp_mask & (T == P)
    sign_error_mask = tp_mask & (T != P)

    STP = int(np.sum(correct_sign_mask))
    sign_errors = int(np.sum(sign_error_mask))

    # Yeung's total number of errors: abs(A_silico - A_true) >= 1 (new_stats.m line 158)
    total_errors = FP + FN + sign_errors

    # Thesis Table 4.1 formulas (chap2b.tex lines 9-52 & chap4.tex lines 169-175):
    # - Coverage % = (TP + TN) / N^2 * 100
    # - False Positives % = FP / N^2 * 100
    # - Inferred Connections % = TP / k_total * 100
    # - FP per Inferred Connection % = FP / (TP + FP) * 100
    # - Inferred Sign % = STP / N^2 * 100
    coverage_pct = ((TP + TN) / total_possible * 100.0) if total_possible > 0 else 0.0
    fp_pct = (FP / total_possible * 100.0) if total_possible > 0 else 0.0
    inferred_connections_pct = (TP / total_true * 100.0) if total_true > 0 else 0.0
    fp_per_inferred_pct = (FP / total_pred * 100.0) if total_pred > 0 else 0.0
    inferred_sign_pct = (STP / total_possible * 100.0) if total_possible > 0 else 0.0

    # new_stats.m 8-element statistics (lines 193-201)
    fpr = (FP / total_zero) if total_zero > 0 else 0.0
    fnr = (FN / total_true) if total_true > 0 else 0.0
    specificity = 1.0 - fpr
    recall = 1.0 - fnr
    precision = (TP / total_pred) if total_pred > 0 else 0.0
    fdr = (FP / total_pred) if total_pred > 0 else 0.0
    signed_ppv = (STP / total_pred) if total_pred > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / total_possible if total_possible > 0 else 0.0

    # stats_obs_pred.m 3x3 confusion matrix (classes = [1, -1, 0])
    classes = [1, -1, 0]
    cmat = np.zeros((3, 3), dtype=int)
    for r_idx, tc in enumerate(classes):
        for c_idx, pc in enumerate(classes):
            mask = (T == tc) & (P == pc)
            if ignore_diagonal:
                mask = mask & (~np.eye(N, dtype=bool))
            cmat[r_idx, c_idx] = int(np.sum(mask))

    error_vec = []
    for r_idx in range(3):
        row_sum = int(np.sum(cmat[r_idx, :]))
        err_r = ((row_sum - cmat[r_idx, r_idx]) / row_sum) if row_sum > 0 else 0.0
        error_vec.append(float(err_r))

    total_cmat = int(np.sum(cmat))
    overall_err = ((total_cmat - int(np.trace(cmat))) / total_cmat) if total_cmat > 0 else 0.0

    return BenchmarkMetrics(
        coverage_pct=coverage_pct,
        false_positives_pct=fp_pct,
        inferred_connections_pct=inferred_connections_pct,
        fp_per_inferred_pct=fp_per_inferred_pct,
        inferred_sign_pct=inferred_sign_pct,
        total_errors=total_errors,
        true_positives=TP,
        false_positives=FP,
        true_negatives=TN,
        false_negatives=FN,
        signed_true_positives=STP,
        sign_errors=sign_errors,
        precision=precision,
        recall=recall,
        f1_score=f1,
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        specificity=specificity,
        fdr=fdr,
        signed_ppv=signed_ppv,
        confusion_matrix_3x3=cmat.tolist(),
        obs_pred_error_vec=error_vec,
        overall_misclassification_error=overall_err,
    )
