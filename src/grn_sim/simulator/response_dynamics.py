"""
Perturbation Response Dynamics & Classification.
Categorizes gene network perturbation experiments into:
1. 'responsive': Clear or several perturbations visible (max relative change >= 0.05, steady state converged).
2. 'flat': Flat levels for each gene with no visible perturbation (max relative change < 0.05, damped/unperturbed).
3. 'oscillating': Dynamical limit cycle, persistent oscillation, or non-convergent trajectory (steady state not reached).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


FLAT_THRESHOLD = 0.05  # 5% relative expression change threshold
RMS_FLAT_THRESHOLD = 0.015


def classify_perturbation_response(
    X: np.ndarray,
    steady_state_diagnostics: Optional[Dict[str, Any]] = None,
    flat_threshold: float = FLAT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Classifies a perturbation experiment into 'responsive', 'flat', or 'oscillating'.

    Args:
        X: (n, m) expression change matrix (relative fold change or ratio - 1).
        steady_state_diagnostics: Diagnostics dict from ODESolver or simulation engine.
        flat_threshold: Maximum relative expression change below which an experiment is flat.

    Returns:
        Structured dictionary containing category key, labels, amplitudes, and per-gene/pert statistics.
    """
    diag = steady_state_diagnostics or {}
    is_steady = bool(diag.get("is_valid_steady_state", True))
    is_oscillating = bool(diag.get("oscillating_detected", False) or not is_steady)
    osc_count = int(diag.get("oscillating_perturbations_count", 1 if is_oscillating else 0))

    X_mat = np.asarray(X, dtype=float)
    if X_mat.size == 0:
        max_amp = 0.0
        mean_amp = 0.0
        rms_amp = 0.0
        var_amp = 0.0
        n_genes, n_perts = 0, 0
    else:
        abs_X = np.abs(X_mat)
        max_amp = float(np.max(abs_X))
        mean_amp = float(np.mean(abs_X))
        rms_amp = float(np.sqrt(np.mean(X_mat ** 2)))
        var_amp = float(np.var(X_mat))
        n_genes, n_perts = X_mat.shape

    # Detailed per-perturbation metrics
    per_pert_stats: List[Dict[str, Any]] = []
    resp_perts = 0
    flat_perts = 0
    pert_diags = diag.get("perturbations", [])

    for j in range(n_perts):
        col = X_mat[:, j] if n_genes > 0 else np.array([])
        p_max = float(np.max(np.abs(col))) if col.size > 0 else 0.0
        p_mean = float(np.mean(np.abs(col))) if col.size > 0 else 0.0
        p_diag = pert_diags[j] if j < len(pert_diags) else {}
        p_osc = bool(p_diag.get("oscillating", False))

        if p_osc or (is_oscillating and p_max > 50.0):
            p_status = "oscillating"
        elif p_max < flat_threshold:
            p_status = "flat"
            flat_perts += 1
        else:
            p_status = "responsive"
            resp_perts += 1

        per_pert_stats.append({
            "pert_idx": j,
            "max_amplitude": round(p_max, 4),
            "mean_amplitude": round(p_mean, 4),
            "status": p_status,
        })

    # Detailed per-gene metrics
    per_gene_stats: List[Dict[str, Any]] = []
    resp_genes = 0
    for i in range(n_genes):
        row = X_mat[i, :] if n_perts > 0 else np.array([])
        g_max = float(np.max(np.abs(row))) if row.size > 0 else 0.0
        g_mean = float(np.mean(np.abs(row))) if row.size > 0 else 0.0
        is_resp = bool(g_max >= flat_threshold)
        if is_resp:
            resp_genes += 1
        per_gene_stats.append({
            "gene_idx": i,
            "max_amplitude": round(g_max, 4),
            "mean_amplitude": round(g_mean, 4),
            "is_responsive": is_resp,
        })

    # Overall Classification
    if is_oscillating or osc_count > 0:
        response_type = "oscillating"
        label = "Oscillating (Limit Cycle / Non-steady)"
        badge_class = "badge-amber"
        badge_icon = "〰"
        description = (
            f"Non-linear limit cycle or oscillation detected ({osc_count} non-steady conditions, "
            f"efnorm > {diag.get('efnorm_threshold', 1e-4)})."
        )
    elif max_amp < flat_threshold or rms_amp < RMS_FLAT_THRESHOLD:
        response_type = "flat"
        label = "Flat Levels (No Visible Perturbation)"
        badge_class = "badge-secondary"
        badge_icon = "⚪"
        description = (
            f"Flat expression levels across all genes (max relative shift {max_amp * 100:.2f}% < {flat_threshold * 100:.1f}%). "
            f"Perturbations produce negligible deviations from baseline."
        )
    else:
        response_type = "responsive"
        label = "Responsive (Clear Perturbations Visible)"
        badge_class = "badge-emerald"
        badge_icon = "⚡"
        description = (
            f"Clear, distinct steady-state perturbations visible across {resp_genes}/{n_genes} genes "
            f"(max relative shift {max_amp * 100:.1f}%, mean {mean_amp * 100:.1f}%)."
        )

    return {
        "response_type": response_type,
        "label": label,
        "short_label": response_type.capitalize(),
        "badge_class": badge_class,
        "badge_icon": badge_icon,
        "description": description,
        "max_amplitude": round(max_amp, 4),
        "mean_amplitude": round(mean_amp, 4),
        "rms_amplitude": round(rms_amp, 4),
        "variance": round(var_amp, 6),
        "flat_threshold": flat_threshold,
        "num_genes": n_genes,
        "num_perts": n_perts,
        "num_responsive_genes": resp_genes,
        "num_responsive_perts": resp_perts,
        "num_flat_perts": flat_perts,
        "num_oscillating_perts": osc_count,
        "per_gene_stats": per_gene_stats,
        "per_pert_stats": per_pert_stats,
        "is_steady": is_steady,
        "is_oscillating": is_oscillating,
    }
