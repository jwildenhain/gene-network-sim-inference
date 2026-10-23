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
    perturbation_trajectories: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Classifies a perturbation experiment into 'responsive', 'flat', or 'oscillating'.

    Rules:
    - Oscillation strictly means that the simulation does NOT reach a steady state
      after perturbation (persistent cyclic fluctuations or non-steady limit cycles).
    - If a simulation reaches a steady state (derivatives vanish / tail is flat):
        * If max relative change >= flat_threshold (5%): 'responsive' (clear level change).
        * If max relative change < flat_threshold (5%): 'flat' (negligible or no change).
    - Simulations reaching steady state are NEVER classified as oscillating.

    Args:
        X: (n, m) expression change matrix (relative fold change or ratio - 1).
        steady_state_diagnostics: Diagnostics dict from ODESolver or simulation engine.
        flat_threshold: Maximum relative expression change below which an experiment is flat.
        perturbation_trajectories: Optional list of trajectory arrays to verify tail dynamics.

    Returns:
        Structured dictionary containing category key, labels, amplitudes, and per-gene/pert statistics.
    """
    diag = steady_state_diagnostics or {}
    
    # Check trajectory tail dynamics if trajectories are supplied
    traj_converged = None
    max_tail_fluct = 0.0
    if perturbation_trajectories:
        traj_converged = True
        for t in perturbation_trajectories:
            arr = np.asarray(t, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 4:
                tail_len = max(3, int(arr.shape[1] * 0.25))
                tail = arr[:, -tail_len:]
                ptp = float(np.max(np.ptp(tail, axis=1)))
                if ptp > max_tail_fluct:
                    max_tail_fluct = ptp
                if ptp > 0.02:
                    traj_converged = False

    raw_is_steady = bool(diag.get("is_valid_steady_state", True))
    raw_is_osc = bool(diag.get("oscillating_detected", False))

    if traj_converged is not None:
        if traj_converged:
            is_steady = True
            is_oscillating = False
        else:
            is_steady = False
            is_oscillating = True
    else:
        # Fallback to diagnostics, ensuring that an oscillation requires not reaching steady state
        if raw_is_steady:
            is_steady = True
            is_oscillating = False
        else:
            is_steady = False
            is_oscillating = bool(raw_is_osc or diag.get("oscillating_perturbations_count", 0) > 0)

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
        p_osc = bool(p_diag.get("oscillating", False)) if not is_steady else False

        if not is_steady and (p_osc or is_oscillating):
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

    # Overall Classification:
    # 1. Oscillating: Simulation does NOT reach a steady state after perturbation
    if (not is_steady) and (is_oscillating or osc_count > 0 or (traj_converged is False)):
        response_type = "oscillating"
        label = "Oscillating (Limit Cycle / Non-steady)"
        badge_class = "badge-amber"
        badge_icon = "〰"
        description = (
            f"Persistent oscillation / limit cycle: Simulation does NOT reach steady state after perturbation "
            f"(tail fluctuation = {max_tail_fluct:.4f}, max shift = {max_amp * 100:.1f}%)."
        )
    # 2. Flat: Steady state reached, but negligible response (< flat_threshold)
    elif max_amp < flat_threshold or rms_amp < RMS_FLAT_THRESHOLD:
        response_type = "flat"
        label = "Flat Levels (Damped / Insensitive)"
        badge_class = "badge-secondary"
        badge_icon = "⚪"
        description = (
            f"Flat expression levels across all genes (max relative shift {max_amp * 100:.2f}% < {flat_threshold * 100:.1f}%). "
            f"Perturbations produce negligible deviations from baseline; trajectory settles horizontally."
        )
    # 3. Responsive: Steady state reached with clear level changes (>= flat_threshold)
    else:
        response_type = "responsive"
        label = "Responsive (Clear Level Changes)"
        badge_class = "badge-emerald"
        badge_icon = "⚡"
        description = (
            f"Clear, distinct steady-state perturbations visible across {resp_genes}/{n_genes} genes "
            f"(max relative shift {max_amp * 100:.1f}%, mean {mean_amp * 100:.1f}%). "
            f"Trajectory smoothly transitions to a new stable perturbed steady state."
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
