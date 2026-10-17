"""
Differential equation kinetics and rate law evaluation for gene regulatory networks.
Corresponds to Equation 2.3 and create_network.m / do_perturbation.m:
    dx_i / dt = s_i(x_1, ..., x_n) - b_i * x_i
where:
    s_i = v_i * prod_{j in Inh} [ Ki_j^m_j / (I_j^m_j + Ki_j^m_j) ] * prod_{k in Act} [ 1 + A_k^m_k / (A_k^m_k + Ka_k^m_k) ] + u_i
"""

from typing import Optional
import numpy as np

from grn_sim.core.network import GeneNetwork, KineticParameters


def evaluate_derivatives(
    x: np.ndarray,
    network: GeneNetwork,
    perturbation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes dx/dt for all genes in the network given current concentrations x.

    Args:
        x: (n,) array of mRNA concentrations.
        network: GeneNetwork containing topology and kinetic parameters.
        perturbation: (n,) external perturbation vector u (added to transcription factor activities).

    Returns:
        (n,) array of derivatives dx/dt.
    """
    n = network.n
    T = network.topology.adjacency
    params = network.params

    # Effective regulator concentration (accounting for tagged perturbations)
    u = perturbation if perturbation is not None else np.zeros(n)
    eff_x = np.maximum(x + u, 1e-12)

    # Compute production terms for all genes
    prod_terms = np.ones(n, dtype=float)

    for k in range(n):
        term = 1.0
        # Regulators acting on gene k
        incoming_indices = np.where(T[:, k] != 0)[0]
        for j in incoming_indices:
            sign_edge = T[j, k]
            val = eff_x[j]
            if sign_edge == 1:
                # Activator: 1 + A^m / (A^m + Ka^m)
                m = params.n_k[j, k]
                ka = params.Ka[j, k]
                val_m = val ** m
                term *= (1.0 + (val_m / (val_m + (ka ** m) + 1e-12)))
            elif sign_edge == -1:
                # Inhibitor: Ki^m / (I^m + Ki^m)
                m = params.n_i[j, k]
                ki = params.Ki[j, k]
                val_m = val ** m
                ki_m = ki ** m
                term *= (ki_m / (val_m + ki_m + 1e-12))

        prod_terms[k] = term

    # dx/dt = v_k * prod_terms - B_k * x_k
    dx_dt = prod_terms * params.v - params.B * x
    return dx_dt
