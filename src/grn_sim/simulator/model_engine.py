"""
Multi-Paradigm Modeling & Simulation Engine for Gene Regulatory Networks.
Supports 4 distinct synthetic generation techniques:
1. "hill_nonlinear": Non-linear Hill kinetics with saturation products & degradation (Wildenhain & Crampin).
2. "cellbox_neural_ode": Continuous neural ODE relaxation (Sander Lab, Cell Systems 2020).
3. "linear_jacobian": Linear dynamical system with matrix inversion steady states.
4. "perturb_seq_counts": Single-cell 400-cell Gamma-Poisson overdispersion Perturb-seq engine.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import SimulationExperiment, PerturbationSimulator
from grn_sim.simulator.noise import apply_measurement_noise


class SimulationModelingEngine(str, Enum):
    HILL_NONLINEAR = "hill_nonlinear"
    CELLBOX_NEURAL_ODE = "cellbox_neural_ode"
    LINEAR_JACOBIAN = "linear_jacobian"
    PERTURB_SEQ_COUNTS = "perturb_seq_counts"


def run_simulation_with_engine(
    network: GeneNetwork,
    engine: str = "hill_nonlinear",
    perturbation_factor: float = 2.0,
    measurement_noise: float = 0.0,
    seed: Optional[int] = None,
    cells_per_condition: int = 400,
) -> SimulationExperiment:
    """
    Executes a high-throughput network perturbation experiment using the chosen modeling technique.
    """
    eng = engine.lower().strip()
    if eng == "cellbox_neural_ode":
        return _run_cellbox_neural_ode(network, perturbation_factor, measurement_noise, seed)
    elif eng == "linear_jacobian":
        return _run_linear_jacobian(network, perturbation_factor, measurement_noise, seed)
    elif eng == "perturb_seq_counts":
        return _run_perturb_seq_engine(network, perturbation_factor, measurement_noise, seed, cells_per_condition)
    else:  # default: hill_nonlinear
        sim = PerturbationSimulator(
            perturbation_factor=perturbation_factor,
            measurement_noise=measurement_noise,
        )
        return sim.run(network, seed=seed)


def _run_cellbox_neural_ode(
    network: GeneNetwork,
    perturbation_factor: float = 2.0,
    measurement_noise: float = 0.0,
    seed: Optional[int] = None,
) -> SimulationExperiment:
    """
    Integrates continuous neural ODE:
        dx_i / dt = tanh( sum_j W_ij * x_j + b_i + u_i ) - alpha_i * x_i
    where W is scaled adjacency, b is basal activation, and alpha is self-degradation.
    """
    rng = np.random.default_rng(seed)
    N = network.n
    W = network.topology.adjacency.astype(float) * 0.8
    # Enforce stable diagonal (no self-activation)
    np.fill_diagonal(W, 0.0)

    alpha = np.ones(N) * 1.0
    basal_b = np.ones(N) * 0.5

    def rhs(t, x, u_vec):
        act = W @ x + basal_b + u_vec
        return np.tanh(act) - alpha * x

    # 1. Unperturbed baseline steady state
    t_span = (0.0, 50.0)
    x0 = np.ones(N) * 0.5
    u_zero = np.zeros(N)
    sol_basal = solve_ivp(rhs, t_span, x0, args=(u_zero,), method="RK45", rtol=1e-5, atol=1e-6)
    x_ss = np.maximum(sol_basal.y[:, -1], 1e-4)

    # 2. Perturbation experiments
    raw_perturbed = np.zeros((N, N), dtype=float)
    X = np.zeros((N, N), dtype=float)
    U = np.zeros((N, N), dtype=float)
    error_conc = np.zeros((N, N), dtype=float)
    error_pert = np.zeros((N, N), dtype=float)
    pert_trajectories = []
    pert_diags = []

    for j in range(N):
        u_vec = np.zeros(N, dtype=float)
        u_vec[j] = perturbation_factor
        U[j, j] = perturbation_factor

        sol_p = solve_ivp(rhs, t_span, x_ss.copy(), args=(u_vec,), method="RK45", rtol=1e-5, atol=1e-6)
        x_p = np.maximum(sol_p.y[:, -1], 1e-4)
        pert_trajectories.append(sol_p.y)

        if measurement_noise > 0.0:
            x_meas = apply_measurement_noise(x_p, measurement_noise, rng)
            error_conc[:, j] = (x_p - x_meas) ** 2
        else:
            x_meas = x_p.copy()

        raw_perturbed[:, j] = x_meas
        X[:, j] = (x_meas / np.maximum(x_ss, 1e-6)) - 1.0
        pert_diags.append({
            "gene_idx": j,
            "is_steady": bool(sol_p.success),
            "oscillating": False,
            "zero_concentration": bool(np.any(x_meas <= 1e-5)),
            "steps_taken": len(sol_p.t),
            "final_efnorm": float(np.linalg.norm(sol_p.y[:, -1] - sol_p.y[:, -2])),
        })

    diag_summary = {
        "is_valid_steady_state": True,
        "oscillating_detected": False,
        "oscillating_perturbations_count": 0,
        "modeling_engine": "cellbox_neural_ode",
        "basal": {"is_steady": True, "steps_taken": len(sol_basal.t)},
        "perturbations": pert_diags,
    }

    return SimulationExperiment(
        network=network,
        true_matrix=network.topology.adjacency.copy(),
        X=X,
        U=U,
        reference_steady_state=x_ss,
        perturbed_states=raw_perturbed,
        error_conc=error_conc,
        error_pert=error_pert,
        is_valid_steady_state=True,
        basal_trajectory=sol_basal.y,
        perturbation_trajectories=pert_trajectories,
        steady_state_diagnostics=diag_summary,
    )


def _run_linear_jacobian(
    network: GeneNetwork,
    perturbation_factor: float = 2.0,
    measurement_noise: float = 0.0,
    seed: Optional[int] = None,
) -> SimulationExperiment:
    """
    Linear dynamical system:
        dx / dt = A * x - B * x + v + u
    Steady state:
        x_ss = (B - A)^(-1) * v
        x_p  = (B - A)^(-1) * (v + u)
        X = (x_p / x_ss) - 1
    """
    rng = np.random.default_rng(seed)
    N = network.n
    # Scaled adjacency ensuring stability (spectral radius < 1)
    adj = network.topology.adjacency.astype(float)
    B = np.eye(N) * 1.5  # Degradation matrix
    max_eig = np.max(np.abs(np.linalg.eigvals(adj))) if np.any(adj) else 1.0
    A_scaled = (adj / max(1.0, max_eig * 1.2))

    system_matrix = B - A_scaled
    inv_sys = np.linalg.pinv(system_matrix)

    basal_v = np.ones(N) * 2.0
    x_ss = np.maximum(inv_sys @ basal_v, 1e-3)

    raw_perturbed = np.zeros((N, N), dtype=float)
    X = np.zeros((N, N), dtype=float)
    U = np.zeros((N, N), dtype=float)
    error_conc = np.zeros((N, N), dtype=float)
    error_pert = np.zeros((N, N), dtype=float)

    for j in range(N):
        u_vec = np.zeros(N)
        u_vec[j] = perturbation_factor
        U[j, j] = perturbation_factor

        x_p = np.maximum(inv_sys @ (basal_v + u_vec), 1e-3)
        if measurement_noise > 0.0:
            x_meas = apply_measurement_noise(x_p, measurement_noise, rng)
            error_conc[:, j] = (x_p - x_meas) ** 2
        else:
            x_meas = x_p.copy()

        raw_perturbed[:, j] = x_meas
        X[:, j] = (x_meas / x_ss) - 1.0

    diag_summary = {
        "is_valid_steady_state": True,
        "oscillating_detected": False,
        "oscillating_perturbations_count": 0,
        "modeling_engine": "linear_jacobian",
        "basal": {"is_steady": True},
        "perturbations": [],
    }

    return SimulationExperiment(
        network=network,
        true_matrix=network.topology.adjacency.copy(),
        X=X,
        U=U,
        reference_steady_state=x_ss,
        perturbed_states=raw_perturbed,
        error_conc=error_conc,
        error_pert=error_pert,
        is_valid_steady_state=True,
        basal_trajectory=np.column_stack([x_ss, x_ss]),
        perturbation_trajectories=[np.column_stack([x_ss, raw_perturbed[:, j]]) for j in range(N)],
        steady_state_diagnostics=diag_summary,
    )


def _run_perturb_seq_engine(
    network: GeneNetwork,
    perturbation_factor: float = 2.0,
    measurement_noise: float = 0.0,
    seed: Optional[int] = None,
    cells_per_condition: int = 400,
) -> SimulationExperiment:
    """
    First solves non-linear Hill ODE steady-states, then simulates single-cell integer counts
    with Gamma-Poisson biological overdispersion and log-normal library depths.
    Averages single cells per condition to form observed bulk X.
    """
    from grn_sim.simulator.perturb_seq import simulate_perturb_seq_counts

    sim = PerturbationSimulator(
        perturbation_factor=perturbation_factor,
        measurement_noise=measurement_noise,
    )
    base_exp = sim.run(network, seed=seed)

    gene_names = getattr(network, "gene_names", [f"Gene_{i+1:02d}" for i in range(network.n)])
    mean_expr = np.column_stack([base_exp.reference_steady_state, base_exp.perturbed_states])
    counts, cpm, obs_df, var_df = simulate_perturb_seq_counts(
        gene_names=gene_names,
        mean_expression_matrix=mean_expr,
        cells_per_condition=cells_per_condition,
        seed=seed or 42,
    )

    # Average CPM per condition to form reconstructed observed matrix
    N = network.n
    obs_ctrl_cpm = np.mean(cpm[obs_df["is_control"], :], axis=0)
    obs_pert_cpm = np.zeros((N, N), dtype=float)
    X_single_cell = np.zeros((N, N), dtype=float)

    for j in range(N):
        g_name = gene_names[j]
        cond_cells = cpm[obs_df["condition"] == g_name, :]
        if len(cond_cells) > 0:
            cpm_j = np.mean(cond_cells, axis=0)
        else:
            cpm_j = base_exp.perturbed_states[:, j]
        obs_pert_cpm[:, j] = cpm_j
        safe_ctrl = np.maximum(obs_ctrl_cpm, 1e-4)
        X_single_cell[:, j] = (cpm_j / safe_ctrl) - 1.0

    diag = dict(base_exp.steady_state_diagnostics or {})
    diag["modeling_engine"] = "perturb_seq_counts"
    diag["cells_per_condition"] = cells_per_condition
    diag["total_single_cells"] = len(obs_df)

    return SimulationExperiment(
        network=network,
        true_matrix=base_exp.true_matrix,
        X=X_single_cell,
        U=base_exp.U,
        reference_steady_state=obs_ctrl_cpm,
        perturbed_states=obs_pert_cpm,
        error_conc=base_exp.error_conc,
        error_pert=base_exp.error_pert,
        is_valid_steady_state=base_exp.is_valid_steady_state,
        basal_trajectory=base_exp.basal_trajectory,
        perturbation_trajectories=base_exp.perturbation_trajectories,
        steady_state_diagnostics=diag,
    )
