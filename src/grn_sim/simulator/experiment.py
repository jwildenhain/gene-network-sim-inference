"""
Perturbation experiment simulator.
Simulates high-throughput gene perturbation assays matching run_perturb_model_12.m:
- Calculates unperturbed reference steady-state x^ss
- Sequentially applies transcriptional perturbation u_j to each gene j
- Integrates system to perturbed steady-state x^{p, j}
- Computes normalized perturbation response matrix X:
    X_ij = (x_i^{p, j} / x_i^{ss}) - 1
- Returns response matrix X and perturbation matrix U
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.ode_solver import ODESolver
from grn_sim.simulator.noise import apply_measurement_noise, apply_biological_variability


@dataclass
class SimulationExperiment:
    """Holds data and ground truth resulting from a simulated perturbation experiment."""
    network: GeneNetwork
    true_matrix: np.ndarray             # (n, n) ground truth interaction matrix T
    X: np.ndarray                       # (n, m) expression response matrix (gene i, perturbation j)
    U: np.ndarray                       # (n, m) perturbation matrix applied to gene i in experiment j
    reference_steady_state: np.ndarray  # (n,) unperturbed steady-state expression
    perturbed_states: np.ndarray        # (n, m) raw steady-state concentrations under perturbation
    error_conc: np.ndarray              # (n, m) concentration measurement error
    error_pert: np.ndarray              # (n, m) perturbation error
    is_valid_steady_state: bool         # Whether all simulations reached valid steady state
    basal_trajectory: Optional[np.ndarray] = None           # (n, steps) trajectory to basal steady state
    perturbation_trajectories: Optional[list] = None        # list of (n, steps) trajectories per perturbed gene j


class PerturbationSimulator:
    """Executes high-throughput perturbation experiments on a GeneNetwork."""

    def __init__(
        self,
        perturbation_factor: float = 2.0,
        measurement_noise: float = 0.0,
        perturbation_noise: float = 0.0,
        biological_variability: float = 0.0,
        solver: Optional[ODESolver] = None,
    ):
        self.perturbation_factor = perturbation_factor
        self.measurement_noise = measurement_noise
        self.perturbation_noise = perturbation_noise
        self.biological_variability = biological_variability
        self.solver = solver or ODESolver()

    def run(
        self,
        network: GeneNetwork,
        seed: Optional[int] = None,
    ) -> SimulationExperiment:
        """
        Runs full set of single-gene perturbation experiments.

        Args:
            network: GeneNetwork model.
            seed: Optional random seed for reproducible noise.

        Returns:
            SimulationExperiment containing X, U, trajectories, and true matrix.
        """
        rng = np.random.default_rng(seed)
        n = network.n

        # Apply biological variability to network parameters if requested
        if self.biological_variability > 0.0:
            sim_net = apply_biological_variability(network, self.biological_variability, rng)
        else:
            sim_net = network

        # 1. Compute reference unperturbed steady state
        x_ss, ss_ok, basal_traj = self.solver.solve_to_steady_state(sim_net)
        if not ss_ok or np.any(x_ss <= 0):
            # Fallback retry with small perturbation if starting state hit zero
            x_ss, ss_ok, basal_traj = self.solver.solve_to_steady_state(sim_net, x0=np.ones(n))

        sim_net.steady_state_unperturbed = x_ss

        # Matrices to store results: n genes by n experiments
        raw_perturbed = np.zeros((n, n), dtype=float)
        X = np.zeros((n, n), dtype=float)
        U = np.zeros((n, n), dtype=float)
        error_conc = np.zeros((n, n), dtype=float)
        error_pert = np.zeros((n, n), dtype=float)
        all_steady = ss_ok
        pert_trajectories = []

        # 2. Perturb each gene sequentially
        for j in range(n):
            u_vec = np.zeros(n, dtype=float)
            # Sample perturbation magnitude with noise
            p_val = self.perturbation_factor
            if self.perturbation_noise > 0.0:
                p_noisy = apply_measurement_noise(np.array([p_val]), self.perturbation_noise, rng)[0]
                u_vec[j] = p_noisy
                error_pert[j, j] = abs(p_val - p_noisy) ** 2
            else:
                u_vec[j] = p_val
                error_pert[j, j] = 0.0

            U[j, j] = u_vec[j]

            # Solve system from reference steady state under perturbation u_vec
            x_pert, p_steady, p_traj = self.solver.solve_to_steady_state(
                sim_net,
                x0=x_ss.copy(),
                perturbation=u_vec,
            )
            pert_trajectories.append(p_traj)

            if not p_steady:
                all_steady = False

            # Apply measurement noise to observed steady state
            if self.measurement_noise > 0.0:
                x_meas = apply_measurement_noise(x_pert, self.measurement_noise, rng)
                error_conc[:, j] = np.abs(x_pert - x_meas) ** 2
            else:
                x_meas = x_pert.copy()
                error_conc[:, j] = 0.0

            raw_perturbed[:, j] = x_meas

            # Compute relative expression change: X_ij = (x_meas / x_ss) - 1
            # Avoid division by zero
            safe_ss = np.maximum(x_ss, 1e-12)
            X[:, j] = (x_meas / safe_ss) - 1.0

        return SimulationExperiment(
            network=sim_net,
            true_matrix=network.topology.adjacency.copy(),
            X=X,
            U=U,
            reference_steady_state=x_ss,
            perturbed_states=raw_perturbed,
            error_conc=error_conc,
            error_pert=error_pert,
            is_valid_steady_state=all_steady,
            basal_trajectory=basal_traj,
            perturbation_trajectories=pert_trajectories,
        )
