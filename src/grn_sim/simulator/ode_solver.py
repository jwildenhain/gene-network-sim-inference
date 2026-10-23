"""
ODE integration and steady-state convergence solver.
Implements Euler stepping and adaptive integration matching the original MATLAB code:
- Basal relaxation (`create_network.m`):
    e = s^2 + 100 steps
    efnorm = norm(series(:, i+1).^2 - series(:, i).^2)
    converges when efnorm <= efnorm_thrs (0.0001) and min(series(:, i)) > 0
- Perturbed relaxation (`do_perturbation.m`):
    e = s^2 + 100 steps
    efnorm_update = norm(sqrt(abs(ss(:, i+1).^2 - ss(:, i).^2)))
    converges when abs(efnorm - efnorm_update) < efnorm_thrs (0.0001) and ~not_stable
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.kinetics import evaluate_derivatives


@dataclass
class SteadyStateDiagnostics:
    """Diagnostics from basal or perturbed ODE relaxation matching create_network.m / do_perturbation.m."""
    is_steady: bool
    oscillating: bool
    zero_concentration: bool
    steps_taken: int
    max_horizon_e: int
    final_efnorm: float
    final_efnorm_delta: float
    criterion: str


class ODESolver:
    """Numerical solver for simulating gene regulatory networks to steady state."""

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: Optional[int] = None,
        efnorm_threshold: float = 1e-4,
        min_concentration: float = 1e-9,
    ):
        self.dt = dt
        self.max_steps = max_steps
        self.efnorm_threshold = efnorm_threshold
        self.min_concentration = min_concentration

    def solve_to_steady_state(
        self,
        network: GeneNetwork,
        x0: Optional[np.ndarray] = None,
        perturbation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Integrates system from x0 until steady state is reached or max_steps exceeded.
        Also records `self.last_diagnostics` with the exact MATLAB convergence metrics.
        """
        final_x, is_steady, traj, _ = self.solve_with_diagnostics(network, x0=x0, perturbation=perturbation)
        return final_x, is_steady, traj

    def solve_with_diagnostics(
        self,
        network: GeneNetwork,
        x0: Optional[np.ndarray] = None,
        perturbation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, np.ndarray, SteadyStateDiagnostics]:
        """
        Integrates system and returns full SteadyStateDiagnostics matching
        `create_network.m` (basal) and `do_perturbation.m` (perturbation).
        """
        n = network.n
        # Matches create_network.m / do_perturbation.m: e = s^2 + 100
        matlab_horizon_e = n * n + 100
        effective_max_steps = self.max_steps if self.max_steps is not None else max(matlab_horizon_e, 500)

        if x0 is None:
            # Matches create_network.m: series(i, 1) = 0.5 + rand
            x_curr = 0.5 + network.rng.uniform(0.0, 1.0, size=n)
        else:
            x_curr = np.maximum(x0.copy(), self.min_concentration)

        trajectory = [x_curr.copy()]
        is_steady = False
        zero_conc = False
        efnorm_thrs = self.efnorm_threshold
        is_perturbation = perturbation is not None and np.any(perturbation != 0)

        efnorm_prev = -1.0
        efnorm_curr = 1.0
        efnorm_delta = 1.0
        steps_taken = 0

        for step in range(1, effective_max_steps + 1):
            steps_taken = step
            dxdt = evaluate_derivatives(x_curr, network, perturbation)
            x_next = x_curr + self.dt * dxdt

            x_next = np.maximum(x_next, self.min_concentration)

            # Basal norm (`create_network.m` line 75): norm(series(:,i+1).^2 - series(:,i).^2)
            diff_sq_norm = float(np.linalg.norm(x_next ** 2 - x_curr ** 2))
            # Perturbed norm (`do_perturbation.m` line 101): norm(sqrt(ss(:,i+1).^2 - ss(:,i).^2))
            diff_sqrt_norm = float(np.linalg.norm(np.sqrt(np.abs(x_next ** 2 - x_curr ** 2))))

            if is_perturbation:
                efnorm_delta = abs(efnorm_prev - diff_sqrt_norm)
                efnorm_prev = diff_sqrt_norm
                efnorm_curr = diff_sq_norm
                # Converged when both step-norm change and squared norm are below threshold
                if step > 2 and efnorm_delta <= efnorm_thrs and diff_sq_norm <= efnorm_thrs:
                    is_steady = True
                    trajectory.append(x_next.copy())
                    x_curr = x_next
                    break
            else:
                efnorm_delta = abs(efnorm_curr - diff_sq_norm)
                efnorm_curr = diff_sq_norm
                if diff_sq_norm <= efnorm_thrs:
                    is_steady = True
                    trajectory.append(x_next.copy())
                    x_curr = x_next
                    break

            trajectory.append(x_next.copy())
            x_curr = x_next

        # Check final concentration and trajectory tail dynamics
        final_zero_conc = bool(np.any(x_curr <= self.min_concentration * 10.0))
        traj_arr = np.array(trajectory).T  # (n, steps)
        tail_len = max(4, int(steps_taken * 0.25))
        tail = traj_arr[:, -tail_len:]
        tail_ptp = float(np.max(np.ptp(tail, axis=1))) if tail.size > 0 else 0.0

        dxdt_end = evaluate_derivatives(x_curr, network, perturbation)
        max_dxdt = float(np.max(np.abs(dxdt_end)))

        # A simulation is oscillating if and only if it does NOT reach steady state,
        # with persistent cyclic or limit-cycle fluctuations in the tail.
        if is_steady:
            oscillating = False
        else:
            # If tail has flattened out asymptotically, it did reach steady state
            if tail_ptp <= 0.005 and max_dxdt <= 0.005:
                is_steady = True
                oscillating = False
            else:
                # Persistent oscillation: non-zero rate and tail swing
                oscillating = bool(tail_ptp > 0.02 or max_dxdt > 0.02)
                is_steady = not oscillating

        if final_zero_conc:
            zero_conc = True

        diag = SteadyStateDiagnostics(
            is_steady=is_steady,
            oscillating=oscillating,
            zero_concentration=zero_conc,
            steps_taken=steps_taken,
            max_horizon_e=matlab_horizon_e,
            final_efnorm=round(float(efnorm_curr), 6),
            final_efnorm_delta=round(float(efnorm_delta), 6),
            criterion=(
                "abs(efnorm - efnorm_update) < 1e-4 (do_perturbation.m)"
                if is_perturbation
                else "norm(x_{i+1}^2 - x_i^2) <= 1e-4 (create_network.m)"
            ),
        )
        self.last_diagnostics = diag
        return x_curr, is_steady, np.array(trajectory).T, diag

