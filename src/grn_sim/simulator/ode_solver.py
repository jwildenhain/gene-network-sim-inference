"""
ODE integration and steady-state convergence solver.
Implements Euler stepping and adaptive integration until equilibrium:
    ||x(t + dt)^2 - x(t)^2||_2 <= efnorm_threshold
"""

from typing import Optional, Tuple
import numpy as np

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.kinetics import evaluate_derivatives


class ODESolver:
    """Numerical solver for simulating gene regulatory networks to steady state."""

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 5000,
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

        Args:
            network: GeneNetwork instance.
            x0: Initial concentrations (if None, initialized to 0.5 + rand matching MATLAB).
            perturbation: (n,) perturbation vector.

        Returns:
            Tuple of:
                - final_x: (n,) steady state concentration array.
                - is_steady: bool flag indicating whether steady state criterion was met.
                - trajectory: (n, steps) trajectory of concentrations over time.
        """
        n = network.n
        if x0 is None:
            # Matches create_network.m: series(i, 1) = 0.5 + rand
            x_curr = 0.5 + network.rng.uniform(0.0, 1.0, size=n)
        else:
            x_curr = np.maximum(x0.copy(), self.min_concentration)

        trajectory = [x_curr.copy()]
        is_steady = False
        efnorm_thrs = self.efnorm_threshold

        for step in range(self.max_steps):
            dxdt = evaluate_derivatives(x_curr, network, perturbation)
            # Euler update
            x_next = x_curr + self.dt * dxdt
            x_next = np.maximum(x_next, self.min_concentration)

            # Check convergence matching MATLAB norm(series(:, i+1).^2 - series(:, i).^2)
            diff_sq = np.linalg.norm(x_next ** 2 - x_curr ** 2)

            trajectory.append(x_next.copy())
            x_curr = x_next

            if diff_sq <= efnorm_thrs:
                is_steady = True
                break

        final_x = x_curr
        return final_x, is_steady, np.array(trajectory).T
