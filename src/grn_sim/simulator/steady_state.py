"""
Fast, precise steady-state solver for the Hill kinetics.

Drop-in alternative to :class:`grn_sim.simulator.ode_solver.ODESolver` (same ``solve_with_diagnostics`` signature),
for large networks and for ground truth that must resolve small downstream effects:

1. relax the system until it has nearly settled, so the solution stays in the basin the dynamics actually reach:
   either with Euler steps of the thesis step size (``relax="euler"``, default: cheapest for a few hundred genes) or
   with a stiff integrator that uses the analytic Jacobian (``relax="lsoda"``, also the fallback when Euler has not
   settled within ``max_euler_steps``). Relaxation stops on the residual ``max |dx/dt| <= newton_switch``, not on the
   thesis criterion, which can fire while the system is still drifting slowly;
2. polish with (backtracking) Newton iterations on the sparse Jacobian to the exact fixed point
   (``max |dx/dt|`` ~ 1e-12 instead of the thesis solver's ~1e-4 step criterion);
3. optionally check local stability (all Jacobian eigenvalues with negative real part).

The thesis Euler solver stays the default of the thesis workflows for MATLAB reproducibility; this one is opt-in.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import spsolve

from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.kinetics import KineticsKernel
from grn_sim.simulator.ode_solver import SteadyStateDiagnostics


class SteadyStateSolver:
    """Relaxation plus Newton polishing; exposes the same interface as ``ODESolver``."""

    def __init__(
        self,
        relax: str = "euler",
        relax_tol: float = 1e-6,
        newton_switch: float = 1e-3,
        dt: float = 0.1,
        max_euler_steps: int = 5000,
        record_every: int = 10,
        newton_tol: float = 1e-10,
        chunk_time: float = 20.0,
        max_time: float = 2000.0,
        max_newton: int = 25,
        check_stability: bool = False,
        min_concentration: float = 1e-9,
    ):
        if relax not in ("euler", "lsoda"):
            raise ValueError(f"relax must be 'euler' or 'lsoda', got {relax!r}")
        self.relax = relax
        self.relax_tol = relax_tol
        self.newton_switch = newton_switch
        self.dt = dt
        self.max_euler_steps = max_euler_steps
        self.record_every = record_every
        self.newton_tol = newton_tol
        self.chunk_time = chunk_time
        self.max_time = max_time
        self.max_newton = max_newton
        self.check_stability = check_stability
        self.min_concentration = min_concentration
        # Attribute read by PerturbationSimulator's diagnostics summary.
        self.efnorm_threshold = relax_tol

    def solve_to_steady_state(
        self,
        network: GeneNetwork,
        x0: Optional[np.ndarray] = None,
        perturbation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        x, ok, traj, _ = self.solve_with_diagnostics(network, x0=x0, perturbation=perturbation)
        return x, ok, traj

    def solve_with_diagnostics(
        self,
        network: GeneNetwork,
        x0: Optional[np.ndarray] = None,
        perturbation: Optional[np.ndarray] = None,
        production_scale: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool, np.ndarray, SteadyStateDiagnostics]:
        n = network.n
        if x0 is None:
            # Same random start as the thesis solver (create_network.m).
            x: np.ndarray = np.asarray(0.5 + network.rng.uniform(0.0, 1.0, size=n), dtype=float)
        else:
            x = np.asarray(np.maximum(np.asarray(x0, dtype=float), self.min_concentration), dtype=float)

        kernel = KineticsKernel(network, perturbation, production_scale)

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return kernel.derivatives(np.asarray(np.maximum(y, 0.0)))

        def jac(_t: float, y: np.ndarray) -> np.ndarray:
            return kernel.jacobian(np.asarray(np.maximum(y, 0.0)))

        trajectory = [x.copy()]
        n_eval = 0
        residual = float(np.max(np.abs(kernel.derivatives(x))))
        # 1a. Euler relaxation (cheap steps) until the residual is small enough for Newton.
        if self.relax == "euler":
            for _ in range(self.max_euler_steps):
                if residual <= self.newton_switch:
                    break
                x = np.asarray(np.maximum(x + self.dt * kernel.derivatives(x), self.min_concentration))
                n_eval += 1
                if n_eval % self.record_every == 0:
                    trajectory.append(x.copy())
                residual = float(np.max(np.abs(kernel.derivatives(x))))
            trajectory.append(x.copy())
        # 1b. Stiff integration: the main relaxation for relax="lsoda", the fallback when Euler has not settled.
        t_total = 0.0
        while residual > self.newton_switch and t_total < self.max_time:
            sol = solve_ivp(rhs, (0.0, self.chunk_time), x, method="LSODA", jac=jac, rtol=1e-8, atol=1e-10,
                            t_eval=np.linspace(0.0, self.chunk_time, 6)[1:])
            n_eval += int(sol.nfev)
            if not sol.success or sol.y.shape[1] == 0:
                break
            x = np.asarray(np.maximum(sol.y[:, -1], self.min_concentration))
            trajectory.extend(sol.y.T)
            t_total += self.chunk_time
            residual = float(np.max(np.abs(kernel.derivatives(x))))
        tail = trajectory[-25:]

        # 2. Newton polishing to the exact fixed point (only from a settled state, so it cannot jump basins).
        newton_steps = 0
        last_step = 0.0
        if residual <= self.newton_switch:
            x_relaxed = x.copy()
            for _ in range(self.max_newton):
                f = kernel.derivatives(x)
                f_norm = float(np.max(np.abs(f)))
                if f_norm < self.newton_tol:
                    break
                step = spsolve(kernel.jacobian(x, sparse=True), -f)
                if not np.all(np.isfinite(step)):
                    break
                # Backtracking: halve the step until the residual decreases.
                alpha = 1.0
                accepted = False
                x_new = x
                while alpha >= 1.0 / 64.0:
                    x_new = np.asarray(np.maximum(x + alpha * step, self.min_concentration))
                    if float(np.max(np.abs(kernel.derivatives(x_new)))) < f_norm:
                        accepted = True
                        break
                    alpha *= 0.5
                # Stay near the relaxed state (absolute floor, so nearly silenced genes may still move).
                drift = np.abs(x_new - x_relaxed) / np.maximum(np.abs(x_relaxed), 0.05)
                if not accepted or float(np.max(drift)) > 0.5:
                    break
                last_step = float(np.max(np.abs(alpha * step)))
                x = x_new
                newton_steps += 1
            residual = float(np.max(np.abs(kernel.derivatives(x))))
            trajectory.append(x.copy())

        is_steady = residual <= max(self.relax_tol, self.newton_tol * 100.0)
        oscillating = False
        if not is_steady and len(tail) >= 4:
            ptp = float(np.max(np.ptp(np.asarray(tail), axis=0)))
            oscillating = ptp > 0.02

        if is_steady and self.check_stability:
            eig = np.linalg.eigvals(jac(0.0, x))
            if float(np.max(eig.real)) >= 0.0:
                is_steady = False

        diag = SteadyStateDiagnostics(
            is_steady=bool(is_steady),
            oscillating=bool(oscillating),
            zero_concentration=bool(np.any(x <= self.min_concentration * 10.0)),
            steps_taken=n_eval + newton_steps,
            max_horizon_e=int(self.max_time),
            final_efnorm=float(residual),
            final_efnorm_delta=float(last_step),
            criterion=f"{self.relax} relaxation, then Newton to max|dx/dt| <= {self.newton_tol:g}",
        )
        self.last_diagnostics = diag
        return x, bool(is_steady), np.asarray(trajectory).T, diag
