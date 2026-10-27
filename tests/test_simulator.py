"""
Unit tests for kinetic rates, ODE solver, and perturbation simulator.
"""

import numpy as np
from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.ode_solver import ODESolver
from grn_sim.simulator.kinetics import evaluate_derivatives
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.noise import apply_measurement_noise


def test_derivatives_shape():
    net = GeneNetwork.create(n=5, k_av=1.0, net_type=TopologyType.RANDOM, seed=42)
    x = np.ones(5)
    dxdt = evaluate_derivatives(x, net)
    assert dxdt.shape == (5,)
    assert np.all(np.isfinite(dxdt))


def test_steady_state_solver():
    net = GeneNetwork.create(n=4, k_av=1.0, net_type=TopologyType.RANDOM, seed=123)
    solver = ODESolver(dt=0.05, max_steps=3000, efnorm_threshold=1e-4)
    x_ss, is_steady, trajectory = solver.solve_to_steady_state(net)
    assert is_steady
    assert x_ss.shape == (4,)
    assert np.all(x_ss > 0)
    assert trajectory.shape[0] == 4


def test_perturbation_experiment():
    net = GeneNetwork.create(n=6, k_av=1.2, net_type=TopologyType.POWER_LAW, seed=99)
    sim = PerturbationSimulator(perturbation_factor=2.0)
    exp = sim.run(net, seed=99)
    assert exp.X.shape == (6, 6)
    assert exp.U.shape == (6, 6)
    assert exp.true_matrix.shape == (6, 6)
    assert np.all(np.isfinite(exp.X))
    assert np.all(np.isfinite(exp.U))


def test_noise_models():
    vals = np.array([1.0, 2.0, 3.0])
    noisy = apply_measurement_noise(vals, noise_level=0.05, rng=np.random.default_rng(42))
    assert noisy.shape == vals.shape
    assert not np.allclose(vals, noisy)
    assert np.all(noisy > 0)
