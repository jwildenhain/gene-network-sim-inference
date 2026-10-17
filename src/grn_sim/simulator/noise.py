"""
Noise and biological variability models.
Implements the exact thesis error formulation (Equation 2.4):
    x_noisy = x * (1 + sqrt(delta) * theta_1 * (theta_2 - 0.5))
where:
    theta_1 ~ N(0, 1) (Gaussian standard normal)
    theta_2 ~ U(0, 1) (Uniform distribution in [0, 1])
    delta = error variance parameter (e.g. 0.01, 0.05, 0.10)
"""

from typing import Optional
import numpy as np

from grn_sim.core.network import GeneNetwork, KineticParameters


def apply_measurement_noise(
    values: np.ndarray,
    noise_level: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Applies Gaussian measurement noise matching thesis Eq 2.4 / do_perturbation.m:
        val_noisy = val * (1 + sqrt(noise_level) * theta_1 * (theta_2 - 0.5))
    """
    if noise_level <= 0.0:
        return values.copy()

    if rng is None:
        rng = np.random.default_rng()

    theta_1 = rng.standard_normal(size=values.shape)
    theta_2 = rng.uniform(0.0, 1.0, size=values.shape)

    noise_factor = 1.0 + (np.sqrt(noise_level) * theta_1 * (theta_2 - 0.5))
    noisy_values = values * noise_factor
    return np.maximum(noisy_values, 1e-12)


def apply_biological_variability(
    network: GeneNetwork,
    bio_var_level: float,
    rng: Optional[np.random.Generator] = None,
) -> GeneNetwork:
    """
    Generates a variant network simulating biological variability (genetic polymorphisms,
    environmental fluctuations) by perturbing the kinetic constants.
    """
    if bio_var_level <= 0.0:
        return network

    if rng is None:
        rng = network.rng

    p = network.params
    n = network.n

    # Perturb basal rate and degradation constants
    v_noisy = apply_measurement_noise(p.v, bio_var_level, rng)
    B_noisy = apply_measurement_noise(p.B, bio_var_level, rng)

    # Perturb half-saturation constants where edges exist
    Ki_noisy = apply_measurement_noise(p.Ki, bio_var_level, rng)
    Ka_noisy = apply_measurement_noise(p.Ka, bio_var_level, rng)

    # Hill coefficients can fluctuate slightly
    n_i_noisy = np.maximum(1.0, apply_measurement_noise(p.n_i, bio_var_level, rng))
    n_k_noisy = np.maximum(1.0, apply_measurement_noise(p.n_k, bio_var_level, rng))

    no_edge = (network.topology.adjacency == 0)
    Ki_noisy[no_edge] = 0.0
    Ka_noisy[no_edge] = 0.0
    n_i_noisy[no_edge] = 0.0
    n_k_noisy[no_edge] = 0.0

    new_params = KineticParameters(
        v=v_noisy,
        B=B_noisy,
        Ki=Ki_noisy,
        Ka=Ka_noisy,
        n_i=n_i_noisy,
        n_k=n_k_noisy,
    )

    new_net = GeneNetwork(topology=network.topology, params=new_params, rng=rng)
    return new_net
