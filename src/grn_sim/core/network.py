"""
GeneNetwork data model and parameter initialization.
Captures the underlying ODE kinetic parameters corresponding to create_network.m:
- v: basal rates of transcription
- B: breakdown / degradation rate constants
- Ki: inhibitor half-saturation constants
- Ka: activator half-saturation constants
- n_i: Hill coefficients for inhibition
- n_k: Hill coefficients for activation
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from grn_sim.core.topology import NetworkTopology, generate_topology, TopologyType


@dataclass
class KineticParameters:
    """Kinetic parameters for the differential equations in create_network.m."""
    v: np.ndarray      # (n,) basal rate of transcription
    B: np.ndarray      # (n,) degradation rate constant
    Ki: np.ndarray     # (n, n) inhibitor half-saturation constants
    Ka: np.ndarray     # (n, n) activator half-saturation constants
    n_i: np.ndarray    # (n, n) Hill coefficients for inhibitors
    n_k: np.ndarray    # (n, n) Hill coefficients for activators


class GeneNetwork:
    """Represents a full gene regulatory network with topology and kinetic parameters."""

    def __init__(
        self,
        topology: NetworkTopology,
        params: Optional[KineticParameters] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.topology = topology
        self.n = topology.n
        self.rng = rng or np.random.default_rng()
        self.params = params or self._init_default_parameters()
        self.steady_state_unperturbed: Optional[np.ndarray] = None

    @classmethod
    def create(
        cls,
        n: int,
        k_av: float = 1.5,
        net_type: TopologyType = TopologyType.POWER_LAW,
        ratio: float = 0.5,
        subdiv: int = 2,
        seed: Optional[int] = None,
    ) -> "GeneNetwork":
        """Factory method to generate a network with randomized kinetic parameters."""
        rng = np.random.default_rng(seed)
        topo = generate_topology(
            n=n,
            k_av=k_av,
            net_type=net_type,
            ratio=ratio,
            subdiv=subdiv,
            rng=rng,
        )
        return cls(topology=topo, rng=rng)

    def _init_default_parameters(self) -> KineticParameters:
        """
        Initializes kinetic parameters matching MATLAB create_network.m lines 20-35:
        - n_i, n_k ~ 1 + 5 * U[0, 1] (Hill coefficients in [1, 6])
        - v ~ 1.05 + 0.1 * N(0, 1) (basal rate)
        - B ~ 0.75 + 0.1 * N(0, 1) (degradation rate)
        - Ki, Ka ~ 1 + 1.5 * U[0, 1] (half-saturation in [1, 2.5])
        """
        n = self.n
        rng = self.rng

        v = 1.05 + 0.1 * rng.standard_normal(n)
        B = 0.75 + 0.1 * rng.standard_normal(n)

        # Ensure positive values
        v = np.maximum(v, 0.05)
        B = np.maximum(B, 0.05)

        n_i = 1.0 + 5.0 * rng.uniform(0.0, 1.0, size=(n, n))
        n_k = 1.0 + 5.0 * rng.uniform(0.0, 1.0, size=(n, n))
        Ki = 1.0 + 1.5 * rng.uniform(0.0, 1.0, size=(n, n))
        Ka = 1.0 + 1.5 * rng.uniform(0.0, 1.0, size=(n, n))

        # Non-interacting pairs zeroed out
        no_edge = (self.topology.adjacency == 0)
        n_i[no_edge] = 0.0
        n_k[no_edge] = 0.0
        Ki[no_edge] = 0.0
        Ka[no_edge] = 0.0

        return KineticParameters(v=v, B=B, Ki=Ki, Ka=Ka, n_i=n_i, n_k=n_k)

    @property
    def adjacency(self) -> np.ndarray:
        return self.topology.adjacency
