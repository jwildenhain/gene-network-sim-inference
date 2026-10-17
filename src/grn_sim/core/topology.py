"""
Network topology generation module for gene regulatory networks.
Supports Random (Erdos-Renyi), Scale-Free (Barabasi-Albert Power Law),
and Hierarchical Modular topologies with signed regulatory edges (+1: activation, -1: inhibition).
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np


class TopologyType(str, Enum):
    RANDOM = "random"
    POWER_LAW = "power_law"
    HIERARCHICAL = "hierarchical"


class NetworkTopology:
    """Represents a generated network graph structure."""

    def __init__(
        self,
        adjacency: np.ndarray,
        topology_type: TopologyType,
        k_av: float,
        ratio: float,
    ):
        """
        Args:
            adjacency: (n, n) signed adjacency matrix. T[j, i] is effect of gene j on gene i.
                       +1: activation, -1: inhibition, 0: no interaction.
            topology_type: Topology classification.
            k_av: Targeted average degree per node.
            ratio: Ratio of activating edges vs inhibiting edges (e.g. 0.5 = equal).
        """
        self.adjacency = adjacency
        self.topology_type = topology_type
        self.k_av = k_av
        self.ratio = ratio
        self.n = adjacency.shape[0]

    @property
    def in_degrees(self) -> np.ndarray:
        """In-degree per gene (number of regulators influencing this gene)."""
        return np.sum(np.abs(self.adjacency) > 0, axis=0)

    @property
    def out_degrees(self) -> np.ndarray:
        """Out-degree per gene (number of targets influenced by this gene)."""
        return np.sum(np.abs(self.adjacency) > 0, axis=1)

    @property
    def total_degrees(self) -> np.ndarray:
        """Sum of in-degrees and out-degrees."""
        return self.in_degrees + self.out_degrees

    @property
    def total_edges(self) -> int:
        """Total number of non-zero interactions."""
        return int(np.sum(np.abs(self.adjacency) > 0))

    @property
    def positive_edges(self) -> int:
        """Number of activating interactions (+1)."""
        return int(np.sum(self.adjacency > 0))

    @property
    def negative_edges(self) -> int:
        """Number of inhibiting interactions (-1)."""
        return int(np.sum(self.adjacency < 0))

    @property
    def connectivity(self) -> float:
        """Connectivity zeta = k_av / n."""
        return float(self.total_edges) / float(self.n * self.n)


def generate_topology(
    n: int,
    k_av: float = 1.5,
    net_type: TopologyType = TopologyType.POWER_LAW,
    ratio: float = 0.5,
    subdiv: int = 2,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> NetworkTopology:
    """
    Generates a gene regulatory network topology matching netsim.m from the thesis.

    Args:
        n: Number of genes in the network.
        k_av: Target average number of interactions per gene.
        net_type: Topology type ('random', 'power_law', or 'hierarchical').
        ratio: Ratio threshold for positive (activating) vs negative (inhibiting) edges.
        subdiv: Number of subdivisions for hierarchical networks (must divide n).
        rng: Optional NumPy random generator for deterministic seeds.

    Returns:
        NetworkTopology instance.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    T = np.zeros((n, n), dtype=int)
    dk = np.ones(n, dtype=float)  # degree distribution tracking
    target_edges = int(np.ceil(k_av * n))

    def _sample_sign() -> int:
        # Matches MATLAB: my_sign = sign(ratio - rand)
        # If rand < ratio -> +1 (activator), else -1 (inhibitor)
        return 1 if rng.uniform(0.0, 1.0) < ratio else -1

    if net_type == TopologyType.RANDOM:
        # Erdős-Rényi procedure:
        # Phase 1: assign outgoing edge from random node to each node i
        for i in range(n):
            rand_src = rng.integers(0, n)
            dk[rand_src] += 1
            T[rand_src, i] = _sample_sign()

        # Phase 2: add additional random links until target degree reached
        added = n
        while added < target_edges:
            src = rng.integers(0, n)
            dst = rng.integers(0, n)
            dk[src] += 1
            dk[dst] += 1
            T[src, dst] = _sample_sign()
            added += 1

    elif net_type == TopologyType.POWER_LAW:
        # Barabási-Albert preferential attachment procedure from netsim.m
        # Phase 1: each target node connects with probability proportional to degree
        for i in range(n):
            while True:
                candidate = rng.integers(0, n)
                prob = dk[candidate] / np.sum(dk)
                if rng.uniform(0.0, 1.0) < prob:
                    dk[candidate] += 1
                    T[candidate, i] = _sample_sign()
                    break

        # Phase 2: add remaining edges with preferential attachment on both source and target
        added = n
        max_attempts = 100000
        attempts = 0
        while added < target_edges and attempts < max_attempts:
            attempts += 1
            src = rng.integers(0, n)
            dst = rng.integers(0, n)
            sum_dk = np.sum(dk)
            if (dk[src] / sum_dk > rng.uniform(0.0, 1.0)) and (dk[dst] / sum_dk > rng.uniform(0.0, 1.0)):
                dk[src] += 1
                dk[dst] += 1
                T[src, dst] = _sample_sign()
                added += 1

    elif net_type == TopologyType.HIERARCHICAL:
        # Modular hierarchical structure:
        if n % subdiv != 0:
            raise ValueError(f"Network size {n} must be divisible by subdiv {subdiv}")
        sub_size = n // subdiv
        dk_sub = np.ones(sub_size, dtype=float)

        # Build base sub-module using power law
        for i in range(sub_size):
            while True:
                cand = rng.integers(0, sub_size)
                prob = dk_sub[cand] / np.sum(dk_sub)
                if rng.uniform(0.0, 1.0) < prob:
                    dk_sub[cand] += 1
                    T[cand, i] = _sample_sign()
                    break

        # Replicate sub-module across all partitions
        for part in range(1, subdiv):
            start_col = part * sub_size
            end_col = (part + 1) * sub_size
            T[:sub_size, start_col:end_col] = T[:sub_size, :sub_size]
            dk[start_col:end_col] = dk_sub

        dk[:sub_size] = dk_sub

        # Add cross-module edges between different sub-modules
        added = sub_size * subdiv
        attempts = 0
        while added < target_edges and attempts < 100000:
            attempts += 1
            mod1 = rng.integers(0, subdiv)
            mod2 = rng.integers(0, subdiv)
            if mod1 != mod2:
                idx1 = rng.integers(0, sub_size) + (mod1 * sub_size)
                idx2 = rng.integers(0, sub_size) + (mod2 * sub_size)
                sum_dk = np.sum(dk)
                if (dk[idx1] / sum_dk > rng.uniform(0.0, 1.0)) and (dk[idx2] / sum_dk > rng.uniform(0.0, 1.0)):
                    dk[idx1] += 1
                    dk[idx2] += 1
                    T[idx1, idx2] = _sample_sign()
                    added += 1

    else:
        raise ValueError(f"Unknown topology type: {net_type}")

    return NetworkTopology(adjacency=T, topology_type=net_type, k_av=k_av, ratio=ratio)
