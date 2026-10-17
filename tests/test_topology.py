"""
Unit tests for network topology generation.
"""

import numpy as np
import pytest
from grn_sim.core.topology import generate_topology, TopologyType


def test_random_topology():
    n = 10
    k_av = 1.5
    topo = generate_topology(n=n, k_av=k_av, net_type=TopologyType.RANDOM, seed=None)
    assert topo.n == n
    assert topo.adjacency.shape == (n, n)
    assert np.all(np.isin(topo.adjacency, [-1, 0, 1]))
    assert topo.total_edges > 0


def test_power_law_topology():
    n = 15
    k_av = 1.5
    topo = generate_topology(n=n, k_av=k_av, net_type=TopologyType.POWER_LAW)
    assert topo.n == n
    assert topo.adjacency.shape == (n, n)
    assert np.all(np.isin(topo.adjacency, [-1, 0, 1]))
    assert topo.total_edges > 0


def test_hierarchical_topology():
    n = 12
    subdiv = 3
    topo = generate_topology(n=n, k_av=1.5, net_type=TopologyType.HIERARCHICAL, subdiv=subdiv)
    assert topo.n == n
    assert topo.adjacency.shape == (n, n)
    assert topo.total_edges > 0


def test_hierarchical_invalid_subdiv():
    with pytest.raises(ValueError):
        generate_topology(n=10, subdiv=3, net_type=TopologyType.HIERARCHICAL)


def test_degrees_and_signs():
    adj = np.array([
        [0,  1, -1],
        [0,  0,  1],
        [-1, 0,  0]
    ])
    from grn_sim.core.topology import NetworkTopology
    topo = NetworkTopology(adjacency=adj, topology_type=TopologyType.RANDOM, k_av=1.33, ratio=0.5)
    assert topo.total_edges == 4
    assert topo.positive_edges == 2
    assert topo.negative_edges == 2
    np.testing.assert_array_equal(topo.in_degrees, [1, 1, 2])
    np.testing.assert_array_equal(topo.out_degrees, [2, 1, 1])
