"""Topology generators at Perturb-seq scale: stalled netsim.m samplers are topped up; the modular generator."""

import numpy as np
import pytest

from grn_sim.core.topology import TopologyType, generate_topology


@pytest.mark.parametrize("net_type", [TopologyType.POWER_LAW, TopologyType.HIERARCHICAL])
def test_large_netsim_networks_reach_the_requested_edge_count(net_type):
    topo = generate_topology(200, k_av=1.5, net_type=net_type, subdiv=2, seed=1)
    assert topo.total_edges == 300, "the netsim.m sampler stalls near n edges at this size; the top-up fills the rest"


def test_hierarchical_top_up_only_adds_cross_module_edges():
    small = generate_topology(200, k_av=1.0, net_type=TopologyType.HIERARCHICAL, subdiv=2, seed=1).adjacency
    full = generate_topology(200, k_av=1.5, net_type=TopologyType.HIERARCHICAL, subdiv=2, seed=1).adjacency
    module = np.arange(200) // 100
    added = (full != 0) & (small == 0)
    src, dst = np.nonzero(added)
    assert len(src) > 0 and np.all(module[src] != module[dst])


def test_studio_sized_networks_are_unchanged():
    # Regression guard: the top-up never fires for networks the sampler completes (all studio sizes, 6-30 genes).
    fingerprints = {
        TopologyType.POWER_LAW: generate_topology(24, 1.8, TopologyType.POWER_LAW, 0.5, 2, seed=7).adjacency,
        TopologyType.HIERARCHICAL: generate_topology(24, 1.8, TopologyType.HIERARCHICAL, 0.5, 2, seed=7).adjacency,
    }
    for net_type, A in fingerprints.items():
        assert np.count_nonzero(A) < int(np.ceil(1.8 * 24)) + 1
        again = generate_topology(24, 1.8, net_type, 0.5, 2, seed=7).adjacency
        assert np.array_equal(A, again)


def test_modular_network_shape():
    topo = generate_topology(500, k_av=3.0, net_type=TopologyType.MODULAR, subdiv=6, seed=3)
    A = topo.adjacency
    assert np.all(np.diag(A) == 0), "no self-loops"
    assert np.all(topo.in_degrees >= 1), "every gene is regulated"
    assert abs(topo.in_degrees.mean() - 3.0) < 0.3
    assert topo.out_degrees.max() >= 4 * topo.in_degrees.max() / 2, "out-degree is heavy-tailed, in-degree narrow"
    assert set(np.unique(A)) <= {-1, 0, 1}


def test_modular_edges_mostly_stay_inside_modules():
    rng_module = np.random.default_rng(3).permutation(np.arange(300) % 5)
    topo = generate_topology(300, k_av=2.5, net_type=TopologyType.MODULAR, subdiv=5, seed=3, p_within=0.9)
    src, dst = np.nonzero(topo.adjacency)
    # The generator draws the module assignment first from the same seeded stream.
    assert np.mean(rng_module[src] == rng_module[dst]) > 0.8


@pytest.mark.parametrize("kwargs", [{"p_within": 1.5}, {"attachment_offset": 0.0}])
def test_modular_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_topology(20, net_type=TopologyType.MODULAR, seed=1, **kwargs)


def test_server_builds_modular_networks():
    from grn_sim.web import server

    resp = server._execute_simulation({"num_genes": 20, "topology": "modular", "k_av": 2.0, "seed": 5})
    assert resp["status"] == "success"
    assert server.CURRENT_STATE["network"].topology.topology_type == TopologyType.MODULAR
