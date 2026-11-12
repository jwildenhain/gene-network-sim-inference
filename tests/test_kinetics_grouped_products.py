"""The grouped-product kernel equals the scatter implementation it replaced, bit for bit."""
import numpy as np
import pytest

from grn_sim.simulator.kinetics import KineticsKernel
from grn_sim.core.network import GeneNetwork


def _scatter_production(k, x, scaled=False):
    term, _, _ = k._terms(x)
    prod = np.ones(k.n, dtype=float)
    np.multiply.at(prod, k.dst, term)
    out = prod * k.v
    return out * k.scale if scaled and k.scale is not None else out


def _scatter_jacobian(k, x):
    n = k.n
    term, dterm, _ = k._terms(x, with_derivative=True)
    s = _scatter_production(k, x, scaled=True)
    rows = np.concatenate([k.dst, np.arange(n)])
    cols = np.concatenate([k.src, np.arange(n)])
    vals = np.concatenate([s[k.dst] * dterm / term, -k.B])
    J = np.zeros((n, n))
    np.add.at(J, (rows, cols), vals)
    return J


@pytest.mark.parametrize("n,k_av,seed", [(12, 1.8, 1), (60, 2.5, 2), (200, 3.0, 3)])
def test_production_and_jacobian_are_bit_identical(n, k_av, seed):
    net = GeneNetwork.create(n=n, k_av=k_av, seed=seed)
    rng = np.random.default_rng(seed)
    k = KineticsKernel(net, perturbation=rng.normal(0, 0.1, n), production_scale=rng.uniform(0.2, 1.0, n))
    for _ in range(5):
        x = rng.uniform(0.01, 5.0, n)
        assert np.array_equal(k.production(x), _scatter_production(k, x))
        assert np.array_equal(k.production(x, scaled=True), _scatter_production(k, x, scaled=True))
        assert np.array_equal(k.jacobian(x), _scatter_jacobian(k, x))


def test_a_network_without_edges_has_unit_products():
    net = GeneNetwork.create(n=8, k_av=1.8, seed=4)
    net.topology.adjacency[:] = 0
    k = KineticsKernel(net)
    x = np.full(8, 2.0)
    assert np.array_equal(k.production(x), k.v)
