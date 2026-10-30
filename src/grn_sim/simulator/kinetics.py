"""
Differential equation kinetics and rate law evaluation for gene regulatory networks.
Corresponds to Equation 2.3 and create_network.m / do_perturbation.m:
    dx_i / dt = s_i(x_1, ..., x_n) - b_i * x_i
where:
    s_i = v_i * prod_{j in Inh} [ Ki_j^m_j / (I_j^m_j + Ki_j^m_j) ] * prod_{k in Act} [ 1 + A_k^m_k / (A_k^m_k + Ka_k^m_k) ] + u_i
"""

from typing import Any, Optional, Tuple
import numpy as np

from grn_sim.core.network import GeneNetwork


class KineticsKernel:
    """
    The Hill kinetics of one network under one perturbation, with the per-edge constants precomputed.

    Build one per solve (the network must not change while it is in use) and call :meth:`derivatives`,
    :meth:`jacobian` or :meth:`production` in the inner loop. The arithmetic is exactly that of the original per-gene
    loop (same operations, products accumulated over each target's regulators in ascending order), so results are
    bit-identical to it.
    """

    def __init__(
        self,
        network: GeneNetwork,
        perturbation: Optional[np.ndarray] = None,
        production_scale: Optional[np.ndarray] = None,
    ):
        n = network.n
        T = network.topology.adjacency
        params = network.params
        self.n = n
        self.u = np.asarray(perturbation, dtype=float) if perturbation is not None else np.zeros(n)
        self.scale = None if production_scale is None else np.asarray(production_scale, dtype=float)
        src, dst = np.nonzero(T)
        order = np.lexsort((src, dst))
        self.src, self.dst = src[order], dst[order]
        self.act = T[self.src, self.dst] == 1
        self.m = np.where(self.act, params.n_k[self.src, self.dst], params.n_i[self.src, self.dst])
        self.k_m = np.where(self.act, params.Ka[self.src, self.dst], params.Ki[self.src, self.dst]) ** self.m
        self.v = params.v
        self.B = params.B

    def _terms(self, x: np.ndarray, with_derivative: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        eff_x = np.maximum(x + self.u, 1e-12)
        val = eff_x[self.src]
        val_m = val ** self.m
        denom = val_m + self.k_m + 1e-12
        # Activator: 1 + A^m / (A^m + Ka^m); inhibitor: Ki^m / (I^m + Ki^m).
        term = np.where(self.act, 1.0 + val_m / denom, self.k_m / denom)
        dterm = None
        if with_derivative:
            # d/dA [A^m / (A^m + K^m)] = m A^(m-1) (K^m + 1e-12) / denom^2; the inhibitor factor has the opposite sign.
            base = self.m * (val_m / val) / denom ** 2
            dterm = np.where(self.act, base * (self.k_m + 1e-12), -base * self.k_m)
            # The effective concentration is clipped at 1e-12: no sensitivity below it.
            dterm = np.where(x[self.src] + self.u[self.src] > 1e-12, dterm, 0.0)
        return term, dterm, eff_x

    def production(self, x: np.ndarray, scaled: bool = False) -> np.ndarray:
        """Production ``s_k = v_k * prod_j term_jk`` (times the knockdown scale if ``scaled``)."""
        term, _, _ = self._terms(x)
        prod_terms = np.ones(self.n, dtype=float)
        np.multiply.at(prod_terms, self.dst, term)
        production = prod_terms * self.v
        if scaled and self.scale is not None:
            production = production * self.scale
        return production

    def derivatives(self, x: np.ndarray) -> np.ndarray:
        return self.production(x, scaled=True) - self.B * x

    def elasticities(self, x: np.ndarray) -> np.ndarray:
        term, dterm, eff_x = self._terms(x, with_derivative=True)
        assert dterm is not None
        E = np.zeros((self.n, self.n), dtype=float)
        E[self.src, self.dst] = eff_x[self.src] * dterm / term
        return E

    def jacobian(self, x: np.ndarray, sparse: bool = False) -> Any:
        n = self.n
        term, dterm, _ = self._terms(x, with_derivative=True)
        assert dterm is not None
        s = self.production(x, scaled=True)
        rows = np.concatenate([self.dst, np.arange(n)])
        cols = np.concatenate([self.src, np.arange(n)])
        vals = np.concatenate([s[self.dst] * dterm / term, -self.B])
        if sparse:
            from scipy.sparse import coo_matrix
            return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
        J = np.zeros((n, n), dtype=float)
        np.add.at(J, (rows, cols), vals)
        return J


def evaluate_derivatives(
    x: np.ndarray,
    network: GeneNetwork,
    perturbation: Optional[np.ndarray] = None,
    production_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes dx/dt for all genes in the network given current concentrations x.

    Args:
        x: (n,) array of mRNA concentrations.
        network: GeneNetwork containing topology and kinetic parameters.
        perturbation: (n,) external perturbation vector u, added to the effective concentration of the gene as a
            regulator (an activity boost: the gene's own transcript is not touched directly).
        production_scale: (n,) multiplier of every gene's production rate (CRISPRi-style knockdown: ``1 - efficiency``
            for the repressed gene, 1 elsewhere). ``None`` leaves the rates unchanged.

    Returns:
        (n,) array of derivatives dx/dt.
    """
    return KineticsKernel(network, perturbation, production_scale).derivatives(x)


def production_rates(
    x: np.ndarray,
    network: GeneNetwork,
    perturbation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Production term ``s_k = v_k * prod_j term_jk`` of every gene (before any knockdown scaling)."""
    return KineticsKernel(network, perturbation).production(x)


def edge_elasticities(
    x: np.ndarray,
    network: GeneNetwork,
    perturbation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Local sensitivity of every edge, ``E[j, k] = d log s_k / d log x_j`` (0 where there is no edge).

    An edge with ``|E| << 1`` at the operating point sits on a flat part of its Hill curve: its regulator can change a
    lot without its target noticing, so knocking the regulator down has little downstream effect.
    """
    return KineticsKernel(network, perturbation).elasticities(x)


def evaluate_jacobian(
    x: np.ndarray,
    network: GeneNetwork,
    perturbation: Optional[np.ndarray] = None,
    production_scale: Optional[np.ndarray] = None,
    sparse: bool = False,
) -> Any:
    """
    Analytic Jacobian ``J[k, j] = d(dx_k/dt) / dx_j`` of :func:`evaluate_derivatives`.

    The production of gene k is a product of Hill factors, so ``ds_k/dx_j = s_k * term_jk' / term_jk``; degradation
    adds ``-B_k`` on the diagonal. ``sparse=True`` returns a ``scipy.sparse`` CSC matrix (edges plus diagonal).
    """
    return KineticsKernel(network, perturbation, production_scale).jacobian(x, sparse=sparse)
