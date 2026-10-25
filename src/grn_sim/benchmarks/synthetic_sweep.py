"""
Synthetic simulation sweep benchmark module.
Replicates the topology, noise, and connectivity sweeps from Chapter 4 of the thesis.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Sequence
import numpy as np

from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.evaluation.metrics import evaluate_inference, BenchmarkMetrics


@dataclass
class SweepResult:
    """Holds results of a parameter sweep across multiple algorithms."""
    parameter_name: str
    parameter_values: List[Any]
    metrics_by_algo: Dict[str, List[BenchmarkMetrics]]


def run_topology_comparison(
    network_sizes: Sequence[int] = (10, 20),
    k_av: float = 1.0,
    repeats: int = 3,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compares S2G, G2S, NIR, and FAST across Random and Power-Law topologies
    (replicating Figure 4.1 in thesis).

    Returns:
        Dictionary mapping topology -> algo -> list of mean coverage values per network size.
    """
    rng = np.random.default_rng(seed)
    topologies = [TopologyType.RANDOM, TopologyType.POWER_LAW]
    results: Dict[str, Dict[str, List[float]]] = {}

    algos = {
        "s2g": SimpleToGeneral(),
        "g2s": GeneralToSpecific(),
        "nir": NIRAlgorithm(k_interactions=max(1, int(round(k_av)))),
        "fast": FastSVD(k_interactions=max(1, int(round(k_av)))),
        "voting": VotingEnsemble(),
    }

    sim = PerturbationSimulator(perturbation_factor=2.0)

    for topo_type in topologies:
        results[topo_type.value] = {name: [] for name in algos}

        for n in network_sizes:
            coverage_sums = {name: 0.0 for name in algos}

            for _rep in range(repeats):
                net = GeneNetwork.create(
                    n=n,
                    k_av=k_av,
                    net_type=topo_type,
                    ratio=0.5,
                    seed=rng.integers(0, 100000),
                )
                exp = sim.run(net, seed=rng.integers(0, 100000))

                for name, algo in algos.items():
                    res = algo.fit(exp.X, exp.U)
                    m = evaluate_inference(exp.true_matrix, res.adjacency_pred)
                    coverage_sums[name] += m.coverage_pct

            for name in algos:
                mean_cov = coverage_sums[name] / repeats
                results[topo_type.value][name].append(round(mean_cov, 2))

    return results


def run_noise_sweep(
    network_size: int = 10,
    noise_levels: Sequence[float] = (0.0, 0.02, 0.05, 0.10),
    k_av: float = 1.5,
    repeats: int = 3,
    seed: int = 123,
) -> Dict[str, List[float]]:
    """
    Evaluates sensitivity of algorithms under increasing measurement noise
    (replicating Section 4.3 in thesis).
    """
    rng = np.random.default_rng(seed)
    algos = {
        "s2g": SimpleToGeneral(),
        "g2s": GeneralToSpecific(),
        "nir": NIRAlgorithm(k_interactions=max(1, int(round(k_av)))),
        "voting": VotingEnsemble(),
    }

    results: Dict[str, List[float]] = {name: [] for name in algos}

    for noise in noise_levels:
        sim = PerturbationSimulator(
            perturbation_factor=2.0,
            measurement_noise=noise,
        )
        cov_sums = {name: 0.0 for name in algos}

        for _rep in range(repeats):
            net = GeneNetwork.create(
                n=network_size,
                k_av=k_av,
                net_type=TopologyType.POWER_LAW,
                seed=rng.integers(0, 100000),
            )
            exp = sim.run(net, seed=rng.integers(0, 100000))

            for name, algo in algos.items():
                res = algo.fit(exp.X, exp.U)
                m = evaluate_inference(exp.true_matrix, res.adjacency_pred)
                cov_sums[name] += m.coverage_pct

        for name in algos:
            results[name].append(round(cov_sums[name] / repeats, 2))

    return results
