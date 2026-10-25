"""
GRN-Sim: Gene Regulatory Network Perturbation Simulator and Reverse Engineering Inference Suite.

Reverse engineered from the Bachelor of Honours Thesis:
"In Silico Modelling of Gene Networks and Reverse Engineering of Transcriptional Networks"
by Jan Wildenhain (2004/2005).
"""

__version__ = "0.1.0"
__author__ = "Jan Wildenhain"

from grn_sim.core.topology import generate_topology, NetworkTopology, TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator, SimulationExperiment
from grn_sim.inference.base import InferenceResult
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.by_chance import ByChance
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.evaluation.metrics import evaluate_inference, BenchmarkMetrics

__all__ = [
    "__version__",
    "BenchmarkMetrics",
    "ByChance",
    "FastSVD",
    "GeneNetwork",
    "GeneralToSpecific",
    "InferenceResult",
    "NIRAlgorithm",
    "NetworkTopology",
    "PerturbationSimulator",
    "SimpleToGeneral",
    "SimulationExperiment",
    "TopologyType",
    "VotingEnsemble",
    "evaluate_inference",
    "generate_topology",
]
