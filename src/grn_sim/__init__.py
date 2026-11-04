"""
GRN-Sim: Gene Regulatory Network Perturbation Simulator and Reverse Engineering Inference Suite.

Reverse engineered from the Bachelor of Honours Thesis:
"In Silico Modelling of Gene Networks and Reverse Engineering of Transcriptional Networks"
by Jan Wildenhain (2004/2005).
"""

import os as _os

# OpenBLAS's thread pool and fork() do not mix. The algorithms fork process pools and the web server runs them from several
# request threads; a fork while the BLAS thread pool exists can leave the child, and the parent waiting for it, stuck
# forever (reproduced: six concurrent G2S fits hung in one run of three; with one BLAS thread, 8 of 8 finished, 3x
# faster). The many small solves here also run best on one thread. This must happen before NumPy loads, so it is set at
# the top of the package; a value you set yourself (environment variable) wins.
for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

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
