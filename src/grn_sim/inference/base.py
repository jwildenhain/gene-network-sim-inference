"""
Base class and data contracts for gene regulatory network inference algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class InferenceResult:
    """Encapsulates the result of network reverse engineering."""
    algorithm_name: str
    A_est: np.ndarray                   # Continuous reconstructed coefficient matrix (n, n)
    adjacency_pred: np.ndarray          # Discretized signed adjacency matrix (-1, 0, 1)
    time_elapsed: float                 # Running time in seconds
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_predicted_edges(self) -> int:
        return int(np.sum(self.adjacency_pred != 0))


class BaseInferenceAlgorithm(ABC):
    """Abstract base class for network inference algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        U: np.ndarray,
        **kwargs,
    ) -> InferenceResult:
        """
        Infers regulatory matrix A from expression matrix X and perturbation matrix U.

        Solves the steady-state equation:
            A * X = -U

        Args:
            X: (n, m) expression response matrix.
            U: (n, m) perturbation matrix.

        Returns:
            InferenceResult containing estimated continuous and signed interaction matrices.
        """
        pass


def regulator_input_fit(algo: "BaseInferenceAlgorithm", X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
    """
    Fits ``algo`` under the regulator-input perturbation model used by the Hill simulator.

    The simulator adds the perturbation u to the *effective regulator concentration*
    (x + u) rather than to dx/dt, so at steady state  X_k ~ c_k * sum_j F_kj (X + U)_j
    with c_k > 0 (degradation only rescales rows). This is the same linear problem
    ``A Z = -V`` solved by the Gardner-style algorithms with Z = X + U and V = -X.
    The target-row estimate F is transposed so the result uses the repository's
    ground-truth convention ``true_matrix[regulator, target]``.
    """
    kw = dict(kwargs)
    kw["perturbation_mode"] = "additive"
    res = algo.fit(X + U, -X, **kw)
    diag = dict(res.diagnostics)
    diag["perturbation_mode"] = "regulator"
    return InferenceResult(
        algorithm_name=res.algorithm_name,
        A_est=res.A_est.T.copy(),
        adjacency_pred=res.adjacency_pred.T.copy(),
        time_elapsed=res.time_elapsed,
        diagnostics=diag,
    )
