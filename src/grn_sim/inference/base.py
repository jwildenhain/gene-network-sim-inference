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
