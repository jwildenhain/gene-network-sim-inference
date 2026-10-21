"""
Voting Ensemble combining Simple-to-General (S2G) and General-to-Specific (G2S).
Implements the voting consensus strategy described in thesis Section 5.1.7 and Figure 5.10,
which sharply eliminates false positives.
"""

import time
from typing import Optional, List
import numpy as np

from grn_sim.inference.base import BaseInferenceAlgorithm, InferenceResult
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific


class VotingEnsemble(BaseInferenceAlgorithm):
    """
    Voting ensemble between S2G and G2S.
    An interaction is predicted only if both S2G and G2S agree on the presence and sign.
    """

    def __init__(
        self,
        s2g: Optional[SimpleToGeneral] = None,
        g2s: Optional[GeneralToSpecific] = None,
        require_sign_agreement: bool = True,
        extra_members: Optional[List[BaseInferenceAlgorithm]] = None,
        min_votes: Optional[int] = None,
    ):
        super().__init__(name="voting_ensemble")
        self.s2g = s2g or SimpleToGeneral()
        self.g2s = g2s or GeneralToSpecific()
        self.require_sign_agreement = require_sign_agreement
        self.extra_members = list(extra_members or [])
        self.min_votes = min_votes

    def fit(self, X: np.ndarray, U: np.ndarray, **kwargs) -> InferenceResult:
        start_time = time.time()

        if self.extra_members:
            return self._fit_majority(X, U, start_time, **kwargs)

        res_s2g = self.s2g.fit(X, U, **kwargs)
        res_g2s = self.g2s.fit(X, U, **kwargs)

        adj_s2g = res_s2g.adjacency_pred
        adj_g2s = res_g2s.adjacency_pred

        N = X.shape[0]
        consensus_adj = np.zeros((N, N), dtype=int)
        consensus_A = np.zeros((N, N), dtype=float)

        if self.require_sign_agreement:
            # Both must have non-zero edge with the same sign
            agree_pos = (adj_s2g > 0) & (adj_g2s > 0)
            agree_neg = (adj_s2g < 0) & (adj_g2s < 0)
            consensus_adj[agree_pos] = 1
            consensus_adj[agree_neg] = -1
        else:
            # Both must agree edge exists
            agree = (adj_s2g != 0) & (adj_g2s != 0)
            consensus_adj[agree] = np.sign(res_s2g.A_est[agree] + res_g2s.A_est[agree])

        # Average weights for consensus edges
        nonzero = (consensus_adj != 0)
        consensus_A[nonzero] = 0.5 * (res_s2g.A_est[nonzero] + res_g2s.A_est[nonzero])

        elapsed = time.time() - start_time
        return InferenceResult(
            algorithm_name=self.name,
            A_est=consensus_A,
            adjacency_pred=consensus_adj,
            time_elapsed=elapsed,
            diagnostics={
                "s2g_edges": res_s2g.total_predicted_edges,
                "g2s_edges": res_g2s.total_predicted_edges,
                "consensus_edges": int(np.sum(consensus_adj != 0)),
            },
        )

    def _fit_majority(self, X: np.ndarray, U: np.ndarray, start_time: float, **kwargs) -> InferenceResult:
        """
        Majority vote over S2G, G2S and ``extra_members`` (all in true_matrix[regulator, target] frame).
        An edge is kept when at least ``min_votes`` members agree on its sign
        (default: strict majority).
        """
        members = [self.s2g, self.g2s] + self.extra_members
        results = [m.fit(X, U, **kwargs) for m in members]
        need = self.min_votes if self.min_votes is not None else len(members) // 2 + 1
        pos = sum((r.adjacency_pred > 0).astype(int) for r in results)
        neg = sum((r.adjacency_pred < 0).astype(int) for r in results)
        adj = np.zeros(pos.shape, dtype=int)
        adj[pos >= need] = 1
        adj[neg >= need] = -1
        A = np.mean([r.A_est for r in results], axis=0) * (adj != 0)
        return InferenceResult(
            algorithm_name=self.name,
            A_est=A,
            adjacency_pred=adj,
            time_elapsed=time.time() - start_time,
            diagnostics={"members": [r.algorithm_name for r in results], "min_votes": need,
                         "consensus_edges": int(np.sum(adj != 0))},
        )
