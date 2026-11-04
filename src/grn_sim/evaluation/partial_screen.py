"""
Scoring of partial screens (m perturbed genes out of n measured) against the simulator's direct truth.

Edges are ``S[regulator, target]``. Only perturbed genes can be scored as regulators (a regulator that was never perturbed
cannot be identified from interventions), so the candidate pairs are split by the target:

* **A: P -> P** edges among the perturbed genes (every method can score these);
* **B: P -> Q** edges from a perturbed gene to an *unperturbed* gene (needs a method that scores responders);
* **balanced** = mean of A and B, so a method that only scores A (or only B) cannot win on the easy block alone.

For each block: AUPR, precision at k (k = number of true edges in the block) and the AUPR a random ranking would have
(the base rate). The truth is ``screen.adjacency`` (direct edges). Scores of 0 everywhere in a block (a method that cannot
score it) give an AUPR at the base rate, so the "scored" flag reports whether the method put any signal there.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from grn_sim.evaluation.metrics import compute_aupr


def perturbed_genes(screen: Any) -> np.ndarray:
    return np.array(sorted({c[0] for c in screen.conditions if len(c) == 1}), dtype=int)


def _p_at_k(T: np.ndarray, S: np.ndarray) -> float:
    k = int(T.sum())
    if k == 0:
        return float("nan")
    top = np.argsort(-S, kind="mergesort")[:k]
    return float(T[top].sum() / k)


def evaluate_partial(screen: Any, S: np.ndarray) -> Dict[str, float]:
    n = screen.n_genes
    P = perturbed_genes(screen)
    inP = np.zeros(n, dtype=bool)
    inP[P] = True
    direct = np.asarray(screen.adjacency) != 0
    S = np.abs(np.real(np.asarray(S, dtype=float)))
    eye = np.eye(n, dtype=bool)
    out: Dict[str, float] = {"n_perturbed": float(len(P))}
    for name, tgt in (("A", inP), ("B", ~inP)):
        mask = inP[:, None] & tgt[None, :] & ~eye
        T, Sm = direct[mask], S[mask]
        out[f"aupr_{name}"] = compute_aupr(T, Sm)
        out[f"p_at_k_{name}"] = _p_at_k(T, Sm)
        out[f"base_{name}"] = float(T.mean()) if T.size else 0.0
        out[f"scored_{name}"] = float(np.count_nonzero(Sm) > 0)
    out["aupr_balanced"] = 0.5 * (out["aupr_A"] + out["aupr_B"])
    return out
