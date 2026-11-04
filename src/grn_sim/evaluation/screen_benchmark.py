"""
Network-inference benchmark on simulated Perturb-seq screens, scored against two truths.

A knockdown screen measures *total* effects, so a method can be right about the biology and still lose points against a
direct-edge truth, and a method that ranks big total effects first looks perfect against a total-effect truth. The
simulator knows both, so every score matrix is evaluated against

* ``direct``: the true regulatory edges (``screen.adjacency``);
* ``total``: pairs (a, j) whose total effect |log2 FC of gene j when a is perturbed| reaches the screen's effect threshold;

with AUPR and precision at k = number of true pairs. Score matrices are ``S[regulator, target]``; ``evaluate_scores``
ignores the diagonal and rows of genes that were not perturbed.

Inputs a method can be given (``input_kind``): ``truth`` (the noise-free log2 FC of the screen: the upper bound a method
can reach on this network) or ``cells`` (an :class:`~grn_sim.perturbseq.ace.ACEEstimate` computed from the simulated
cells, i.e. what an analyst has).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from grn_sim.evaluation.metrics import compute_aupr
from grn_sim.perturbseq.ace import ACEEstimate, estimate_ace, truth_ace


def truth_matrices(screen: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(direct, total, node_rows)``: boolean [regulator, target] truths and the perturbed-gene mask."""
    n = screen.n_genes
    direct = np.asarray(screen.adjacency) != 0
    total = np.zeros((n, n), dtype=bool)
    rows = np.zeros(n, dtype=bool)
    for k, c in enumerate(screen.conditions):
        if len(c) != 1:
            continue
        g = c[0]
        rows[g] = True
        total[g] = np.abs(screen.total_effects[:, k]) >= screen.effect_threshold
    np.fill_diagonal(direct, False)
    np.fill_diagonal(total, False)
    return direct, total, rows


def _precision_at_k(T: np.ndarray, S: np.ndarray) -> float:
    k = int(T.sum())
    if k == 0:
        return 0.0
    top = np.argsort(-S.ravel(), kind="mergesort")[:k]
    return float(T.ravel()[top].sum() / k)


def evaluate_scores(screen: Any, S: np.ndarray) -> Dict[str, float]:
    """AUPR and precision@k of the score matrix ``S[regulator, target]`` against the direct and the total truth."""
    direct, total, rows = truth_matrices(screen)
    S = np.abs(np.real(np.asarray(S, dtype=float)))
    mask = rows[:, None] & ~np.eye(len(rows), dtype=bool)
    out: Dict[str, float] = {}
    for name, T in (("direct", direct), ("total", total)):
        Tm, Sm = T[mask], S[mask]
        out[f"aupr_{name}"] = compute_aupr(Tm, Sm)
        out[f"p_at_k_{name}"] = _precision_at_k(Tm, Sm)
        out[f"base_{name}"] = float(Tm.mean())
    return out


def screen_inputs(run: Any, input_kind: str = "cells", seed: int = 0, use_called_guide: bool = True,
                  min_cells: int = 20) -> ACEEstimate:
    if input_kind == "truth":
        return truth_ace(run.screen)
    if input_kind == "cells":
        return estimate_ace(run.adata, gene_names=list(run.screen.gene_names), use_called_guide=use_called_guide,
                            min_cells=min_cells, seed=seed)
    raise ValueError("input_kind must be 'truth' or 'cells'")


@dataclass
class Method:
    name: str
    score: Callable[[ACEEstimate, Any], np.ndarray]     # (ace, screen) -> S[regulator, target]


def score_with_algorithm(algo: Any, ace: ACEEstimate, orientation: str = "regulator_target", **fit_kwargs) -> np.ndarray:
    """Runs an (X, U) algorithm on an ACE estimate and returns its score matrix as S[regulator, target]."""
    X, U = ace.to_xu()
    res = algo.fit(X, U, **fit_kwargs)
    A = np.abs(res.A_est)
    return A.T if orientation == "target_regulator" else A


def run_methods(run: Any, methods: List[Method], input_kind: str = "cells", seed: int = 0) -> List[Dict[str, Any]]:
    import time
    ace = screen_inputs(run, input_kind, seed)
    rows = []
    for m in methods:
        t = time.time()
        S = m.score(ace, run.screen)
        row = {"method": m.name, "seconds": round(time.time() - t, 2)}
        row.update(evaluate_scores(run.screen, S))
        rows.append(row)
    return rows
