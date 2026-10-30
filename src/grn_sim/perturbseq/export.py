"""AnnData export of a :class:`ConditionScreen` (condition means turned into single-cell counts, ground truth in ``uns``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from grn_sim.perturbseq.conditions import ConditionScreen
from grn_sim.simulator.perturb_seq import create_perturb_seq_anndata


def design_matrix(screen: ConditionScreen) -> np.ndarray:
    """(n_genes, n_conditions) perturbation design: ``-efficiency`` (or ``+factor`` for input) on the perturbed genes."""
    U = np.zeros((screen.n_genes, len(screen.conditions)), dtype=float)
    sign = 1.0 if screen.mode == "input" else -1.0
    for k, c in enumerate(screen.conditions):
        U[list(c), k] = sign * screen.efficiency[list(c)]
    return U


def screen_to_anndata(
    screen: ConditionScreen,
    cells_per_condition: int = 400,
    mean_library_depth: int = 20000,
    seed: int = 42,
    as_sparse: bool = True,
) -> Any:
    """Single-cell counts for the control plus every screen condition, with the screen's ground truth attached."""
    adata = create_perturb_seq_anndata(
        gene_names=screen.gene_names,
        mean_expression_matrix=np.column_stack([screen.reference, screen.means]),
        ground_truth_network=screen.adjacency,
        perturbation_matrix=design_matrix(screen),
        cells_per_condition=cells_per_condition,
        mean_library_depth=mean_library_depth,
        seed=seed,
        as_sparse=as_sparse,
        condition_names=screen.condition_names,
        ground_truth=screen.to_uns(),
    )
    summary = screen.summary()
    summary.pop("label_counts", None)
    if isinstance(adata, dict):
        adata["screen_summary"] = summary
    else:
        adata.uns["screen_summary"] = summary
        adata.uns["vcc_challenge"]["perturbation_mode"] = screen.mode
    return adata
