"""AnnData export of a :class:`ConditionScreen` (condition means turned into single-cell counts, ground truth in ``uns``)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from grn_sim.perturbseq.cells import CellPopulation, population_to_counts
from grn_sim.perturbseq.conditions import ConditionScreen
from grn_sim.perturbseq.measurement import MeasurementConfig, measure_population
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


def population_to_anndata(
    population: CellPopulation,
    screen: ConditionScreen,
    mean_library_depth: int = 20000,
    depth_dispersion: float = 0.35,
    seed: Optional[int] = None,
    measurement: Optional[MeasurementConfig] = None,
) -> Any:
    """
    AnnData of simulated cells: UMI counts in ``X`` (sparse), per-cell truth in ``obs`` (condition, guide, realised
    dose, escaped) and the screen's ground truth in ``uns["ground_truth"]``; the expression before measurement is kept
    in ``layers["true_expression"]``.

    With ``measurement`` (stage 3) the counts come from :func:`measure_population`: doublets, ambient RNA, gene capture
    bias, batches, NB sampling, and guide UMIs (``obsm["guide_counts"]``, names in ``uns["guide_names"]``) with
    guide calling; ``obs`` then also holds ``batch``, ``is_doublet``, ``assigned_guide``, ``assignment_status`` and
    whether the call is correct. ``mean_library_depth`` / ``depth_dispersion`` apply only without ``measurement``.
    """
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    obs = pd.DataFrame(population.obs_records())
    obs["perturbation"] = ["non-targeting" if c == "control" else c for c in population.condition]
    obs["is_control"] = [c == "control" for c in population.condition]
    obsm: dict = {}
    uns_extra: dict = {}
    if measurement is None:
        counts_dense, depth = population_to_counts(population, mean_library_depth, depth_dispersion, seed)
        X = sparse.csr_matrix(counts_dense)
        true_expr = population.expression
        obs["library_size"] = depth
    else:
        measured = measure_population(population, measurement, seed)
        X = measured.counts
        true_expr = measured.true_expression
        extra = pd.DataFrame(measured.obs)
        for col in extra.columns:
            obs[col] = extra[col].values
        obsm["guide_counts"] = measured.guide_counts
        uns_extra["guide_names"] = np.asarray(measured.guide_names, dtype=object)
        uns_extra["measurement"] = {k: v for k, v in vars(measurement).items()}
        uns_extra["gene_capture"] = measured.gene_capture
        uns_extra["batch_shift"] = measured.batch_shift
    obs["total_counts"] = np.asarray(X.sum(axis=1)).ravel()
    obs["detected_genes"] = np.asarray((X > 0).sum(axis=1)).ravel()
    obs.index = pd.Index([f"cell_{i:07d}" for i in range(population.n_cells)])
    var = pd.DataFrame({"gene_name": population.gene_names}, index=population.gene_names)
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
        obsm=obsm,
        layers={"true_expression": np.asarray(true_expr, dtype=np.float32)},
    )
    adata.uns.update(uns_extra)
    adata.uns["ground_truth"] = screen.to_uns()
    summary = screen.summary()
    summary.pop("label_counts", None)
    adata.uns["screen_summary"] = summary
    cfg = population.meta.get("config")
    adata.uns["cell_model"] = {
        "noise_model": population.noise_model,
        "mode": screen.mode,
        "mean_library_depth": int(mean_library_depth),
        **({k: (list(v) if isinstance(v, tuple) else ("none" if v is None else v)) for k, v in vars(cfg).items()} if cfg else {}),
    }
    return adata
