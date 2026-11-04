"""
One-call Perturb-seq pipeline: network -> screen (stage 1) -> cells (stage 2) -> measurement (stage 3) -> realism
statistics (stage 4). Shared by the CLI (``grn-sim perturbseq``), the studio API and scripts.
"""

from __future__ import annotations

import time

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from grn_sim.core.network import GeneNetwork
from grn_sim.perturbseq.calibration import PRESETS
from grn_sim.perturbseq.cells import CellConfig, CellPopulation, simulate_cells
from grn_sim.perturbseq.conditions import ConditionScreen, simulate_conditions
from grn_sim.perturbseq.design import ScreenDesign, select_perturbed
from grn_sim.perturbseq.export import population_to_anndata, screen_to_anndata
from grn_sim.perturbseq.measurement import MeasurementConfig
from grn_sim.perturbseq.realism import observed_statistics, realism_report, realism_score, truth_statistics

CELL_MODELS = ("lna", "sde", "legacy")


@dataclass
class ScreenRun:
    """Everything one pipeline run produced."""

    screen: ConditionScreen
    population: Optional[CellPopulation]
    adata: Any
    settings: Dict[str, Any]
    seconds: Dict[str, float] = field(default_factory=dict)
    design: ScreenDesign = field(default_factory=ScreenDesign.full)
    selection_info: Dict[str, Any] = field(default_factory=dict)

    def realism(self, phenotype_test: str = "hotelling") -> Dict[str, Any]:
        """Ground-truth + observed statistics and the report rows against the genome-scale targets."""
        t = time.time()
        stats = dict(truth_statistics(self.screen))
        if self.population is not None:
            stats.update(observed_statistics(self.adata, phenotype_test=phenotype_test,
                                             seed=int(self.settings.get("seed") or 0)))
        partial = self.design.is_partial
        rows = realism_report(stats, partial_selection=partial and self.design.selection == "effective")
        self.seconds["statistics"] = round(time.time() - t, 2)
        out = {"statistics": stats, "report": rows, "score": realism_score(rows), "settings": self.settings,
               "seconds": dict(self.seconds)}
        if partial:
            from grn_sim.perturbseq.partial_stats import design_statistics
            out["design"] = {"describe": self.design.describe(), "selection": dict(self.selection_info),
                             "categories": design_statistics(self.screen, self.selection_info)}
        return out


def _stamp_design(adata: Any, design: ScreenDesign, info: Dict[str, Any], screen: ConditionScreen) -> None:
    """Records the screen design in ``adata.uns['screen_design']`` (scalars and arrays only, so it survives an h5ad round trip)."""
    perturbed = sorted({c[0] for c in screen.conditions if len(c) == 1})
    adata.uns["screen_design"] = {
        "kind": design.kind,
        "label": design.label(),
        "selection_rule": str(info.get("rule", "full")),
        "n_perturbed": int(len(perturbed)),
        "n_requested": int(info.get("n_requested", len(perturbed))),
        "shortfall": int(info.get("shortfall", 0)),
        "perturbed_genes": np.asarray([screen.gene_names[g] for g in perturbed], dtype=object),
        "perturbed_gene_index": np.asarray(perturbed, dtype=int),
    }


def build_network(preset: str, n_genes: int, seed: Optional[int] = None) -> GeneNetwork:
    """A network of ``n_genes`` genes from a named preset (see :data:`grn_sim.perturbseq.calibration.PRESETS`)."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    return PRESETS[preset].build(n_genes, seed=seed)


def run_screen(
    network: GeneNetwork,
    mode: str = "knockdown",
    pairs: Optional[int] = None,
    cells_per_condition: int = 100,
    n_control_cells: Optional[int] = None,
    cell_model: str = "lna",
    measurement: bool = True,
    gene_names: Optional[List[str]] = None,
    guide_efficiency_mean: Optional[float] = None,
    seed: Optional[int] = 0,
    genes: Optional[List[int]] = None,
    n_perturbed: Optional[int] = None,
    perturbed_selection: str = "effective",
    design: Optional[ScreenDesign] = None,
) -> ScreenRun:
    """
    Runs the screen on ``network``.

    Design: by default every gene is perturbed once. ``design`` (a :class:`ScreenDesign`) or the shortcuts ``n_perturbed`` /
    ``genes`` make a *partial* screen: only those genes are perturbed, all genes are measured. ``n_perturbed`` picks that many
    genes by ``perturbed_selection``: ``"effective"`` (default; genes whose knockdown moves other genes in this network) or
    ``"random"``; ``genes`` lists the indices explicitly. Partial screens report their own statistics category
    (``ScreenRun.realism()["design"]``) and are scored per block (``grn_sim.evaluation.partial_screen``).

    ``cell_model``: ``"lna"`` / ``"sde"`` (stage-2 cells) or ``"legacy"`` (independent Gamma-Poisson cells around
    each condition mean, no guides). ``measurement`` applies the stage-3 droplet model to stage-2 cells (doublets,
    ambient RNA, batches, guide calling); without it counts are plain Poisson samples of each cell's composition.
    """
    if cell_model not in CELL_MODELS:
        raise ValueError(f"cell_model must be one of {CELL_MODELS}, got {cell_model!r}")
    if cells_per_condition < 1:
        raise ValueError("cells_per_condition must be >= 1")
    if design is None:
        design = ScreenDesign.from_options(n_perturbed, perturbed_selection, genes)
    genes, selection_info = select_perturbed(network, design, mode=mode, seed=seed)
    if design.is_partial and not genes:
        raise ValueError("this network has no effective genes to perturb (no knockdown moves another gene)")
    settings = {"mode": mode, "pairs": int(pairs or 0), "cells_per_condition": int(cells_per_condition),
                "n_control_cells": None if n_control_cells is None else int(n_control_cells),
                "cell_model": cell_model, "measurement": bool(measurement), "n_genes": int(network.n),
                "seed": seed, "guide_efficiency_mean": guide_efficiency_mean,
                "n_perturbed": network.n if genes is None else len(genes), "design": design.label(),
                "perturbed_selection": selection_info.get("rule"), "perturbed_selection_shortfall": selection_info.get("shortfall", 0),
                "perturbed_genes": None if genes is None else [int(g) for g in genes]}
    seconds: Dict[str, float] = {}
    t = time.time()
    screen = simulate_conditions(network, mode=mode, genes=genes, pairs=pairs or None, gene_names=gene_names, seed=seed)
    seconds["screen"] = round(time.time() - t, 2)
    seed_i = int(seed or 0)
    if cell_model == "legacy":
        t = time.time()
        adata = screen_to_anndata(screen, cells_per_condition=cells_per_condition, seed=seed_i)
        seconds["cells"] = round(time.time() - t, 2)
        _stamp_design(adata, design, selection_info, screen)
        return ScreenRun(screen, None, adata, settings, seconds, design, selection_info)
    cfg_kwargs: Dict[str, Any] = {"noise_model": cell_model}
    if guide_efficiency_mean is not None:
        cfg_kwargs["guide_efficiency_mean"] = float(guide_efficiency_mean)
    t = time.time()
    population = simulate_cells(network, screen, cells_per_condition=cells_per_condition,
                                n_control_cells=n_control_cells, config=CellConfig(**cfg_kwargs), seed=seed_i + 1)
    seconds["cells"] = round(time.time() - t, 2)
    t = time.time()
    adata = population_to_anndata(population, screen, seed=seed_i + 2,
                                  measurement=MeasurementConfig() if measurement else None)
    seconds["measurement"] = round(time.time() - t, 2)
    _stamp_design(adata, design, selection_info, screen)
    return ScreenRun(screen, population, adata, settings, seconds, design, selection_info)
