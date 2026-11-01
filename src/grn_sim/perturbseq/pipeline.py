"""
One-call Perturb-seq pipeline: network -> screen (stage 1) -> cells (stage 2) -> measurement (stage 3) -> realism
statistics (stage 4). Shared by the CLI (``grn-sim perturbseq``), the studio API and scripts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from grn_sim.core.network import GeneNetwork
from grn_sim.perturbseq.calibration import PRESETS
from grn_sim.perturbseq.cells import CellConfig, CellPopulation, simulate_cells
from grn_sim.perturbseq.conditions import ConditionScreen, simulate_conditions
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

    def realism(self, phenotype_test: str = "hotelling") -> Dict[str, Any]:
        """Ground-truth + observed statistics and the report rows against the genome-scale targets."""
        t = time.time()
        stats = dict(truth_statistics(self.screen))
        if self.population is not None:
            stats.update(observed_statistics(self.adata, phenotype_test=phenotype_test,
                                             seed=int(self.settings.get("seed") or 0)))
        rows = realism_report(stats)
        self.seconds["statistics"] = round(time.time() - t, 2)
        return {"statistics": stats, "report": rows, "score": realism_score(rows), "settings": self.settings,
                "seconds": dict(self.seconds)}


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
) -> ScreenRun:
    """
    Runs the screen on ``network``.

    ``cell_model``: ``"lna"`` / ``"sde"`` (stage-2 cells) or ``"legacy"`` (independent Gamma-Poisson cells around
    each condition mean, no guides). ``measurement`` applies the stage-3 droplet model to stage-2 cells (doublets,
    ambient RNA, batches, guide calling); without it counts are plain Poisson samples of each cell's composition.
    """
    if cell_model not in CELL_MODELS:
        raise ValueError(f"cell_model must be one of {CELL_MODELS}, got {cell_model!r}")
    if cells_per_condition < 1:
        raise ValueError("cells_per_condition must be >= 1")
    settings = {"mode": mode, "pairs": int(pairs or 0), "cells_per_condition": int(cells_per_condition),
                "n_control_cells": None if n_control_cells is None else int(n_control_cells),
                "cell_model": cell_model, "measurement": bool(measurement), "n_genes": int(network.n),
                "seed": seed, "guide_efficiency_mean": guide_efficiency_mean}
    seconds: Dict[str, float] = {}
    t = time.time()
    screen = simulate_conditions(network, mode=mode, pairs=pairs or None, gene_names=gene_names, seed=seed)
    seconds["screen"] = round(time.time() - t, 2)
    seed_i = int(seed or 0)
    if cell_model == "legacy":
        t = time.time()
        adata = screen_to_anndata(screen, cells_per_condition=cells_per_condition, seed=seed_i)
        seconds["cells"] = round(time.time() - t, 2)
        return ScreenRun(screen, None, adata, settings, seconds)
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
    return ScreenRun(screen, population, adata, settings, seconds)
