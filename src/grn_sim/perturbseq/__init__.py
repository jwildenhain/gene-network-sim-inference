"""
Realistic Perturb-seq data generation on top of the GRN kinetics.

Stage 1 (``conditions``): condition-level steady states for single and double knockdowns/knockouts, plus the ground
truth a benchmark needs (total effects, edge elasticities, linear response, per-perturbation labels, epistasis).
``screen_to_anndata`` turns a screen into a single-cell AnnData with that ground truth in ``uns``.

Stage 2 (``cells``): per-cell perturbation doses (guides, escapers, knockout alleles) and network-propagated
cell-to-cell noise, with a fast linear-noise model (``"lna"``) or exact chemical-Langevin integration (``"sde"``);
``population_to_anndata`` exports the cells.

Stage 3 (``measurement``): doublets, ambient RNA, gene capture bias, batches, UMI sampling and guide capture /
assignment, with the truth recorded next to every observation (``measure_population``).

Stage 4 (``calibration``, ``realism``): anchored Hill kinetics (``calibrate_kinetics``), network presets
(``PRESETS["replogle_like"]``), ground-truth and observed realism statistics scored against genome-scale Perturb-seq
targets (``truth_statistics``, ``observed_statistics``, ``realism_report``); sweep with
``scripts/calibrate_perturbseq.py``.
"""

from grn_sim.perturbseq.conditions import (
    LABELS,
    PERTURBATION_MODES,
    ConditionScreen,
    simulate_conditions,
)
from grn_sim.perturbseq.cells import NOISE_MODELS, CellConfig, CellPopulation, population_to_counts, simulate_cells
from grn_sim.perturbseq.measurement import MeasuredScreen, MeasurementConfig, measure_population
from grn_sim.perturbseq.export import design_matrix, population_to_anndata, screen_to_anndata

__all__ = [
    "MeasuredScreen",
    "MeasurementConfig",
    "measure_population",
    "NOISE_MODELS",
    "CellConfig",
    "CellPopulation",
    "population_to_anndata",
    "population_to_counts",
    "simulate_cells",
    "LABELS",
    "PERTURBATION_MODES",
    "ConditionScreen",
    "design_matrix",
    "screen_to_anndata",
    "simulate_conditions",
]
