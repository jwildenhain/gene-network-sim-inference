"""
Calibrated parameters of the statistical generators and the noise levels built on them.

Chosen with ``scripts/calibrate_statgen.py`` (100-gene modular graphs, CRISPRi screens with 40 cells per perturbation and the
measurement layer, two seeds per grid point): each generator's ``medium`` parameters put the observed screen statistics inside
the genome-scale Perturb-seq ranges of :data:`grn_sim.perturbseq.realism.TARGETS` (on-target knockdown 0.80-0.95, share of
perturbations with a phenotype 0.25-0.45, affected-pair density 0.015-0.05, top-5% effect share >= 0.25, control false
positives <= 5%); the genome-scale share of indirect effects is left out (no 100-gene network can reach it). Where a range of
values met the targets, the middle of the range was taken, not the edge of the grid.

``low`` and ``high`` halve and double the noise (cell-to-cell noise and the shared cell factor) and keep the effect sizes, so
the three levels differ in signal-to-noise only. The calibration log, statistics per grid point and the reasons for each
choice are in ``docs/results/statgen_calibration.md``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from grn_sim.statgen.models import ModelParams

CALIBRATED: Dict[str, ModelParams] = {
    # second tuning round (effect scale x noise x weight spread, 4 seeds per grid point): heavier-tailed edge weights for the
    # top-5% effect share, larger effects to keep phenotype share and pair density inside their ranges
    "linear_sem": ModelParams(effect_scale=0.25, noise=0.2, weight_spread=1.7),
    "nonlinear_scm": ModelParams(effect_scale=0.6, noise=0.03, weight_spread=1.4),
    "nb_copula": ModelParams(effect_scale=0.25, noise=0.2, weight_spread=1.7),
    "boolean": ModelParams(effect_scale=1.2, noise=1.0, weight_spread=1.7),
}


def params_for(generator: str, noise_factor: float = 1.0) -> ModelParams:
    """The calibrated parameters of ``generator`` with the noise scaled by ``noise_factor`` (0.5 low, 1 medium, 2 high)."""
    if generator not in CALIBRATED:
        raise ValueError(f"generator must be one of {sorted(CALIBRATED)}, got {generator!r}")
    p = CALIBRATED[generator]
    return replace(p, noise=p.noise * noise_factor, shared_noise=p.shared_noise * noise_factor)
