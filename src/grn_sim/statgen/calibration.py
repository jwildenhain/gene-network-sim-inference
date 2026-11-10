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
    "linear_sem": ModelParams(effect_scale=0.15, noise=0.2),
    "nonlinear_scm": ModelParams(effect_scale=0.6, noise=0.07),
    "nb_copula": ModelParams(effect_scale=0.15, noise=0.2),
    "boolean": ModelParams(effect_scale=1.2, noise=1.0),
}


def params_for(generator: str, noise_factor: float = 1.0) -> ModelParams:
    """The calibrated parameters of ``generator`` with the noise scaled by ``noise_factor`` (0.5 low, 1 medium, 2 high)."""
    if generator not in CALIBRATED:
        raise ValueError(f"generator must be one of {sorted(CALIBRATED)}, got {generator!r}")
    p = CALIBRATED[generator]
    return replace(p, noise=p.noise * noise_factor, shared_noise=p.shared_noise * noise_factor)
