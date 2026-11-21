"""The simplified SCEPTRE test (conditional resampling on library size) of ``estimate_ace``."""
import warnings

import numpy as np
import pytest

from grn_sim.perturbseq.ace import estimate_ace
from grn_sim.perturbseq.pipeline import build_network, run_screen


@pytest.fixture(scope="module")
def screens():
    warnings.simplefilter("ignore")
    run = run_screen(build_network("replogle_like", 30, seed=3), mode="knockdown", cells_per_condition=60, seed=3)
    names = list(run.screen.gene_names)
    welch = estimate_ace(run.adata, gene_names=names, use_called_guide=True, min_cells=20, seed=1, test="welch")
    sce = estimate_ace(run.adata, gene_names=names, use_called_guide=True, min_cells=20, seed=1, test="sceptre", sceptre_draws=100)
    return welch, sce


def test_effect_sizes_are_those_of_the_welch_estimate(screens):
    welch, sce = screens
    assert np.allclose(welch.lfc, sce.lfc) and np.allclose(welch.se, sce.se)
    assert sce.z.shape == welch.z.shape and np.all(np.isfinite(sce.z))


def test_z_scores_agree_with_welch_and_are_calibrated_off_target(screens):
    welch, sce = screens
    m = np.ones(sce.z.shape, dtype=bool)
    m[np.arange(len(sce.nodes)), sce.nodes] = False
    assert np.corrcoef(welch.z[m], sce.z[m])[0, 1] > 0.6
    assert 0.5 < np.median(np.abs(sce.z[m])) / 0.6745 < 3.0          # a z-score scale, not collapsed to 0 or inflated


def test_unknown_test_name_is_rejected():
    with pytest.raises(ValueError):
        estimate_ace(None, test="nope")


def test_the_screen_benchmark_passes_the_test_choice_through():
    from grn_sim.evaluation import screen_benchmark as sb

    warnings.simplefilter("ignore")
    run = run_screen(build_network("replogle_like", 20, seed=4), mode="knockdown", cells_per_condition=40, seed=4)
    welch = sb.screen_inputs(run, "cells", 1, min_cells=15)
    other = sb.screen_inputs(run, "cells", 1, min_cells=15, test="sceptre")
    assert np.allclose(welch.lfc, other.lfc) and not np.allclose(welch.z, other.z)
    with pytest.raises(ValueError):
        sb.screen_inputs(run, "cells", 1, test="nope")
