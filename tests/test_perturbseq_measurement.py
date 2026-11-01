"""Stage 3 of the Perturb-seq generator: doublets, ambient RNA, capture bias, batches, UMI and guide sampling."""

import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.perturbseq import (
    MeasurementConfig,
    measure_population,
    population_to_anndata,
    simulate_cells,
    simulate_conditions,
)


@pytest.fixture(scope="module")
def screen_and_cells():
    net = GeneNetwork.create(n=20, k_av=2.0, net_type=TopologyType.MODULAR, subdiv=2, seed=3)
    screen = simulate_conditions(net, mode="knockdown", pairs=2, seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=150, seed=1)
    return screen, pop


def column(measured, key):
    return np.array([o[key] for o in measured.obs])


@pytest.mark.parametrize("kwargs", [
    {"mean_library_depth": 0}, {"n_batches": 0}, {"dispersion": -1}, {"doublet_rate": 1.0},
    {"ambient_fraction": -0.1}, {"assign_min_umis": 0}, {"ambient_concentration": 0},
])
def test_invalid_measurement_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MeasurementConfig(**kwargs).validate()


def test_doublets_combine_two_cells(screen_and_cells):
    _, pop = screen_and_cells
    m = measure_population(pop, MeasurementConfig(doublet_rate=0.1), seed=2)
    dbl = column(m, "is_doublet")
    assert abs(dbl.mean() - 0.1) < 0.02
    partner = column(m, "doublet_partner")
    d = np.flatnonzero(dbl)[0]
    assert partner[d] != d and np.allclose(m.true_expression[d], pop.expression[d] + pop.expression[partner[d]])
    assert np.all(partner[~dbl] == -1)
    umis = np.asarray(m.counts.sum(axis=1)).ravel()
    assert 1.6 < umis[dbl].mean() / umis[~dbl].mean() < 2.4


def test_without_artefacts_counts_follow_the_cell_composition(screen_and_cells):
    _, pop = screen_and_cells
    cfg = MeasurementConfig(doublet_rate=0.0, ambient_fraction=0.0, gene_capture_sd=0.0, batch_gene_sd=0.0,
                            batch_depth_sd=0.0, dispersion=0.0, depth_sd=0.0, mean_library_depth=1e6)
    m = measure_population(pop, cfg, seed=3)
    comp = pop.expression / pop.expression.sum(axis=1, keepdims=True)
    observed = m.counts.toarray() / 1e6
    assert np.allclose(observed, comp, atol=3e-3)


def test_ambient_rna_leaks_soup_into_every_droplet(screen_and_cells):
    _, pop = screen_and_cells
    base = dict(doublet_rate=0.0, gene_capture_sd=0.0, batch_gene_sd=0.0, batch_depth_sd=0.0, dispersion=0.0)
    clean = measure_population(pop, MeasurementConfig(ambient_fraction=0.0, **base), seed=4)
    dirty = measure_population(pop, MeasurementConfig(ambient_fraction=0.2, **base), seed=4)
    assert abs(column(dirty, "ambient_fraction").mean() - 0.2) < 0.02
    # Fully knocked-out / silenced genes pick up reads from the soup.
    zero = pop.expression <= 1e-12
    if zero.any():
        assert dirty.counts.toarray()[zero].mean() > clean.counts.toarray()[zero].mean()


def test_batches_shift_genes_and_hold_controls(screen_and_cells):
    _, pop = screen_and_cells
    cfg = MeasurementConfig(n_batches=3, batch_gene_sd=0.5, doublet_rate=0.0)
    m = measure_population(pop, cfg, seed=5)
    batch = column(m, "batch")
    assert set(batch) == {"batch_1", "batch_2", "batch_3"}
    ctrl = np.array([c == "control" for c in pop.condition])
    assert all((batch[ctrl] == b).any() for b in ("batch_1", "batch_2", "batch_3"))
    X = m.counts.toarray().astype(float)
    norm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
    means = np.vstack([norm[ctrl & (batch == b)].mean(axis=0) for b in ("batch_1", "batch_2")])
    shift = m.batch_shift[0] - m.batch_shift[1]
    expressed = pop.expression[ctrl].mean(axis=0) > np.median(pop.expression[ctrl].mean(axis=0))
    assert np.corrcoef(means[0, expressed] - means[1, expressed], shift[expressed])[0, 1] > 0.6


def test_guide_calls_are_mostly_right_and_flag_doublets(screen_and_cells):
    _, pop = screen_and_cells
    m = measure_population(pop, MeasurementConfig(doublet_rate=0.08), seed=6)
    dbl = column(m, "is_doublet")
    status = column(m, "assignment_status")
    assert set(status) <= {"single", "unassigned", "multiple"}
    assert column(m, "assignment_correct")[~dbl].mean() > 0.85
    single_target = np.array([len(t) <= 1 for t in pop.targets])
    assert (status[dbl] == "multiple").mean() > 0.5
    assert (status[~dbl & single_target] == "multiple").mean() < 0.05
    pair = np.array([len(t) == 2 for t in pop.targets])
    assert (status[~dbl & pair] == "multiple").mean() > 0.7, "double perturbations legitimately carry two guides"
    assert m.guide_counts.shape == (pop.n_cells, len(m.guide_names))


def test_more_guide_dropout_means_more_unassigned_cells(screen_and_cells):
    _, pop = screen_and_cells
    low = measure_population(pop, MeasurementConfig(guide_dropout=0.0), seed=7)
    high = measure_population(pop, MeasurementConfig(guide_dropout=0.5), seed=7)
    assert (column(high, "assignment_status") == "unassigned").mean() > (column(low, "assignment_status") == "unassigned").mean() + 0.2


def test_measurement_is_reproducible(screen_and_cells):
    _, pop = screen_and_cells
    a = measure_population(pop, seed=8)
    b = measure_population(pop, seed=8)
    assert (a.counts != b.counts).nnz == 0 and (a.guide_counts != b.guide_counts).nnz == 0


def test_measured_population_exports_to_anndata(tmp_path, screen_and_cells):
    anndata = pytest.importorskip("anndata")
    screen, pop = screen_and_cells
    adata = population_to_anndata(pop, screen, seed=9, measurement=MeasurementConfig())
    assert {"batch", "is_doublet", "assigned_guide", "assignment_status", "assignment_correct"} <= set(adata.obs.columns)
    assert adata.obsm["guide_counts"].shape == (adata.n_obs, len(adata.uns["guide_names"]))
    path = tmp_path / "measured.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert back.uns["measurement"]["doublet_rate"] == pytest.approx(0.04)
    assert np.array_equal(back.X.toarray(), adata.X.toarray())
