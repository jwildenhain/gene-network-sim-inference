"""Stage 2 of the Perturb-seq generator: per-cell doses, escapers, network-propagated noise (LNA and SDE)."""

import numpy as np
import pytest

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.perturbseq import CellConfig, population_to_anndata, simulate_cells, simulate_conditions


def network(n=10, seed=4):
    return GeneNetwork.create(n=n, k_av=2.0, net_type=TopologyType.MODULAR, subdiv=2, seed=seed)


def cells_of(pop, condition):
    return np.array([c == condition for c in pop.condition])


QUIET = dict(extrinsic_gene_sd=0.0, extrinsic_shared_sd=0.0)


@pytest.mark.parametrize("kwargs", [
    {"noise_model": "gillespie"}, {"guides_per_gene": 0}, {"guide_efficiency_mean": 1.0}, {"escape_rate": 1.5},
    {"dose_levels": (0.2, 1.0)}, {"dose_levels": (0.0, 0.5, 0.4, 1.0)}, {"omega": 0.0}, {"extrinsic_gene_sd": -1},
    {"blas_threads": 0},
])
def test_invalid_cell_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CellConfig(**kwargs).validate()


def test_population_layout_controls_guides_and_escapers():
    net = network()
    screen = simulate_conditions(net, mode="knockdown", pairs=2, seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=400, n_control_cells=300, config=CellConfig(escape_rate=0.2), seed=1)
    assert pop.n_cells == 300 + 400 * len(screen.conditions)
    assert pop.expression.shape == (pop.n_cells, 10) and np.all(pop.expression >= 0)
    ctrl = cells_of(pop, "control")
    assert ctrl.sum() == 300 and all(g.startswith("non-targeting_") for g, c in zip(pop.guide, ctrl) if c)
    single = np.array([len(t) == 1 for t in pop.targets])
    assert abs(pop.escaped[single].mean() - 0.2) < 0.03
    assert np.all(pop.dose[single & pop.escaped, 0] == 0.0)
    guides = {g for g, t in zip(pop.guide, pop.targets) if t == (3,)}
    assert guides == {"Gene_004_g1", "Gene_004_g2"}


def test_own_gene_follows_the_cell_dose():
    net = network()
    screen = simulate_conditions(net, mode="knockdown", seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=300, config=CellConfig(escape_rate=0.1, **QUIET), seed=2)
    for g in (0, 5):
        m = np.array([t == (g,) for t in pop.targets])
        dose, own = pop.dose[m, 0], pop.expression[m, g]
        order = np.argsort(dose)
        assert np.all(np.diff(own[order]) <= 1e-9), "less expression at a higher knockdown dose"
        escaped = pop.escaped[m]
        assert np.allclose(own[escaped], screen.reference[g])


def test_knockout_alleles():
    net = network()
    screen = simulate_conditions(net, mode="knockout", seed=0)
    cfg = CellConfig(escape_rate=0.1, knockout_biallelic_rate=0.7, **QUIET)
    pop = simulate_cells(net, screen, cells_per_condition=500, config=cfg, seed=3)
    m = np.array([t == (2,) for t in pop.targets])
    dose = pop.dose[m, 0]
    assert set(np.unique(dose)) <= {0.0, 0.5, 1.0}
    assert abs(np.mean(dose == 1.0) - 0.9 * 0.7) < 0.06
    assert np.all(pop.expression[m][dose == 1.0, 2] == 0.0)


def test_without_noise_and_escape_cells_reproduce_the_condition_means():
    net = network()
    screen = simulate_conditions(net, mode="knockdown", efficiency=0.8, seed=0)
    cfg = CellConfig(escape_rate=0.0, guide_efficiency_concentration=1e6, cell_efficiency_concentration=1e6, **QUIET)
    pop = simulate_cells(net, screen, cells_per_condition=20, config=cfg, seed=4)
    for k, name in enumerate(screen.condition_names):
        assert np.allclose(pop.expression[cells_of(pop, name)].mean(axis=0), screen.means[:, k], rtol=5e-3, atol=1e-6)


def test_doses_between_grid_levels_are_interpolated_accurately():
    # Grid: (0, 0.5, 0.75, 0.8 [the screen's nominal], 0.9, 1). Cells at 0.83 fall between grid levels; compare with
    # an exact solve at 0.83. Interpolation error stays far below the cell-to-cell noise (~15%).
    net = network()
    screen = simulate_conditions(net, mode="knockdown", efficiency=0.8, seed=0)
    exact = simulate_conditions(net, mode="knockdown", efficiency=0.83, seed=0)
    cfg = CellConfig(escape_rate=0.0, guide_efficiency_mean=0.83, guide_efficiency_concentration=1e6,
                     cell_efficiency_concentration=1e6, **QUIET)
    pop = simulate_cells(net, screen, cells_per_condition=5, config=cfg, seed=4)
    for k, name in enumerate(screen.condition_names):
        got = pop.expression[cells_of(pop, name)].mean(axis=0)
        assert np.allclose(got, exact.means[:, k], rtol=0.05, atol=1e-4)


def test_noise_is_correlated_along_the_network():
    net = GeneNetwork.create(n=30, k_av=2.0, net_type=TopologyType.MODULAR, subdiv=3, seed=6)
    screen = simulate_conditions(net, mode="knockdown", seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=5, n_control_cells=3000,
                         config=CellConfig(extrinsic_shared_sd=0.0), seed=5)
    L = np.log(pop.expression[cells_of(pop, "control")])
    C = np.abs(np.corrcoef(L.T))
    A = net.topology.adjacency != 0
    linked = A | A.T
    np.fill_diagonal(linked, False)
    unlinked = ~linked
    np.fill_diagonal(unlinked, False)
    assert np.nanmean(C[linked]) > 2 * np.nanmean(C[unlinked])


def test_lna_and_sde_agree_for_well_expressed_genes():
    net = network(8, seed=4)
    screen = simulate_conditions(net, mode="knockdown", seed=0)
    stats = {}
    for model in ("lna", "sde"):
        pop = simulate_cells(net, screen, cells_per_condition=2, n_control_cells=1500,
                             config=CellConfig(noise_model=model, omega=500.0, extrinsic_gene_sd=0.1), seed=7)
        stats[model] = np.log(np.maximum(pop.expression[cells_of(pop, "control")], 1e-9))
    keep = screen.reference * 500.0 > 100  # >100 molecules: where the linear-noise approximation is meant to hold
    a, b = stats["lna"][:, keep], stats["sde"][:, keep]
    assert np.all(np.abs(a.mean(0) - b.mean(0)) < 0.1)
    ratio = a.std(0) / b.std(0)
    assert np.all((ratio > 0.7) & (ratio < 1.4))
    iu = np.triu_indices(keep.sum(), 1)
    assert np.corrcoef(np.corrcoef(a.T)[iu], np.corrcoef(b.T)[iu])[0, 1] > 0.8


def test_same_seed_same_cells():
    net = network()
    screen = simulate_conditions(net, mode="knockdown", pairs=1, seed=0)
    a = simulate_cells(net, screen, cells_per_condition=20, seed=9)
    b = simulate_cells(net, screen, cells_per_condition=20, seed=9)
    assert np.array_equal(a.expression, b.expression) and a.guide == b.guide


def test_population_exports_to_anndata(tmp_path):
    anndata = pytest.importorskip("anndata")
    net = network()
    screen = simulate_conditions(net, mode="knockdown", pairs=1, seed=0)
    pop = simulate_cells(net, screen, cells_per_condition=15, seed=1)
    adata = population_to_anndata(pop, screen, mean_library_depth=5000, seed=2)
    assert adata.n_obs == pop.n_cells and adata.n_vars == 10
    assert {"condition", "guide", "dose_1", "dose_2", "escaped", "perturbation", "is_control"} <= set(adata.obs.columns)
    assert adata.layers["true_expression"].shape == (pop.n_cells, 10)
    path = tmp_path / "cells.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    assert back.uns["cell_model"]["noise_model"] == "lna"
    assert np.allclose(back.uns["ground_truth"]["total_effects_log2"], screen.total_effects)


def test_server_exports_cells_with_the_cell_model():
    pytest.importorskip("anndata")
    from grn_sim.web import server

    server._execute_simulation({"num_genes": 8, "topology": "modular", "seed": 7, "auto_infer": False})
    data, filename = server._build_perturb_seq_h5ad_bytes(cells_per_condition=4, noise_model="lna")
    assert filename.endswith("_lna_4cells.h5ad") and len(data) > 0
    with pytest.raises(ValueError):
        server._parse_perturb_seq_options({"noise": ["gillespie"]})
