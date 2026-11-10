"""
Pooled CRISPR screens from the statistical models: conditions, ground truth, cells and the measurement layer.

The output is the same :class:`grn_sim.perturbseq.pipeline.ScreenRun` the mechanistic pipeline returns (a
``ConditionScreen`` with the truth, a ``CellPopulation`` and an AnnData), so every estimator, scorer and realism statistic
works on both kinds of data. Only the source of the expression differs: here a statistical model of the graph, there Hill
kinetics solved to a steady state. The guide / dose model (guide efficiencies, escapers, mono- and biallelic knockouts) and
the droplet measurement layer (doublets, ambient RNA, capture, batches, guide calling) are shared.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grn_sim.perturbseq.cells import CellConfig, CellPopulation
from grn_sim.perturbseq.conditions import ConditionScreen, _select_pairs
from grn_sim.perturbseq.design import ScreenDesign
from grn_sim.perturbseq.export import population_to_anndata
from grn_sim.perturbseq.measurement import MeasurementConfig
from grn_sim.statgen.models import ModelParams, StatModel, make_model

LOG2E = 1.0 / np.log(2.0)


def _beta(rng: np.random.Generator, mean: np.ndarray, concentration: float) -> np.ndarray:
    m = np.clip(mean, 1e-3, 1 - 1e-3)
    return rng.beta(m * concentration, (1.0 - m) * concentration)


def _nominal(mode: str, cfg: CellConfig) -> float:
    return 1.0 if mode == "knockout" else cfg.guide_efficiency_mean


def _labels(model: StatModel, genes: Sequence[int], lfc: np.ndarray, threshold: float) -> List[str]:
    out = []
    for k, g in enumerate(genes):
        off = np.delete(np.abs(lfc[:, k]), g)
        if not np.any(model.adjacency[g, :] != 0):
            out.append("no_targets")
        elif off.size and off.max() >= threshold:
            out.append("effective")
        else:
            out.append("no_downstream_effect")
    return out


def condition_screen(model: StatModel, genes: Optional[Sequence[int]] = None, pairs: Any = None,
                     cfg: Optional[CellConfig] = None, effect_threshold: float = 0.1, seed: Optional[int] = 0,
                     gene_names: Optional[List[str]] = None) -> ConditionScreen:
    """Noise-free screen of ``model``: one single condition per gene of ``genes`` (all by default) plus ``pairs`` doubles."""
    cfg = cfg or CellConfig()
    n = model.n
    names = list(gene_names) if gene_names else [f"G{i + 1}" for i in range(n)]
    genes = list(range(n)) if genes is None else [int(g) for g in genes]
    dose = _nominal(model.mode, cfg)
    ref = np.log(model.baseline)
    singles = model.log_mean((), np.zeros((1, 0)))[0]
    cols, conditions = [], []
    for g in genes:
        cols.append(model.log_mean((g,), np.array([[dose]]))[0])
        conditions.append((g,))
    lfc_single = (np.column_stack(cols) - singles[:, None]) * LOG2E if cols else np.zeros((n, 0))
    labels = _labels(model, genes, lfc_single, effect_threshold)
    rng = np.random.default_rng(int(seed or 0) + 4049)
    pool = [g for g, lab in zip(genes, labels) if lab == "effective"]
    pair_list = _select_pairs(pairs, pool if len(pool) >= 2 else genes, rng)
    single_of = {g: k for k, g in enumerate(genes)}
    dev_cols, eps = [], []
    for a, b in pair_list:
        z = model.log_mean((a, b), np.array([[dose, dose]]))[0]
        cols.append(z)
        conditions.append((a, b))
        dual = (z - singles) * LOG2E
        la = lfc_single[:, single_of[a]] if a in single_of else (model.log_mean((a,), np.array([[dose]]))[0] - singles) * LOG2E
        lb = lfc_single[:, single_of[b]] if b in single_of else (model.log_mean((b,), np.array([[dose]]))[0] - singles) * LOG2E
        d = dual - (la + lb)
        d[[a, b]] = 0.0
        dev_cols.append(d)
        eps.append(float(np.linalg.norm(d)))
    Z = np.column_stack(cols) if cols else np.zeros((n, 0))
    means = np.exp(ref[:, None] + Z - singles[:, None])
    total = (Z - singles[:, None]) * LOG2E
    return ConditionScreen(
        gene_names=names, mode=model.mode, efficiency=np.full(n, dose), reference=np.exp(ref), conditions=conditions,
        means=means, steady=np.ones(len(conditions), dtype=bool), total_effects=total,
        adjacency=np.sign(np.asarray(model.adjacency)).astype(float), elasticity=model.local_effects(),
        linear_response=model.linear_response(), labels=labels, effect_threshold=effect_threshold,
        epistasis=np.asarray(eps) if eps else None, pair_deviation=np.column_stack(dev_cols) if dev_cols else None,
        meta={"generator": model.name, "params": vars(model.params), "seed": seed})


def simulate_cells(model: StatModel, screen: ConditionScreen, cells_per_condition: int = 40,
                   n_control_cells: Optional[int] = None, cfg: Optional[CellConfig] = None,
                   seed: Optional[int] = 0) -> CellPopulation:
    """Cells of every condition of ``screen``: per-cell guide doses (shared dose model) through the statistical model."""
    cfg = cfg or CellConfig()
    rng = np.random.default_rng(seed)
    n, names, mode = screen.n_genes, screen.gene_names, screen.mode
    knockout = mode == "knockout"
    guide_mean = _beta(rng, np.full((n, cfg.guides_per_gene), cfg.guide_efficiency_mean), cfg.guide_efficiency_concentration)

    def doses(gene: int, g_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = g_idx.size
        escaped = rng.uniform(size=k) < cfg.escape_rate
        if knockout:
            d = np.where(rng.uniform(size=k) < cfg.knockout_biallelic_rate, 1.0, 0.5)
        else:
            d = _beta(rng, guide_mean[gene, g_idx], cfg.cell_efficiency_concentration)
        return np.where(escaped, 0.0, d), escaped

    blocks, cond, guide, targets, dose_rows, esc_rows = [], [], [], [], [], []
    n_ctrl = cells_per_condition if n_control_cells is None else int(n_control_cells)
    if n_ctrl:
        g_idx = rng.integers(0, cfg.n_control_guides, size=n_ctrl)
        blocks.append(model.sample((), np.zeros((n_ctrl, 0)), rng))
        cond += ["control"] * n_ctrl
        guide += [f"non-targeting_{i + 1}" for i in g_idx]
        targets += [()] * n_ctrl
        dose_rows.append(np.zeros((n_ctrl, 2)))
        esc_rows.append(np.zeros(n_ctrl, dtype=bool))
    k = cells_per_condition
    for c_idx, tg in enumerate(screen.conditions):
        cols, esc = [], np.ones(k, dtype=bool)
        idxs = []
        for g in tg:
            g_idx = rng.integers(0, cfg.guides_per_gene, size=k)
            d, e = doses(g, g_idx)
            cols.append(d)
            esc &= e
            idxs.append(g_idx)
        D = np.column_stack(cols)
        blocks.append(model.sample(tg, D, rng))
        cond += [screen.condition_names[c_idx]] * k
        guide += ["+".join(f"{names[g]}_g{int(i) + 1}" for g, i in zip(tg, row)) for row in zip(*idxs)]
        targets += [tuple(tg)] * k
        padded = np.zeros((k, 2))
        padded[:, :D.shape[1]] = D
        dose_rows.append(padded)
        esc_rows.append(esc)
    return CellPopulation(gene_names=list(names), expression=np.vstack(blocks), condition=cond, guide=guide, targets=targets,
                          dose=np.vstack(dose_rows), escaped=np.concatenate(esc_rows), noise_model=f"statgen:{model.name}",
                          meta={"config": cfg, "seed": seed, "mode": mode, "generator": model.name})


def _select(model: StatModel, design: ScreenDesign, cfg: CellConfig, seed: Optional[int]) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """The perturbed genes of a design, chosen on the statistical model itself (same rules as the mechanistic screens)."""
    n = model.n
    design.validate(n)
    if not design.is_partial:
        return None, {"rule": "full", "n_requested": n, "n_selected": n, "n_effective_available": None, "shortfall": 0}
    if design.genes is not None:
        return sorted(design.genes), {"rule": "explicit", "n_requested": len(design.genes), "n_selected": len(design.genes),
                                      "n_effective_available": None, "shortfall": 0}
    rng = np.random.default_rng(int(seed or 0) + 7919)
    want = int(design.n_perturbed or 0)
    info = {"rule": design.selection, "n_requested": want, "n_effective_available": None, "shortfall": 0}
    if design.selection == "random":
        genes = sorted(int(g) for g in rng.choice(n, size=want, replace=False))
    elif design.selection == "hubs":
        out = (model.adjacency != 0).sum(axis=1) + rng.uniform(0, 0.5, size=n)
        genes = sorted(int(g) for g in np.argsort(-out, kind="stable")[:want])
    elif design.selection == "expressed":
        genes = sorted(int(g) for g in np.argsort(-model.baseline, kind="stable")[:want])
    else:
        probe = condition_screen(model, cfg=cfg, seed=seed)
        effective = [c[0] for c, lab in zip(probe.conditions, probe.labels) if lab == "effective"]
        take = min(want, len(effective))
        genes = sorted(int(g) for g in rng.choice(np.array(effective, dtype=int), size=take, replace=False)) if take else []
        info.update(n_effective_available=len(effective), shortfall=max(0, want - take))
    info["n_selected"] = len(genes)
    return genes, info


def run_stat_screen(adjacency: np.ndarray, generator: str, params: ModelParams, mode: str = "knockdown",
                    pairs: Optional[int] = None, cells_per_condition: int = 40, design: Optional[ScreenDesign] = None,
                    measurement: bool = True, seed: Optional[int] = 0, cfg: Optional[CellConfig] = None) -> Any:
    """One statistical screen of the graph ``adjacency``: the analogue of :func:`grn_sim.perturbseq.pipeline.run_screen`."""
    from grn_sim.perturbseq.pipeline import ScreenRun, _stamp_design

    cfg = cfg or CellConfig(guide_efficiency_mean=0.92)
    design = design or ScreenDesign.full()
    seed_i = int(seed or 0)
    model = make_model(generator, adjacency, params, mode, np.random.default_rng(seed_i + 101))
    seconds: Dict[str, float] = {}
    t = time.time()
    genes, info = _select(model, design, cfg, seed)
    if design.is_partial and not genes:
        raise ValueError("this network has no effective genes to perturb")
    screen = condition_screen(model, genes=genes, pairs=pairs or None, cfg=cfg, seed=seed)
    seconds["screen"] = round(time.time() - t, 2)
    t = time.time()
    population = simulate_cells(model, screen, cells_per_condition, cfg=cfg, seed=seed_i + 1)
    seconds["cells"] = round(time.time() - t, 2)
    t = time.time()
    adata = population_to_anndata(population, screen, seed=seed_i + 2, measurement=MeasurementConfig() if measurement else None)
    seconds["measurement"] = round(time.time() - t, 2)
    _stamp_design(adata, design, info, screen)
    adata.uns["generator"] = {"name": model.name, **{k: float(v) for k, v in vars(params).items()}}
    settings = {"mode": mode, "pairs": int(pairs or 0), "cells_per_condition": int(cells_per_condition), "generator": model.name,
                "n_genes": int(model.n), "seed": seed, "measurement": bool(measurement), "design": design.label(),
                "params": vars(params)}
    return ScreenRun(screen, population, adata, settings, seconds, design, info)


def preset_graph(preset: str, n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    The signed graph the mechanistic preset ``preset`` draws for ``(n, seed)`` - the same random calls as
    ``NetworkPreset.build`` up to the topology, without the kinetics - so mechanistic and statistical data can be generated
    on identical graphs.
    """
    from grn_sim.core.topology import TopologyType, generate_topology
    from grn_sim.perturbseq.calibration import PRESETS

    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    p = PRESETS[preset]
    rng = np.random.default_rng(seed)
    modules = (p.n_modules or int(np.clip(n // 50, 5, 10))) if p.topology == "modular" else 2
    topo = generate_topology(n, k_av=p.k_av, net_type=TopologyType(p.topology), ratio=p.ratio, subdiv=modules, rng=rng,
                             p_within=p.p_within, attachment_offset=p.attachment_offset)
    return np.asarray(topo.adjacency)
