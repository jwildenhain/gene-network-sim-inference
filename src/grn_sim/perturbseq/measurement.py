"""
Stage 3 of the Perturb-seq generator: measurement and screen artefacts.

Turns the simulated expression of every cell (stage 2) into what a droplet Perturb-seq experiment records, applying
the artefacts in the order the experiment introduces them:

1. **Doublets**: a fraction of droplets holds two cells; their transcriptomes and guides add up.
2. **Ambient RNA**: every droplet also captures a little of the pooled (soup) profile.
3. **Gene capture bias**: a fixed per-gene detection efficiency (lognormal), the same in every cell.
4. **Batches** (gem groups / lanes): a per-gene log shift per batch and a per-batch sequencing depth factor.
   Cells are spread over batches at random, so every batch holds control cells.
5. **Library size and UMI sampling**: lognormal library size, then Poisson (``dispersion = 0``) or negative-binomial
   sampling of the droplet's composition. Biological variation is already in the expression, so the default technical
   overdispersion is small.
6. **Guide capture and assignment**: true guides get UMIs ~ NB(``guide_umi_mean``); every droplet also picks up a few
   ambient guide UMIs from the whole library. A guide is called when it has at least ``assign_min_umis`` UMIs and at
   least ``assign_min_fraction`` of the droplet's guide UMIs; cells end up ``single``, ``unassigned`` or ``multiple``.

Every artefact is recorded next to the observation (``is_doublet``, ``batch``, ``ambient_fraction``, true vs assigned
guide), so a pipeline's QC (doublet removal, guide calling, batch correction) can be scored against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from grn_sim.perturbseq.cells import CellPopulation


@dataclass
class MeasurementConfig:
    """Measurement and screen-artefact parameters (defaults: a typical 10x Perturb-seq run)."""

    mean_library_depth: float = 20000.0
    depth_sd: float = 0.35                 # log-sd of library size between droplets
    dispersion: float = 0.05               # NB overdispersion of UMI sampling (0: Poisson)
    gene_capture_sd: float = 0.3           # log-sd of the fixed per-gene capture efficiency
    n_batches: int = 4
    batch_gene_sd: float = 0.1             # log-sd of the per-gene shift of each batch
    batch_depth_sd: float = 0.15           # log-sd of the depth factor of each batch
    ambient_fraction: float = 0.03         # mean share of a droplet's RNA coming from the soup
    ambient_concentration: float = 50.0    # Beta concentration of the per-droplet ambient share
    doublet_rate: float = 0.04             # share of droplets holding two cells
    guide_umi_mean: float = 30.0           # mean UMIs of a captured guide
    guide_dispersion: float = 0.5          # NB overdispersion of guide UMIs
    guide_dropout: float = 0.05            # chance a present guide is not captured at all
    ambient_guide_umis: float = 0.5        # mean ambient guide UMIs per droplet, spread over the library
    assign_min_umis: int = 3
    assign_min_fraction: float = 0.2

    def validate(self) -> None:
        if self.mean_library_depth <= 0 or self.guide_umi_mean <= 0:
            raise ValueError("mean_library_depth and guide_umi_mean must be > 0")
        if self.n_batches < 1:
            raise ValueError("n_batches must be >= 1")
        for name in ("depth_sd", "dispersion", "gene_capture_sd", "batch_gene_sd", "batch_depth_sd",
                     "guide_dispersion", "ambient_guide_umis"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("ambient_fraction", "doublet_rate", "guide_dropout", "assign_min_fraction"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.ambient_concentration <= 0:
            raise ValueError("ambient_concentration must be > 0")
        if self.assign_min_umis < 1:
            raise ValueError("assign_min_umis must be >= 1")


@dataclass
class MeasuredScreen:
    """Droplet-level observations of a simulated screen plus the truth behind them."""

    counts: Any                      # scipy.sparse CSR (droplets x genes)
    guide_counts: Any                # scipy.sparse CSR (droplets x guides)
    guide_names: List[str]
    obs: List[Dict[str, Any]]        # per droplet: truth and observation
    true_expression: np.ndarray      # (droplets x genes) expression before measurement (doublets summed)
    gene_capture: np.ndarray         # (genes,) fixed capture efficiency
    batch_shift: np.ndarray          # (batches x genes) log shift
    meta: Dict[str, Any] = field(default_factory=dict)


def _nb(rng: np.random.Generator, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Negative binomial with the given mean and dispersion (variance = mean + dispersion * mean^2); Poisson at 0."""
    mean = np.maximum(mean, 0.0)
    if dispersion <= 0:
        return rng.poisson(mean)
    shape = 1.0 / dispersion
    return rng.poisson(rng.gamma(shape, mean / shape))


def measure_population(
    population: CellPopulation,
    config: Optional[MeasurementConfig] = None,
    seed: Optional[int] = None,
) -> MeasuredScreen:
    """Simulates the droplet measurement of ``population`` (see the module docstring for the steps)."""
    from scipy import sparse

    cfg = config or MeasurementConfig()
    cfg.validate()
    rng = np.random.default_rng(seed)
    n_cells, n_genes = population.expression.shape
    if not np.all(np.isfinite(population.expression)):
        raise ValueError("cell expression contains non-finite values; cannot measure")
    expr = np.maximum(population.expression, 0.0)

    # 1. Doublets: droplet d holds cell d, and with probability doublet_rate a second, random cell.
    partner = np.full(n_cells, -1)
    is_doublet = rng.uniform(size=n_cells) < cfg.doublet_rate
    if n_cells > 1 and is_doublet.any():
        draw = rng.integers(0, n_cells - 1, size=int(is_doublet.sum()))
        own = np.flatnonzero(is_doublet)
        partner[own] = draw + (draw >= own)  # any cell but itself
    true_expr = expr.copy()
    true_expr[is_doublet] += expr[partner[is_doublet]]

    # 2. Ambient RNA: mix each droplet's composition with the soup (pooled composition of all cells).
    totals = true_expr.sum(axis=1, keepdims=True)
    comp = np.divide(true_expr, totals, out=np.zeros_like(true_expr), where=totals > 0)
    soup = expr.sum(axis=0)
    soup = soup / soup.sum() if soup.sum() > 0 else np.full(n_genes, 1.0 / n_genes)
    a = np.clip(cfg.ambient_fraction, 1e-6, 1 - 1e-6)
    ambient = rng.beta(a * cfg.ambient_concentration, (1 - a) * cfg.ambient_concentration, size=n_cells) \
        if cfg.ambient_fraction > 0 else np.zeros(n_cells)
    comp = (1.0 - ambient[:, None]) * comp + ambient[:, None] * soup

    # 3.-4. Gene capture bias and batch effects (multiplicative on the composition, then renormalised).
    capture = rng.lognormal(0.0, cfg.gene_capture_sd, size=n_genes) if cfg.gene_capture_sd > 0 else np.ones(n_genes)
    batch = rng.integers(0, cfg.n_batches, size=n_cells)
    batch_shift = cfg.batch_gene_sd * rng.standard_normal((cfg.n_batches, n_genes))
    batch_depth = np.exp(cfg.batch_depth_sd * rng.standard_normal(cfg.n_batches))
    weight = comp * capture[None, :] * np.exp(batch_shift[batch])
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-300)

    # 5. Library size (doublets carry about twice the RNA) and UMI sampling.
    lib = rng.lognormal(np.log(cfg.mean_library_depth) - 0.5 * cfg.depth_sd ** 2, cfg.depth_sd, size=n_cells)
    lib *= batch_depth[batch] * np.where(is_doublet, 2.0, 1.0)
    counts = _nb(rng, weight * lib[:, None], cfg.dispersion).astype(np.int32)

    # 6. Guides: true guide identities per droplet (doublets carry both cells' guides).
    def split(guide: str) -> List[str]:
        return [g for g in guide.split("+") if g]

    cell_guides = [split(g) for g in population.guide]
    guide_names = sorted({g for gs in cell_guides for g in gs})
    gidx = {g: i for i, g in enumerate(guide_names)}
    library_share = np.zeros(len(guide_names))
    for gs in cell_guides:
        for g in gs:
            library_share[gidx[g]] += 1
    library_share = library_share / max(library_share.sum(), 1.0)
    rows, cols, vals = [], [], []
    true_guides: List[List[str]] = []
    for d in range(n_cells):
        present = list(cell_guides[d]) + (list(cell_guides[partner[d]]) if is_doublet[d] else [])
        true_guides.append(present)
        umis: Dict[int, int] = {}
        for g in present:
            if rng.uniform() >= cfg.guide_dropout:
                umis[gidx[g]] = umis.get(gidx[g], 0) + int(_nb(rng, np.array([cfg.guide_umi_mean]), cfg.guide_dispersion)[0])
        n_amb = int(rng.poisson(cfg.ambient_guide_umis)) if guide_names else 0
        if n_amb:
            for j in rng.choice(len(guide_names), size=n_amb, p=library_share):
                umis[int(j)] = umis.get(int(j), 0) + 1
        for j, v in umis.items():
            if v > 0:
                rows.append(d)
                cols.append(j)
                vals.append(v)
    guide_counts = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, len(guide_names)), dtype=np.int32)

    obs: List[Dict[str, Any]] = []
    for d in range(n_cells):
        row = guide_counts.getrow(d)
        total = int(row.sum())
        called = [guide_names[j] for j, v in zip(row.indices, row.data)
                  if v >= cfg.assign_min_umis and v >= cfg.assign_min_fraction * total]
        status = "unassigned" if not called else ("single" if len(called) == 1 else "multiple")
        truth = "+".join(sorted(set(true_guides[d])))
        obs.append({
            "batch": f"batch_{batch[d] + 1}",
            "is_doublet": bool(is_doublet[d]),
            "doublet_partner": int(partner[d]),
            "ambient_fraction": float(ambient[d]),
            "library_size": float(lib[d]),
            "true_guides": truth,
            "guide_umis": total,
            "assigned_guide": called[0] if status == "single" else ("" if status == "unassigned" else "+".join(sorted(called))),
            "assignment_status": status,
            # Correct: a singlet whose called guides are exactly its true guides (two for a double perturbation).
            "assignment_correct": bool(not is_doublet[d] and called and sorted(called) == sorted(set(true_guides[d]))),
        })

    return MeasuredScreen(
        counts=sparse.csr_matrix(counts),
        guide_counts=guide_counts,
        guide_names=guide_names,
        obs=obs,
        true_expression=true_expr,
        gene_capture=capture,
        batch_shift=batch_shift,
        meta={"config": cfg, "seed": seed},
    )
