"""
Screen designs: which genes of the network are perturbed.

A genome-scale CRISPRi screen perturbs a few hundred genes and measures thousands. ``ScreenDesign`` separates *which genes are
perturbed* from how the screen is generated (network preset, cell model, measurement):

* ``full``: every gene is perturbed once (the default, and what the studio did before);
* ``partial``: ``n_perturbed`` genes are perturbed, all genes are measured. The perturbed genes are chosen by ``selection``:

  - ``effective`` (default): among the genes whose knockdown moves other genes in the simulated network (the screen's
    ``effective`` label: a noise-free single-knockdown screen of the whole network decides), drawn at random with the run's
    seed. This mirrors how real screens pick targets (genes with a readable phenotype). If the network has fewer effective
    genes than requested, all of them are used and ``info["shortfall"]`` says how many are missing;
  - ``random``: uniformly at random among all genes (no selection effect);
  - an explicit list of gene indices (``genes``).

Statistics of a partial screen are their own category (:func:`grn_sim.perturbseq.partial_stats.design_statistics`): the share
of true edges a screen can identify at all, the genes that respond to at least one perturbation, per-block densities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SELECTIONS = ("effective", "random")
KINDS = ("full", "partial")


@dataclass(frozen=True)
class ScreenDesign:
    kind: str = "full"
    n_perturbed: Optional[int] = None
    selection: str = "effective"
    genes: Optional[Tuple[int, ...]] = None

    # -- constructors ------------------------------------------------------------------------------
    @classmethod
    def full(cls) -> "ScreenDesign":
        return cls("full")

    @classmethod
    def partial(cls, n_perturbed: int, selection: str = "effective") -> "ScreenDesign":
        return cls("partial", int(n_perturbed), selection)

    @classmethod
    def explicit(cls, genes: List[int]) -> "ScreenDesign":
        genes_t = tuple(sorted({int(g) for g in genes}))
        return cls("partial", len(genes_t), "explicit", genes_t)

    @classmethod
    def from_options(cls, n_perturbed: Optional[int] = None, selection: str = "effective",
                     genes: Optional[List[int]] = None) -> "ScreenDesign":
        """The design asked for by the CLI / API options: nothing or 0 means ``full``."""
        if genes:
            return cls.explicit(list(genes))
        if not n_perturbed:
            return cls.full()
        return cls.partial(int(n_perturbed), selection)

    # -- queries -----------------------------------------------------------------------------------
    @property
    def is_partial(self) -> bool:
        return self.kind == "partial"

    def label(self) -> str:
        if not self.is_partial:
            return "full"
        return f"partial_{self.n_perturbed}_{self.selection}"

    def validate(self, n_genes: int) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"design kind must be one of {KINDS}, got {self.kind!r}")
        if not self.is_partial:
            return
        if self.genes is not None:
            if not self.genes or min(self.genes) < 0 or max(self.genes) >= n_genes:
                raise ValueError(f"perturbed genes must be indices in [0, {n_genes - 1}]")
            return
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.selection!r}")
        if self.n_perturbed is None or not 1 <= int(self.n_perturbed) <= n_genes:
            raise ValueError(f"n_perturbed must be between 1 and the number of genes ({n_genes}), got {self.n_perturbed!r}")

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "n_perturbed": self.n_perturbed, "selection": self.selection if self.is_partial else None,
                "label": self.label()}


def select_perturbed(network: Any, design: ScreenDesign, mode: str = "knockdown", seed: Optional[int] = 0,
                     efficiency: Any = 0.8) -> Tuple[Optional[List[int]], Dict[str, Any]]:
    """
    The perturbed gene indices of ``design`` on ``network`` (``None`` for a full screen) and a dict describing how they were
    chosen (``rule``, ``n_requested``, ``n_selected``, ``n_effective_available``, ``shortfall``).

    ``effective`` runs one noise-free single-knockdown screen of the whole network (the same solver as the real screen) to
    find the genes whose perturbation moves other genes (label ``effective``); the cost is one pass over all genes.
    """
    n = int(network.n)
    design.validate(n)
    if not design.is_partial:
        return None, {"rule": "full", "n_requested": n, "n_selected": n, "n_effective_available": None, "shortfall": 0}
    if design.genes is not None:
        return sorted(design.genes), {"rule": "explicit", "n_requested": len(design.genes), "n_selected": len(design.genes),
                                      "n_effective_available": None, "shortfall": 0}
    rng = np.random.default_rng(int(seed or 0) + 7919)
    want = int(design.n_perturbed or 0)
    if design.selection == "random":
        genes = sorted(int(g) for g in rng.choice(n, size=want, replace=False))
        return genes, {"rule": "random", "n_requested": want, "n_selected": want, "n_effective_available": None, "shortfall": 0}
    from grn_sim.perturbseq.conditions import simulate_conditions
    probe = simulate_conditions(network, mode=mode if mode != "input" else "knockdown", efficiency=efficiency, seed=seed)
    singles = [c[0] for c in probe.conditions if len(c) == 1]
    effective = [g for g, lab in zip(singles, probe.labels) if lab == "effective"]
    take = min(want, len(effective))
    genes = sorted(int(g) for g in rng.choice(np.array(effective, dtype=int), size=take, replace=False)) if take else []
    return genes, {"rule": "effective", "n_requested": want, "n_selected": take, "n_effective_available": len(effective),
                   "shortfall": max(0, want - take)}
