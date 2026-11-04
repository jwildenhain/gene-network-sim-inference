"""
Statistics of a partial screen (m perturbed genes out of n measured), as their own categories.

A partial screen differs from a full one in what can be known, not only in how large it is, so its statistics are grouped by
that question instead of being compared with the genome-scale realism targets one by one:

* ``design``: how many genes were perturbed, by which rule, how many of them are effective (move other genes);
* ``identifiability``: how many of the true regulatory edges have a *perturbed* regulator at all (only those can be found from
  interventions) and how they split into P->P, P->Q and the rest (Q = unperturbed genes);
* ``coverage``: how many unperturbed genes respond to at least one perturbation, and how many perturbations reach each;
* ``blocks``: effect density and mediated share separately for P->P and P->Q, the two blocks a partial screen can be scored on.

All of them are computed from the noise-free :class:`ConditionScreen` (the truth), like ``truth_statistics``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def design_statistics(screen: Any, info: Optional[Dict[str, Any]] = None,
                      effect_threshold: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    thr = screen.effect_threshold if effect_threshold is None else float(effect_threshold)
    n = screen.n_genes
    singles = [k for k, c in enumerate(screen.conditions) if len(c) == 1]
    P = np.array([screen.conditions[k][0] for k in singles], dtype=int)
    inP = np.zeros(n, dtype=bool)
    inP[P] = True
    lfc = screen.total_effects[:, singles]                       # (n, m): response of every gene to each perturbation
    affected = np.abs(lfc) >= thr
    affected[P, np.arange(len(P))] = False                       # the perturbed gene itself is not an effect
    adj = np.asarray(screen.adjacency) != 0
    np.fill_diagonal(adj, False)
    info = info or {}

    effective = sum(1 for lab in screen.labels if lab == "effective")        # one label per single perturbation
    design = {
        "n_genes": float(n),
        "n_perturbed": float(len(P)),
        "fraction_perturbed": float(len(P) / max(n, 1)),
        "n_perturbed_effective": float(effective),
        "fraction_perturbed_effective": float(effective / max(len(P), 1)),
        "n_requested": float(info.get("n_requested", len(P))),
        "shortfall": float(info.get("shortfall", 0)),
    }
    if info.get("n_effective_available") is not None:
        design["n_effective_available_in_network"] = float(info["n_effective_available"])

    n_edges = int(adj.sum())
    pp = int((adj & inP[:, None] & inP[None, :]).sum())
    pq = int((adj & inP[:, None] & ~inP[None, :]).sum())
    out_q = n_edges - pp - pq
    regulators = adj.any(axis=1)
    identifiability = {
        "edges_total": float(n_edges),
        "edges_P_to_P": float(pp),
        "edges_P_to_Q": float(pq),
        "edges_from_unperturbed": float(out_q),
        "fraction_edges_identifiable": float((pp + pq) / n_edges) if n_edges else 0.0,
        "fraction_edges_P_to_P": float(pp / n_edges) if n_edges else 0.0,
        "fraction_regulators_perturbed": float((regulators & inP).sum() / max(int(regulators.sum()), 1)),
    }

    Q = np.flatnonzero(~inP)
    responds = affected[Q].any(axis=1) if len(Q) else np.array([], dtype=bool)
    per_gene = affected[Q].sum(axis=1) if len(Q) else np.array([])
    direct_reg = (adj[np.ix_(P, Q)].any(axis=0)) if len(Q) and len(P) else np.array([], dtype=bool)
    coverage = {
        "n_unperturbed": float(len(Q)),
        "fraction_unperturbed_responding": float(responds.mean()) if len(Q) else 0.0,
        "median_perturbations_per_responding_gene": float(np.median(per_gene[responds])) if len(Q) and responds.any() else 0.0,
        "fraction_unperturbed_with_perturbed_direct_regulator": float(direct_reg.mean()) if len(Q) else 0.0,
    }

    direct_cols = np.stack([adj[g, :] for g in P], axis=1) if len(P) else np.zeros((n, 0), dtype=bool)   # (n, m)
    blocks: Dict[str, float] = {}
    for name, rows in (("P_to_P", inP), ("P_to_Q", ~inP)):
        aff = affected[rows]
        dr = direct_cols[rows]
        n_pairs = int(rows.sum()) * len(P) - (len(P) if name == "P_to_P" else 0)
        n_aff = int(aff.sum())
        blocks[f"affected_pair_density_{name}"] = float(n_aff / max(n_pairs, 1))
        blocks[f"fraction_effects_indirect_{name}"] = float((aff & ~dr).sum() / n_aff) if n_aff else 0.0
    return {"design": design, "identifiability": identifiability, "coverage": coverage, "blocks": blocks}
