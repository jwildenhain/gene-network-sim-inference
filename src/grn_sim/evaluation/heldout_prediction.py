"""
Leave-perturbation-out prediction: can a structure learned without a perturbation predict its response?

For every fold, a set of effective single perturbations (knockdown detected, ``|own z| >= min_own_z``) is hidden:
their rows get weight 0 when the network is fitted. The hidden node ``a`` is then predicted from the fitted direct
effects ``G`` as the linear steady-state response ``own * T[a, :] / T[a, a]`` with ``T = (I - G)^-1`` and ``own`` the mean
on-target effect of the training nodes. Compared with two baselines that use no structure:

* ``mean_profile``: the mean response of the training nodes;
* ``neighbour``: the training nodes that move gene ``a`` most (top 5 by |effect on a|), their normalised responses
  averaged with the sign of that effect (the "genes that regulate me also respond to me" heuristic).

Score: Pearson r over genes except ``a`` itself, between the prediction and the observed response of ``a`` (and, when
the noise-free log2 fold changes are given, those). Only genes that are nodes can be predicted (the others are skipped).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from grn_sim.inference.inspre import fit_inspre


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def heldout_prediction(ace, truth_lfc: Optional[np.ndarray] = None, folds: int = 5, k: int = 3, alpha: Optional[float] = 0.05,
                       min_own_z: float = 2.0, seed: int = 0) -> Dict[str, float]:
    est = ace.denoised(alpha) if alpha is not None else ace
    nodes = est.nodes
    p = len(nodes)
    R, w = est.normalised(min_own_z)
    ok = w > 0
    eff = np.flatnonzero(ok)
    rng = np.random.default_rng(seed)
    rng.shuffle(eff)
    parts = np.array_split(eff, folds)
    node_cols = nodes
    # square response matrix of the nodes (rows: perturbed node, columns: node genes)
    Rn = R[:, node_cols]
    L = est.lfc[:, node_cols]
    out: Dict[str, list] = {"inspre": [], "mean_profile": [], "neighbour": []}
    out_t: Dict[str, list] = {"inspre": [], "mean_profile": [], "neighbour": []}
    for hold in parts:
        if len(hold) == 0:
            continue
        wt = w.copy()
        wt[hold] = 0.0
        train = np.setdiff1d(eff, hold)
        G, _ = fit_inspre(Rn, target_edges=int(2 * k * p), weights=wt)
        T = np.linalg.inv(np.eye(p) - G)
        own_pred = float(np.mean(est.own[train]))
        mean_prof = L[train].mean(axis=0)
        for a in hold:
            mask = np.ones(p, dtype=bool)
            mask[a] = False
            pred = {"inspre": own_pred * T[a] / T[a, a], "mean_profile": mean_prof}
            eff_on_a = L[train, a]
            top = train[np.argsort(-np.abs(eff_on_a))[:5]]
            pred["neighbour"] = np.mean([np.sign(L[b, a]) * Rn[b] for b in top], axis=0) * own_pred
            obs = L[a]
            for name, v in pred.items():
                out[name].append(_pearson(v[mask], obs[mask]))
                if truth_lfc is not None:
                    out_t[name].append(_pearson(v[mask], truth_lfc[a][node_cols][mask]))
    res: Dict[str, float] = {f"r_observed_{n}": float(np.mean(v)) for n, v in out.items() if v}
    if truth_lfc is not None:
        res.update({f"r_truth_{n}": float(np.mean(v)) for n, v in out_t.items() if v})
    res["n_heldout"] = int(len(out["inspre"]))
    return res
