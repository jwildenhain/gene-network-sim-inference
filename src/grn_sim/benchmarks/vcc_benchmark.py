"""
Virtual Cell Challenge (VCC 2026) Comparative Benchmark Suite.
Compares Mechanistic Dynamical GRN Inference (S2G, G2S, Voting Ensemble, NIR, FAST)
against Empirical Cross-Atlas Perturbation Transfer (AtlasShift).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from grn_sim.core.topology import TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.by_chance import ByChance
from grn_sim.inference.atlasshift import AtlasShift
from grn_sim.evaluation.metrics import evaluate_inference


def compute_profile_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Pearson correlation, Cosine similarity, and RMSE between two response vectors."""
    yt = np.asarray(y_true, dtype=float).flatten()
    yp = np.asarray(y_pred, dtype=float).flatten()

    # Avoid div-by-zero on zero vectors
    norm_t = np.linalg.norm(yt)
    norm_p = np.linalg.norm(yp)
    cosine = float(np.dot(yt, yp) / (norm_t * norm_p)) if norm_t > 1e-12 and norm_p > 1e-12 else 0.0

    std_t = np.std(yt)
    std_p = np.std(yp)
    if std_t > 1e-12 and std_p > 1e-12:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson = 0.0

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    return {"pearson_r": round(pearson, 4), "cosine_sim": round(cosine, 4), "rmse": round(rmse, 4)}


def run_vcc_comparative_benchmark(
    num_genes: int = 16,
    k_av: float = 1.8,
    seed: int = 2026,
    rewire_fractions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Executes a multi-context comparative study:
    1. Simulates multiple source cell contexts (Context 1 = K562-like, Context 2 = HCT116-like, Context 3 = HEK293T-like).
    2. Simulates an unseen destination context (Context A) with varying network rewiring.
    3. Evaluates Mechanistic GRN (S2G, G2S, Voting) vs Empirical AtlasShift on:
       - Causal interaction matrix recovery (Coverage, FDR, Total Errors)
       - Transcriptional response prediction (Pearson r, Cosine Sim, RMSE)
       - Out-of-Distribution (OOD) perturbation generalization
       - Context rewiring sensitivity curve
    """
    if rewire_fractions is None:
        rewire_fractions = [0.0, 0.1, 0.2, 0.35, 0.5]

    gene_names = [f"Gene_{i+1:02d}" for i in range(num_genes)]

    # 1. Base Shared Regulatory Network (Scale-free Barabási-Albert)
    base_net = GeneNetwork.create(
        n=num_genes,
        k_av=k_av,
        net_type=TopologyType.POWER_LAW,
        ratio=0.55,
        seed=seed,
    )
    sim = PerturbationSimulator(perturbation_factor=1.0, measurement_noise=0.01)

    # 2. Simulate 3 Source Atlases (slight basal expression differences simulating cell lines)
    source_experiments = []
    source_names = ["K562_GWPS", "HCT116_Perturb", "HEK293T_Perturb"]
    for idx, s_name in enumerate(source_names):
        # Slightly perturb basal rates to model cell-type specific baseline
        s_net = GeneNetwork.create(
            n=num_genes,
            k_av=k_av,
            net_type=TopologyType.POWER_LAW,
            ratio=0.55,
            seed=seed,
        )
        s_net.params.v = np.clip(base_net.params.v * (1.0 + 0.15 * np.sin(idx + 1 + np.arange(num_genes))), 0.2, 4.0)
        exp_s = sim.run(s_net, seed=seed + idx * 10)
        source_experiments.append((s_name, exp_s))

    # 3. Simulate Target Context with 15% Rewiring (Context A)
    target_net = GeneNetwork.create(
        n=num_genes,
        k_av=k_av,
        net_type=TopologyType.POWER_LAW,
        ratio=0.55,
        seed=seed,
    )
    # Rewire 15% of edges in target context
    rng = np.random.default_rng(seed + 999)
    adj = target_net.topology.adjacency.copy()
    nz_rows, nz_cols = np.where(adj != 0)
    num_rewire = max(1, int(len(nz_rows) * 0.15))
    pick_idx = rng.choice(len(nz_rows), size=num_rewire, replace=False)
    for p in pick_idx:
        r_old, c_old = nz_rows[p], nz_cols[p]
        adj[r_old, c_old] = 0
        new_c = (c_old + 1) % num_genes
        if new_c == r_old:
            new_c = (new_c + 1) % num_genes
        adj[r_old, new_c] = 1 if rng.random() > 0.45 else -1
    target_net.topology.adjacency = adj
    target_exp = sim.run(target_net, seed=seed + 999)

    X_tgt = target_exp.X
    U_tgt = target_exp.U
    T_true = target_exp.true_matrix

    # 4. Instantiate & Fit All Models
    s2g = SimpleToGeneral()
    g2s = GeneralToSpecific()
    vote = VotingEnsemble(s2g=s2g, g2s=g2s)
    nir = NIRAlgorithm(k_interactions=max(1, int(np.round(k_av))))
    fast = FastSVD(k_interactions=max(1, int(np.round(k_av))))
    by_chance = ByChance(k_interactions=max(1, int(np.round(k_av))))

    # Fit AtlasShift with the 3 source atlases
    atlasshift = AtlasShift(weights=[2.0, 1.0, 1.0], de_threshold=0.15)
    for s_name, exp_s in source_experiments:
        atlasshift.add_source(s_name, exp_s.X, exp_s.U, weight=1.0)

    res_s2g = s2g.fit(X_tgt, U_tgt)
    res_g2s = g2s.fit(X_tgt, U_tgt)
    res_vote = vote.fit(X_tgt, U_tgt)
    res_nir = nir.fit(X_tgt, -U_tgt)
    res_fast = fast.fit(X_tgt, -U_tgt)
    res_bc = by_chance.fit(X_tgt, U_tgt)
    res_atlas = atlasshift.fit(X_tgt, U_tgt)

    # 5. Evaluate Network Edge Recovery (Causal Topology)
    eval_dict = {
        "s2g": evaluate_inference(T_true, res_s2g.adjacency_pred),
        "g2s": evaluate_inference(T_true, res_g2s.adjacency_pred),
        "vote": evaluate_inference(T_true, res_vote.adjacency_pred),
        "nir": evaluate_inference(T_true, res_nir.adjacency_pred),
        "fast": evaluate_inference(T_true, res_fast.adjacency_pred),
        "atlasshift": evaluate_inference(T_true, res_atlas.adjacency_pred),
        "by_chance": evaluate_inference(T_true, res_bc.adjacency_pred),
    }

    # Summary table formatted for display
    summary_table = []
    for k, m in eval_dict.items():
        summary_table.append({
            "algorithm": k.upper(),
            "paradigm": "Empirical Transfer (VCC 2026)" if k == "atlasshift" else ("Null Baseline" if k == "by_chance" else "Mechanistic Dynamical GRN"),
            "coverage_pct": round(m.coverage_pct, 1),
            "false_positives_pct": round(m.false_positives_pct, 1),
            "power_pct": round(m.inferred_connections_pct, 1),
            "fdr_pct": round(m.fp_per_inferred_pct, 1),
            "total_errors": int(m.total_errors),
        })

    # 6. Response Profile Prediction: In-Distribution vs Out-of-Distribution (OOD)
    # Split targets into Seen (80%) vs Unseen (20% OOD)
    n_targets = num_genes
    seen_mask = np.ones(n_targets, dtype=bool)
    n_unseen = max(2, int(n_targets * 0.25))
    unseen_indices = list(range(n_targets - n_unseen, n_targets))
    for u_idx in unseen_indices:
        seen_mask[u_idx] = False

    # Predict response for target j:
    # Mechanistic GRN: X_pred[:, j] = -pinv(A_est) @ U[:, j]
    # AtlasShift: X_pred[:, j] = weighted average of source X[:, j] (or 0 for OOD)
    inv_A_vote = np.linalg.pinv(res_vote.A_est) if np.any(res_vote.A_est) else np.linalg.pinv(res_s2g.A_est)
    inv_A_s2g = np.linalg.pinv(res_s2g.A_est)

    sim_seen_vote, sim_seen_atlas = [], []
    sim_ood_vote, sim_ood_atlas = [], []

    for j in range(n_targets):
        y_true_j = X_tgt[:, j]
        # Mechanistic prediction: X_pred = pinv(A_est) @ U
        y_pred_mech = inv_A_vote @ U_tgt[:, j]
        # AtlasShift prediction
        if j not in unseen_indices:
            # Transfer weighted average across sources
            y_pred_atlas = np.mean([exp_s.X[:, j] for _, exp_s in source_experiments], axis=0)
        else:
            # Unseen OOD perturbation: AtlasShift has no transfer data for this target gene!
            y_pred_atlas = np.zeros(num_genes, dtype=float)

        score_mech = compute_profile_similarity(y_true_j, y_pred_mech)
        score_atlas = compute_profile_similarity(y_true_j, y_pred_atlas)

        if j not in unseen_indices:
            sim_seen_vote.append(score_mech)
            sim_seen_atlas.append(score_atlas)
        else:
            sim_ood_vote.append(score_mech)
            sim_ood_atlas.append(score_atlas)

    avg_seen_vote = {k: round(float(np.mean([x[k] for x in sim_seen_vote])), 3) for k in ["pearson_r", "cosine_sim", "rmse"]}
    avg_seen_atlas = {k: round(float(np.mean([x[k] for x in sim_seen_atlas])), 3) for k in ["pearson_r", "cosine_sim", "rmse"]}
    avg_ood_vote = {k: round(float(np.mean([x[k] for x in sim_ood_vote])), 3) for k in ["pearson_r", "cosine_sim", "rmse"]}
    avg_ood_atlas = {k: round(float(np.mean([x[k] for x in sim_ood_atlas])), 3) for k in ["pearson_r", "cosine_sim", "rmse"]}

    # 7. Context Rewiring Sensitivity Sweep (rho in [0..0.5])
    rewire_decay = {
        "rewire_rates": [int(r * 100) for r in rewire_fractions],
        "mechanistic_r": [],
        "atlasshift_r": [],
    }
    for rf in rewire_fractions:
        net_rf = GeneNetwork.create(n=num_genes, k_av=k_av, net_type=TopologyType.POWER_LAW, ratio=0.55, seed=seed)
        if rf > 0:
            rng_r = np.random.default_rng(seed + int(rf * 1000))
            a_rf = net_rf.topology.adjacency.copy()
            nz_r, nz_c = np.where(a_rf != 0)
            n_rew = max(1, int(len(nz_r) * rf))
            p_idx = rng_r.choice(len(nz_r), size=n_rew, replace=False)
            for p in p_idx:
                a_rf[nz_r[p], nz_c[p]] = 0
                nc = (nz_c[p] + 2) % num_genes
                a_rf[nz_r[p], nc] = 1 if rng_r.random() > 0.45 else -1
            net_rf.topology.adjacency = a_rf
        exp_rf = sim.run(net_rf, seed=seed + 100)

        # Evaluate mean Pearson r
        r_mech_list, r_atlas_list = [], []
        for j in range(num_genes):
            yt_j = exp_rf.X[:, j]
            yp_m = inv_A_vote @ exp_rf.U[:, j]
            yp_a = np.mean([exp_s.X[:, j] for _, exp_s in source_experiments], axis=0)
            r_mech_list.append(compute_profile_similarity(yt_j, yp_m)["pearson_r"])
            r_atlas_list.append(compute_profile_similarity(yt_j, yp_a)["pearson_r"])
        rewire_decay["mechanistic_r"].append(round(float(np.mean(r_mech_list)), 3))
        rewire_decay["atlasshift_r"].append(round(float(np.mean(r_atlas_list)), 3))

    return {
        "status": "success",
        "parameters": {
            "num_genes": num_genes,
            "k_av": k_av,
            "sources": source_names,
            "seen_targets": n_targets - n_unseen,
            "unseen_ood_targets": n_unseen,
        },
        "summary_table": summary_table,
        "in_distribution": {
            "mechanistic": avg_seen_vote,
            "atlasshift": avg_seen_atlas,
        },
        "out_of_distribution": {
            "mechanistic": avg_ood_vote,
            "atlasshift": avg_ood_atlas,
        },
        "rewire_decay": rewire_decay,
    }
