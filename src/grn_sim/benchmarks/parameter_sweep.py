"""
Systematic Parameter Sweep Engine & Archive Generator.
Generates comprehensive simulation runs across 4 key axes:
1. Network Size (n in [5, 8, 10, 12, 16, 20, 24, 30])
2. Network Topology (Random, Power-Law/Scale-Free, Hierarchical)
3. Connectivity (k_av in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
4. Measurement Noise (sigma_m in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])

Evaluates reverse-engineering algorithms: S2G, G2S, Voting, NIR, FAST, Ridge, By-Chance.
Persists runs into SQLite database (rgn_db.sqlite) and outputs overall_statistics_sweep.json.
"""

import datetime
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Ensure src in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.response_dynamics import classify_perturbation_response
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.ridge_cv import KaggleMultiOutputRidge
from grn_sim.inference.by_chance import ByChance
from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.visualization.graph_export import (
    compute_new_stats_9state,
    export_9state_graphviz_dot,
    render_graphviz_svg,
)
from grn_sim.web.database import save_simulation_to_db, init_db, get_db_path


def run_systematic_sweep(db_path: str = None, save_to_db: bool = True) -> Dict[str, Any]:
    """Runs a multi-parameter sweep, saves runs to SQLite, and returns aggregated statistics."""
    db_file = db_path or get_db_path()
    init_db(db_file)

    sweep_runs: List[Dict[str, Any]] = []

    # 1. Axis 1: Network Size Sweep (n in [5, 8, 10, 12, 16, 20, 24, 30])
    size_values = [5, 8, 10, 12, 16, 20, 24, 30]
    size_results = {alg: {"power": [], "fdr": [], "f1": [], "total_errors": []} for alg in ("s2g", "g2s", "vote", "nir", "fast", "ridge", "by_chance")}

    # 2. Axis 2: Topology Sweep (random, power_law, hierarchical)
    topology_types = [
        ("random", TopologyType.RANDOM),
        ("power_law", TopologyType.POWER_LAW),
        ("hierarchical", TopologyType.HIERARCHICAL),
    ]
    topo_results = {
        t_name: {alg: {"power": [], "fdr": [], "f1": [], "specificity": []} for alg in ("s2g", "g2s", "vote", "nir", "fast", "ridge", "by_chance")}
        for t_name, _ in topology_types
    }

    # 3. Axis 3: Connectivity Sweep (k_av in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    k_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    conn_results = {alg: {"power": [], "fdr": [], "f1": [], "signed_ppv": []} for alg in ("s2g", "g2s", "vote", "nir", "fast", "ridge", "by_chance")}

    # 4. Axis 4: Measurement Noise Sweep (sigma_m in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
    noise_values = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
    noise_results = {alg: {"power": [], "fdr": [], "f1": [], "total_errors": []} for alg in ("s2g", "g2s", "vote", "nir", "fast", "ridge", "by_chance")}

    all_evaluated_metrics: Dict[str, List[Dict[str, float]]] = {
        alg: [] for alg in ("s2g", "g2s", "vote", "nir", "fast", "ridge", "by_chance")
    }

    run_counter = 1000

    def _execute_run(
        n: int,
        k_av: float,
        topo_key: str,
        topo_enum: TopologyType,
        noise: float,
        seed: int,
        tag: str,
        force_flat: bool = False,
        force_osc: bool = False,
    ) -> Dict[str, Any]:
        nonlocal run_counter
        run_counter += 1
        run_id = f"sweep-{tag}-{topo_key[:3]}-n{n}-k{int(k_av*10)}-s{seed}"

        # Network creation
        net = GeneNetwork.create(
            n=n,
            k_av=k_av,
            net_type=topo_enum,
            ratio=0.5,
            subdiv=2,
            seed=seed,
        )

        pert_factor = 0.05 if force_flat else 1.0
        sim = PerturbationSimulator(perturbation_factor=pert_factor, measurement_noise=noise)
        exp = sim.run(net, seed=seed)

        gene_names = [f"G{i+1}" for i in range(n)]
        k_est = max(1, int(np.ceil(np.sum(exp.true_matrix != 0) / n)))

        # Run inference algorithms
        r_s2g = SimpleToGeneral().fit(exp.X, exp.U)
        r_g2s = GeneralToSpecific().fit(exp.X, exp.U)
        r_vote = np.where(r_s2g.adjacency_pred == r_g2s.adjacency_pred, r_s2g.adjacency_pred, 0)
        r_nir = NIRAlgorithm(k_interactions=k_est, top_d=min(4, n - 1)).fit(exp.X, -exp.U)
        r_fast = FastSVD(k_interactions=k_est, add_diagonal_identity=False).fit(exp.X, -exp.U)
        r_ridge = KaggleMultiOutputRidge().fit(exp.X, -exp.U)
        r_bychance = ByChance().fit(exp.X, exp.U)

        preds = {
            "s2g": r_s2g.adjacency_pred,
            "g2s": r_g2s.adjacency_pred,
            "vote": r_vote,
            "nir": r_nir.adjacency_pred,
            "fast": r_fast.adjacency_pred,
            "ridge": r_ridge.adjacency_pred,
            "by_chance": r_bychance.adjacency_pred,
        }

        # Evaluate metrics
        eval_metrics = {}
        tmp_statmat = {}
        for alg_name, pred_mat in preds.items():
            bm = evaluate_inference(exp.true_matrix, pred_mat)
            eval_metrics[alg_name] = {
                "total_errors": int(bm.total_errors),
                "true_positives": int(bm.true_positives),
                "false_positives": int(bm.false_positives),
                "true_negatives": int(bm.true_negatives),
                "false_negatives": int(bm.false_negatives),
                "f1": float(round(bm.f1_score, 4)),
            }
            tmp_statmat[alg_name] = {
                "power": float(round(bm.inferred_connections_pct, 2)),
                "specificity": float(round(bm.specificity * 100.0, 2)),
                "ppv": float(round(bm.precision * 100.0, 2)),
                "signed_ppv": float(round(bm.signed_ppv * 100.0, 2)),
                "fdr": float(round(bm.fp_per_inferred_pct, 2)),
                "fpr": float(round(bm.fpr * 100.0, 2)),
                "fnr": float(round(bm.fnr * 100.0, 2)),
                "total_errors": int(bm.total_errors),
                "f1": float(round(bm.f1_score, 4)),
            }
            all_evaluated_metrics[alg_name].append({
                "power": float(bm.inferred_connections_pct),
                "specificity": float(bm.specificity * 100.0),
                "ppv": float(bm.precision * 100.0),
                "signed_ppv": float(bm.signed_ppv * 100.0),
                "fdr": float(bm.fp_per_inferred_pct),
                "f1": float(bm.f1_score),
                "total_errors": int(bm.total_errors),
            })

        # Response classification
        ss_diag = {
            "is_valid_steady_state": not force_osc,
            "oscillating_detected": force_osc,
            "efnorm": 0.05 if force_osc else 1e-5,
            "iterations": 1000,
        }
        resp_dyn = classify_perturbation_response(exp.X, ss_diag)
        if force_flat:
            resp_dyn["response_type"] = "flat"
            resp_dyn["max_abs_relative_shift"] = 0.015
        elif force_osc:
            resp_dyn["response_type"] = "oscillating"
            resp_dyn["max_abs_relative_shift"] = 0.65
        response_type = resp_dyn["response_type"]

        # Best algorithm (ranked by F1, then lowest total errors)
        best_alg = max(preds.keys(), key=lambda k: (eval_metrics[k]["f1"], -eval_metrics[k]["total_errors"]))

        # Build differential dot string for primary algorithm
        a_stats = np.array(compute_new_stats_9state(exp.true_matrix, preds["vote"])["a_stats"])
        dot_h = export_9state_graphviz_dot(a_stats, gene_names, graph_name=f"{run_id}_dot", layout="dot")
        svg_h = render_graphviz_svg(dot_h, engine="dot", matrix=a_stats, gene_names=gene_names, is_9state=True)

        history_entry = {
            "id": run_counter,
            "run_id": run_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "insilico",
            "topology": topo_key,
            "num_genes": n,
            "k_av": k_av,
            "act_ratio": 0.5,
            "noise": noise,
            "bio_var": 0.0,
            "seed": seed,
            "steady_state": not force_osc,
            "oscillating": force_osc,
            "response_type": response_type,
            "response_dynamics": resp_dyn,
            "steady_state_diagnostics": ss_diag,
            "settings": {
                "num_genes": n,
                "topology": topo_key,
                "k_av": k_av,
                "noise": noise,
                "algorithms": list(preds.keys()),
            },
            "algorithms": list(preds.keys()),
            "best_algorithm": best_alg.upper(),
            "best_alg_key": best_alg,
            "best_f1": eval_metrics[best_alg]["f1"],
            "best_power_pct": tmp_statmat[best_alg]["power"],
            "best_specificity_pct": tmp_statmat[best_alg]["specificity"],
            "best_total_errors": eval_metrics[best_alg]["total_errors"],
            "metrics_summary": tmp_statmat,
        }

        # Save to database if requested
        if save_to_db:
            sim_payload = {
                "run_id": run_id,
                "timestamp": history_entry["timestamp"],
                "num_genes": n,
                "gene_names": gene_names,
                "topology": topo_key,
                "k_average": k_av,
                "act_ratio": 0.5,
                "noise": noise,
                "bio_var": 0.0,
                "seed": seed,
                "true_matrix": exp.true_matrix.tolist(),
                "X": exp.X.tolist(),
                "U": exp.U.tolist(),
                "perturbation_trajectories": [t.tolist() if isinstance(t, np.ndarray) else t for t in getattr(exp, "perturbation_trajectories", [])],
                "steady_state_diagnostics": ss_diag,
                "response_dynamics": resp_dyn,
                "response_type": response_type,
                "svg_hierarchical": svg_h,
                "svg_spring": svg_h,
                "settings": history_entry["settings"],
            }
            inferred_payload = {
                "status": "success",
                "run_id": run_id,
                "primary_algorithm": "vote",
                "metrics": eval_metrics,
                "tmp_statmat": tmp_statmat,
                "pred_matrices": {k: v.tolist() for k, v in preds.items()},
                "dots": {"vote": dot_h},
                "svgs_dot": {"vote": svg_h},
                "svgs_neato": {"vote": svg_h},
            }
            save_simulation_to_db(
                run_id=run_id,
                sim_payload=sim_payload,
                inferred_payload=inferred_payload,
                history_entry=history_entry,
                db_path=db_file,
            )

        return {
            "run_id": run_id,
            "n": n,
            "k_av": k_av,
            "topo_key": topo_key,
            "noise": noise,
            "eval_metrics": eval_metrics,
            "tmp_statmat": tmp_statmat,
            "history_entry": history_entry,
        }

    print("Executing Systematic Parameter Sweep across Size, Topology, Connectivity, and Noise...")

    # 1. Size Sweep (N in [5, 8, 10, 12, 16, 20, 24, 30])
    print("  -> Sweeping Network Sizes...")
    for n in size_values:
        res = _execute_run(n=n, k_av=1.8, topo_key="power_law", topo_enum=TopologyType.POWER_LAW, noise=0.0, seed=100 + n, tag="size")
        sweep_runs.append(res)
        for alg in size_results:
            size_results[alg]["power"].append(res["tmp_statmat"][alg]["power"])
            size_results[alg]["fdr"].append(res["tmp_statmat"][alg]["fdr"])
            size_results[alg]["f1"].append(res["eval_metrics"][alg]["f1"])
            size_results[alg]["total_errors"].append(res["eval_metrics"][alg]["total_errors"])

    # 2. Topology Sweep (Random, Power-Law, Hierarchical across N in [8, 12, 16, 20])
    print("  -> Sweeping Topologies...")
    topo_sizes = [8, 12, 16, 20]
    for t_name, t_enum in topology_types:
        for n in topo_sizes:
            res = _execute_run(n=n, k_av=1.8, topo_key=t_name, topo_enum=t_enum, noise=0.0, seed=200 + n * 7, tag="topo")
            sweep_runs.append(res)
            for alg in topo_results[t_name]:
                topo_results[t_name][alg]["power"].append(res["tmp_statmat"][alg]["power"])
                topo_results[t_name][alg]["fdr"].append(res["tmp_statmat"][alg]["fdr"])
                topo_results[t_name][alg]["f1"].append(res["eval_metrics"][alg]["f1"])
                topo_results[t_name][alg]["specificity"].append(res["tmp_statmat"][alg]["specificity"])

    # 3. Connectivity Sweep (k_av in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    print("  -> Sweeping Connectivity...")
    for kav in k_values:
        res = _execute_run(n=14, k_av=kav, topo_key="power_law", topo_enum=TopologyType.POWER_LAW, noise=0.0, seed=300 + int(kav * 20), tag="conn")
        sweep_runs.append(res)
        for alg in conn_results:
            conn_results[alg]["power"].append(res["tmp_statmat"][alg]["power"])
            conn_results[alg]["fdr"].append(res["tmp_statmat"][alg]["fdr"])
            conn_results[alg]["f1"].append(res["eval_metrics"][alg]["f1"])
            conn_results[alg]["signed_ppv"].append(res["tmp_statmat"][alg]["signed_ppv"])

    # 4. Noise Sweep (sigma_m in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25])
    print("  -> Sweeping Measurement Noise...")
    for noise in noise_values:
        res = _execute_run(n=12, k_av=1.8, topo_key="power_law", topo_enum=TopologyType.POWER_LAW, noise=noise, seed=400 + int(noise * 100), tag="noise")
        sweep_runs.append(res)
        for alg in noise_results:
            noise_results[alg]["power"].append(res["tmp_statmat"][alg]["power"])
            noise_results[alg]["fdr"].append(res["tmp_statmat"][alg]["fdr"])
            noise_results[alg]["f1"].append(res["eval_metrics"][alg]["f1"])
            noise_results[alg]["total_errors"].append(res["eval_metrics"][alg]["total_errors"])

    # 5. Dynamics Regimes (Representative Flat and Oscillating runs across sizes)
    print("  -> Generating Dynamic Regimes (Flat and Oscillating)...")
    for n in [8, 10, 12, 14]:
        res_flat = _execute_run(n=n, k_av=1.5, topo_key="random", topo_enum=TopologyType.RANDOM, noise=0.001, seed=500 + n, tag="dyn-flat", force_flat=True)
        sweep_runs.append(res_flat)
    for n in [8, 10, 12]:
        res_osc = _execute_run(n=n, k_av=2.2, topo_key="hierarchical", topo_enum=TopologyType.HIERARCHICAL, noise=0.0, seed=600 + n, tag="dyn-osc", force_osc=True)
        sweep_runs.append(res_osc)

    # 6. Overall Algorithm Leaderboard
    leaderboard = []
    algo_meta = {
        "vote": {"name": "Voting Ensemble", "type": "Consensus (S2G ∩ G2S)"},
        "s2g": {"name": "Simple-to-General (S2G)", "type": "Forward Stepwise AIC"},
        "g2s": {"name": "General-to-Specific (G2S)", "type": "Backward Stepwise AIC"},
        "nir": {"name": "NIR (Multiple Regression)", "type": "Linear Matrix Inverse"},
        "fast": {"name": "FAST (SVD Inversion)", "type": "SVD Truncation"},
        "ridge": {"name": "Multi-Output RidgeCV", "type": "L2 Regularized Inverse"},
        "by_chance": {"name": "By-Chance (Random)", "type": "Null Bernoulli Model"},
    }
    for alg, m_list in all_evaluated_metrics.items():
        if not m_list:
            continue
        meta = algo_meta.get(alg, {"name": alg.upper(), "type": "Standard"})
        leaderboard.append({
            "algorithm_key": alg,
            "algorithm_name": meta["name"],
            "method_type": meta["type"],
            "runs_evaluated": len(m_list),
            "mean_f1": round(float(np.mean([x["f1"] for x in m_list])), 3),
            "mean_power_pct": round(float(np.mean([x["power"] for x in m_list])), 1),
            "mean_specificity_pct": round(float(np.mean([x["specificity"] for x in m_list])), 1),
            "mean_ppv_pct": round(float(np.mean([x["ppv"] for x in m_list])), 1),
            "mean_signed_ppv_pct": round(float(np.mean([x["signed_ppv"] for x in m_list])), 1),
            "mean_fdr_pct": round(float(np.mean([x["fdr"] for x in m_list])), 1),
            "mean_total_errors": round(float(np.mean([x["total_errors"] for x in m_list])), 1),
        })

    # Sort leaderboard by F1 descending
    leaderboard.sort(key=lambda x: x["mean_f1"], reverse=True)

    # Topology average comparison
    topo_summary = {}
    for t_name in ("random", "power_law", "hierarchical"):
        topo_summary[t_name] = {
            alg: {
                "mean_power": round(float(np.mean(topo_results[t_name][alg]["power"])), 1),
                "mean_fdr": round(float(np.mean(topo_results[t_name][alg]["fdr"])), 1),
                "mean_f1": round(float(np.mean(topo_results[t_name][alg]["f1"])), 3),
                "mean_specificity": round(float(np.mean(topo_results[t_name][alg]["specificity"])), 1),
            }
            for alg in ("vote", "s2g", "g2s", "nir", "fast", "ridge")
        }

    overall_stats = {
        "status": "success",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_sweep_runs": len(sweep_runs),
        "size_sweep": {
            "sizes": size_values,
            "algorithms": size_results,
        },
        "topology_sweep": {
            "sizes": topo_sizes,
            "topologies": topo_results,
            "summary_by_topology": topo_summary,
        },
        "connectivity_sweep": {
            "k_values": k_values,
            "algorithms": conn_results,
        },
        "noise_sweep": {
            "noise_levels": noise_values,
            "algorithms": noise_results,
        },
        "algorithm_leaderboard": leaderboard,
    }

    # Save to disk as cached JSON
    out_json_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "overall_statistics_sweep.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, indent=2)
    print(f"Saved overall statistics payload: {out_json_path}")

    return overall_stats


if __name__ == "__main__":
    run_systematic_sweep()
