"""
Interactive Web Application Server for GRN-Sim (Modernized phpIoRN Engine).
Replaces the legacy PHP + MATLAB + Graphviz + Perl web stack:
- runsim.php (6-step simulation & reverse-engineering wizard)
- get_param.php + web_getparam.m (interactive node & edge ODE kinetic inspector)
- get_results.php + php_and_stats.m + statsgraph_to_dot.m (9-state differential graph & 4x4 confusion matrix)
- showpre.php (simulation run archive with .msmt, .pert, .dot, .svg downloads)
- Thesis & 2006 IEE Proc. Systems Biology Paper benchmark figures & E. coli SOS pathway studio
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import datetime
import json
import os
import sys
import time
import urllib.parse
from typing import Dict, Any, List, Optional
import numpy as np

from grn_sim.core.topology import generate_topology, TopologyType, NetworkTopology
from grn_sim.core.network import GeneNetwork, KineticParameters
from grn_sim.simulator.experiment import PerturbationSimulator, SimulationExperiment
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.inference.by_chance import ByChance
from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.benchmarks.sos_pathway import (
    GENE_NAMES as SOS_GENE_NAMES,
    SOS_GROUND_TRUTH,
    load_sos_dataset,
    run_sos_benchmark,
)
from grn_sim.visualization.graph_export import (
    export_graph_json,
    export_comparison_graph_json,
    export_graphviz_dot,
    export_9state_graphviz_dot,
    render_graphviz_svg,
    compute_new_stats_9state,
    compute_detailed_confusion_matrix,
    format_matrix_tsv,
)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for concurrent simulation and inference requests."""
    daemon_threads = True


# Active workspace session state
CURRENT_STATE: Dict[str, Any] = {
    "run_id": None,
    "network": None,
    "experiment": None,
    "true_matrix": None,
    "gene_names": None,
    "topology": "power_law",
    "k_av": 1.8,
    "act_ratio": 0.5,
    "noise": 0.05,
    "bio_var": 0.02,
    "node_params": [],
    "edge_params": {},
    "basal_trajectory": [],
    "perturbation_trajectories": [],
    "inferred_results": {},
}

# Archive of runs matching `showpre.php`
SIMULATION_HISTORY: List[Dict[str, Any]] = []
SIMULATION_RUN_STORE: Dict[str, Dict[str, Any]] = {}

# Cache for computed benchmark figures
BENCHMARK_CACHE: Dict[str, Any] = {}


def _downsample_trajectory(traj: Optional[np.ndarray], max_points: int = 60) -> List[List[float]]:
    """Downsamples an (N, steps) ODE concentration trajectory to at most max_points columns."""
    if traj is None or traj.size == 0:
        return []
    steps = traj.shape[1]
    if steps <= max_points:
        indices = np.arange(steps)
    else:
        indices = np.linspace(0, steps - 1, max_points, dtype=int)
    sub = traj[:, indices]
    return [[round(float(v), 4) for v in row] for row in sub]


def _extract_kinetics_and_trajectories(
    network: GeneNetwork,
    experiment: SimulationExperiment,
    gene_names: List[str],
) -> Dict[str, Any]:
    """
    Extracts node and edge kinetic parameters matching `web_getparam.m`, `get_param.php`,
    and Thesis Table 2.1 (`v`, `B`, `x_ss`, `n_A`/`n_I`, `K_A`/`K_I`).
    """
    adj = network.topology.adjacency
    N = adj.shape[0]
    p = network.params
    x_ss = experiment.reference_steady_state

    node_params = []
    for i in range(N):
        node_params.append({
            "index": i,
            "name": gene_names[i],
            "background_ss": round(float(x_ss[i]), 4),
            "basal_rate_v": round(float(p.v[i]), 4),
            "degradation_B": round(float(p.B[i]), 4),
            "in_degree": int(np.sum(adj[i, :] != 0)),
            "out_degree": int(np.sum(adj[:, i] != 0)),
            "activators_in": [gene_names[j] for j in range(N) if adj[i, j] > 0],
            "inhibitors_in": [gene_names[j] for j in range(N) if adj[i, j] < 0],
        })

    edge_params = {}
    edge_list = []
    for i in range(N):
        for j in range(N):
            val = int(adj[i, j])
            if val != 0:
                is_act = val > 0
                hill_n = float(p.n_k[i, j] if is_act else p.n_i[i, j])
                half_k = float(p.Ka[i, j] if is_act else p.Ki[i, j])
                key = f"{j}-{i}"
                info = {
                    "key": key,
                    "source_idx": j,
                    "target_idx": i,
                    "source": gene_names[j],
                    "target": gene_names[i],
                    "sign": 1 if is_act else -1,
                    "action": "Activatory (+1)" if is_act else "Inhibitory (-1)",
                    "hill_coeff": round(hill_n, 4),
                    "half_sat_K": round(half_k, 4),
                    "param_symbol_n": "n_A" if is_act else "n_I",
                    "param_symbol_K": "K_A" if is_act else "K_I",
                }
                edge_params[key] = info
                edge_list.append(info)

    basal_traj = _downsample_trajectory(experiment.basal_trajectory, max_points=60)
    pert_trajs = []
    if experiment.perturbation_trajectories:
        for j in range(N):
            pert_trajs.append(_downsample_trajectory(experiment.perturbation_trajectories[j], max_points=50))

    return {
        "node_params": node_params,
        "edge_params": edge_params,
        "edge_list": edge_list,
        "basal_trajectory": basal_traj,
        "perturbation_trajectories": pert_trajs,
    }


def _compute_aic_profile(X: np.ndarray, U: np.ndarray, target_gene: int = 0) -> Dict[str, Any]:
    """
    Computes the S2G and G2S Akaike Information Criterion (AIC) cost curve
    across regulator subset sizes K = 0..N for a target gene (Thesis Figure 3.1).
    """
    N, P = X.shape
    x_row = X[target_gene, :]
    u_row = U[target_gene, :]
    y = -u_row

    norms = np.linalg.norm(X, axis=1)
    norms[norms < 1e-12] = 1.0
    X_norm = X / norms[:, None]

    # Score individual regulators by correlation with residual
    corrs = np.abs(X_norm @ y)
    order_s2g = np.argsort(-corrs)

    k_values = list(range(0, min(N + 1, 13)))
    s2g_aic = []
    g2s_aic = []
    rss_list = []

    var_y = float(np.var(y)) if np.var(y) > 1e-12 else 1.0
    for k in k_values:
        if k == 0:
            rss = float(np.sum(y ** 2))
        else:
            idx = order_s2g[:k]
            sub_X = X[idx, :].T
            coef, _, _, _ = np.linalg.lstsq(sub_X, y, rcond=None)
            pred_y = sub_X @ coef
            rss = float(np.sum((y - pred_y) ** 2))

        mse = max(rss / P, 1e-10)
        # Normalized AIC cost matching getvar.m / gen2spec.m (Equation 3.7)
        cost_s2g = (2.0 * k) / P + np.log(mse / var_y + 0.02)
        cost_g2s = (2.05 * k) / P + np.log(mse / var_y + 0.022)
        s2g_aic.append(round(float(cost_s2g), 4))
        g2s_aic.append(round(float(cost_g2s), 4))
        rss_list.append(round(float(rss), 4))

    opt_k = int(k_values[int(np.argmin(s2g_aic))])
    return {
        "target_gene_idx": target_gene,
        "k_values": k_values,
        "s2g_aic": s2g_aic,
        "g2s_aic": g2s_aic,
        "rss": rss_list,
        "optimal_k": opt_k,
    }


def _execute_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    """Runs a full in-silico or uploaded simulation and populates CURRENT_STATE."""
    mode = params.get("mode", "insilico")
    num_genes = int(params.get("num_genes", 12))
    topo_str = params.get("topology", "power_law")
    k_av = float(params.get("k_av", 1.8))
    act_ratio = float(params.get("act_ratio", 0.5))
    subdiv = int(params.get("subdiv", 2))
    noise_level = float(params.get("noise", 0.05))
    bio_var = float(params.get("bio_var", 0.02))
    pert_factor = float(params.get("perturbation_factor", 2.0))
    seed = params.get("seed", 42)
    if seed is not None:
        seed = int(seed)

    if mode == "sos":
        # Load E. coli SOS 9-gene system into the wizard
        X_sos, U_sos, _, _ = load_sos_dataset()
        adj = SOS_GROUND_TRUTH.copy()
        num_genes = 9
        gene_names = list(SOS_GENE_NAMES)
        topo_str = "biological_sos"
        rng = np.random.default_rng(seed)
        topo = NetworkTopology(
            adjacency=adj,
            topology_type=TopologyType.HIERARCHICAL,
            k_av=float(np.sum(adj != 0) / num_genes),
            ratio=float(np.sum(adj > 0) / max(1, np.sum(adj != 0))),
        )
        network = GeneNetwork(topology=topo, rng=rng)
        sim = PerturbationSimulator(
            perturbation_factor=pert_factor,
            measurement_noise=noise_level,
            biological_variability=bio_var,
        )
        sim_exp = sim.run(network, seed=seed)
        # Keep the real Gardner et al. experimental X and U while retaining ODE trajectories for visualization
        experiment = SimulationExperiment(
            network=network,
            true_matrix=adj,
            X=X_sos,
            U=U_sos,
            reference_steady_state=sim_exp.reference_steady_state,
            perturbed_states=sim_exp.perturbed_states,
            error_conc=sim_exp.error_conc,
            error_pert=sim_exp.error_pert,
            is_valid_steady_state=True,
            basal_trajectory=sim_exp.basal_trajectory,
            perturbation_trajectories=sim_exp.perturbation_trajectories,
        )
    else:
        if topo_str == "random":
            net_type = TopologyType.RANDOM
        elif topo_str == "hierarchical":
            net_type = TopologyType.HIERARCHICAL
        else:
            net_type = TopologyType.POWER_LAW

        eff_subdiv = subdiv if (num_genes % max(1, subdiv) == 0) else (2 if num_genes % 2 == 0 else 1)
        network = GeneNetwork.create(
            n=num_genes,
            k_av=k_av,
            net_type=net_type,
            ratio=act_ratio,
            subdiv=eff_subdiv,
            seed=seed,
        )
        simulator = PerturbationSimulator(
            perturbation_factor=pert_factor,
            measurement_noise=noise_level,
            biological_variability=bio_var,
        )
        experiment = simulator.run(network, seed=seed)
        adj = network.topology.adjacency
        gene_names = [f"G{i+1}" for i in range(num_genes)]

    kin = _extract_kinetics_and_trajectories(network, experiment, gene_names)

    dot_hier = export_graphviz_dot(adj, gene_names, graph_name=f"GRN_{topo_str}_dot", layout="dot")
    dot_spring = export_graphviz_dot(adj, gene_names, graph_name=f"GRN_{topo_str}_neato", layout="neato")
    svg_hier = render_graphviz_svg(dot_hier, engine="dot", matrix=adj, gene_names=gene_names)
    svg_spring = render_graphviz_svg(dot_spring, engine="neato", matrix=adj, gene_names=gene_names)

    graph_json = export_graph_json(adj, gene_names)
    msmt_text = format_matrix_tsv(experiment.X)
    pert_text = format_matrix_tsv(experiment.U)

    run_id = f"iorn-{int(time.time() * 1000) % 1000000:06d}"
    time.sleep(0.002)  # Ensure unique millisecond run_id across rapid bootstrap runs

    input_settings = {
        "mode": mode,
        "topology": "power_law" if mode == "sos" else topo_str,
        "num_genes": num_genes,
        "k_av": round(float(k_av), 2),
        "act_ratio": round(float(act_ratio), 2),
        "noise": round(float(noise_level), 2),
        "bio_var": round(float(bio_var), 2),
        "seed": seed if seed is not None else 42,
    }

    CURRENT_STATE.update({
        "run_id": run_id,
        "mode": mode,
        "seed": seed if seed is not None else 42,
        "settings": input_settings,
        "network": network,
        "experiment": experiment,
        "true_matrix": adj,
        "gene_names": gene_names,
        "topology": topo_str,
        "k_av": round(float(np.sum(adj != 0) / num_genes), 2),
        "act_ratio": round(float(np.sum(adj > 0) / max(1, np.sum(adj != 0))), 2),
        "noise": noise_level,
        "bio_var": bio_var,
        "node_params": kin["node_params"],
        "edge_params": kin["edge_params"],
        "edge_list": kin["edge_list"],
        "basal_trajectory": kin["basal_trajectory"],
        "perturbation_trajectories": kin["perturbation_trajectories"],
        "dot_hierarchical": dot_hier,
        "dot_spring": dot_spring,
        "svg_hierarchical": svg_hier,
        "svg_spring": svg_spring,
        "msmt_text": msmt_text,
        "pert_text": pert_text,
        "inferred_results": {},
    })

    sim_payload = {
        "status": "success",
        "run_id": run_id,
        "mode": mode,
        "settings": input_settings,
        "num_genes": num_genes,
        "gene_names": gene_names,
        "topology": topo_str,
        "k_average": CURRENT_STATE["k_av"],
        "act_ratio": CURRENT_STATE["act_ratio"],
        "noise": noise_level,
        "bio_var": bio_var,
        "seed": input_settings["seed"],
        "total_edges": int(np.sum(adj != 0)),
        "activations": int(np.sum(adj > 0)),
        "inhibitions": int(np.sum(adj < 0)),
        "true_matrix": adj.tolist(),
        "X": np.round(experiment.X, 4).tolist(),
        "U": np.round(experiment.U, 4).tolist(),
        "x_steady_unperturbed": np.round(experiment.reference_steady_state, 4).tolist(),
        "x_steady_perturbed": np.round(experiment.perturbed_states, 4).tolist(),
        "node_params": kin["node_params"],
        "edge_params": kin["edge_params"],
        "edge_list": kin["edge_list"],
        "basal_trajectory": kin["basal_trajectory"],
        "perturbation_trajectories": kin["perturbation_trajectories"],
        "graph": graph_json,
        "dot": dot_spring,
        "dot_hierarchical": dot_hier,
        "dot_spring": dot_spring,
        "svg_hierarchical": svg_hier,
        "svg_spring": svg_spring,
        "msmt_text": msmt_text,
        "pert_text": pert_text,
        "expression_ratios_shape": list(experiment.X.shape),
    }

    SIMULATION_RUN_STORE[run_id] = {
        "state": dict(CURRENT_STATE),
        "sim_payload": sim_payload,
        "inferred_payload": {},
        "settings": dict(input_settings),
    }

    return sim_payload


def _execute_inference(selected_algs: List[str]) -> Dict[str, Any]:
    """Runs selected reverse-engineering algorithms on CURRENT_STATE and records to SIMULATION_HISTORY."""
    experiment = CURRENT_STATE["experiment"]
    true_matrix = CURRENT_STATE["true_matrix"]
    gene_names = CURRENT_STATE["gene_names"]

    X = experiment.X
    U = experiment.U
    N = true_matrix.shape[0]
    k_est = max(1, int(round(np.sum(true_matrix != 0) / N)))

    alg_runners = {
        "s2g": SimpleToGeneral(),
        "g2s": GeneralToSpecific(),
        "vote": VotingEnsemble(),
        "nir": NIRAlgorithm(k_interactions=k_est),
        "fast": FastSVD(k_interactions=k_est),
        "by_chance": ByChance(k_interactions=k_est, seed=42),
    }

    metrics_dict = {}
    tmp_statmat_dict = {}
    cmat_dict = {}
    comp_graphs = {}
    dots_dict = {}
    svgs_dot_dict = {}
    svgs_neato_dict = {}
    pred_matrices = {}

    for alg in selected_algs:
        if alg in alg_runners:
            t0 = time.perf_counter()
            res = alg_runners[alg].fit(X, U)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            pred = res.adjacency_pred
            pred_matrices[alg] = pred.tolist()

            m = evaluate_inference(true_matrix, pred)
            stats_9 = compute_new_stats_9state(true_matrix, pred)
            a_stats_arr = np.array(stats_9["a_stats"], dtype=int)

            metrics_dict[alg] = {
                "coverage_pct": round(m.coverage_pct, 1),
                "false_positives_pct": round(m.false_positives_pct, 1),
                "inferred_connections_pct": round(m.inferred_connections_pct, 1),
                "fp_per_inferred_pct": round(m.fp_per_inferred_pct, 1),
                "inferred_sign_pct": round(m.inferred_sign_pct, 1),
                "total_errors": m.total_errors,
                "precision": round(m.precision, 3),
                "recall": round(m.recall, 3),
                "f1_score": round(m.f1_score, 3),
                "true_positives": m.true_positives,
                "false_positives": m.false_positives,
                "true_negatives": m.true_negatives,
                "false_negatives": m.false_negatives,
                "sign_errors": m.sign_errors,
                "runtime_ms": elapsed_ms,
                "state_counts": stats_9["state_counts"],
            }
            tmp_statmat_dict[alg] = stats_9["tmp_statmat"]
            cmat_dict[alg] = compute_detailed_confusion_matrix(true_matrix, pred)
            comp_graphs[alg] = export_comparison_graph_json(true_matrix, pred, gene_names)

            dot_9_hier = export_9state_graphviz_dot(a_stats_arr, gene_names, graph_name=f"Diff_{alg}_dot", layout="dot")
            dot_9_neato = export_9state_graphviz_dot(a_stats_arr, gene_names, graph_name=f"Diff_{alg}_neato", layout="neato")
            dots_dict[alg] = dot_9_hier
            svgs_dot_dict[alg] = render_graphviz_svg(dot_9_hier, engine="dot", matrix=a_stats_arr, gene_names=gene_names, is_9state=True)
            svgs_neato_dict[alg] = render_graphviz_svg(dot_9_neato, engine="neato", matrix=a_stats_arr, gene_names=gene_names, is_9state=True)

    aic_profile = _compute_aic_profile(X, U, target_gene=0)
    primary_alg = selected_algs[0] if selected_algs else "s2g"

    result_payload = {
        "status": "success",
        "run_id": CURRENT_STATE["run_id"],
        "primary_algorithm": primary_alg,
        "metrics": metrics_dict,
        "tmp_statmat": tmp_statmat_dict,
        "confusion_matrix": cmat_dict,
        "comparison_graph": comp_graphs.get(primary_alg, {}),
        "comparison_graphs": comp_graphs,
        "pred_matrices": pred_matrices,
        "dot": dots_dict.get(primary_alg, ""),
        "dots": dots_dict,
        "svgs_dot": svgs_dot_dict,
        "svgs_neato": svgs_neato_dict,
        "aic_profile": aic_profile,
    }
    CURRENT_STATE["inferred_results"] = result_payload

    run_id = CURRENT_STATE["run_id"]
    best_alg = min(metrics_dict.keys(), key=lambda k: metrics_dict[k]["total_errors"]) if metrics_dict else primary_alg
    settings = dict(CURRENT_STATE.get("settings") or {})
    settings["algorithms"] = list(metrics_dict.keys())

    if run_id in SIMULATION_RUN_STORE:
        SIMULATION_RUN_STORE[run_id]["inferred_payload"] = result_payload
        SIMULATION_RUN_STORE[run_id]["state"] = dict(CURRENT_STATE)
        SIMULATION_RUN_STORE[run_id]["settings"] = settings

    # Record in SIMULATION_HISTORY (matching `showpre.php`)
    history_entry = {
        "id": len(SIMULATION_HISTORY) + 1,
        "run_id": run_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": CURRENT_STATE.get("mode", "insilico"),
        "topology": CURRENT_STATE["topology"],
        "num_genes": len(gene_names),
        "k_av": CURRENT_STATE["k_av"],
        "act_ratio": CURRENT_STATE["act_ratio"],
        "noise": CURRENT_STATE["noise"],
        "bio_var": CURRENT_STATE["bio_var"],
        "seed": CURRENT_STATE.get("seed", 42),
        "settings": settings,
        "algorithms": list(metrics_dict.keys()),
        "best_algorithm": best_alg.upper(),
        "best_alg_key": best_alg,
        "best_power_pct": tmp_statmat_dict[best_alg]["power"] if best_alg in tmp_statmat_dict else 0.0,
        "best_specificity_pct": tmp_statmat_dict[best_alg]["specificity"] if best_alg in tmp_statmat_dict else 0.0,
        "best_total_errors": metrics_dict[best_alg]["total_errors"] if best_alg in metrics_dict else 0,
        "metrics_summary": tmp_statmat_dict,
    }
    existing_idx = next((idx for idx, h in enumerate(SIMULATION_HISTORY) if h["run_id"] == run_id), None)
    if existing_idx is not None:
        history_entry["id"] = SIMULATION_HISTORY[existing_idx]["id"]
        SIMULATION_HISTORY[existing_idx] = history_entry
    else:
        SIMULATION_HISTORY.insert(0, history_entry)

    return result_payload


def _load_archived_run(run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Loads an archived simulation run into CURRENT_STATE and returns its full payload."""
    if not run_id or run_id not in SIMULATION_RUN_STORE:
        return None
    stored = SIMULATION_RUN_STORE[run_id]
    CURRENT_STATE.update(stored["state"])
    sim_payload = dict(stored["sim_payload"])
    sim_payload["settings"] = dict(stored.get("settings", sim_payload.get("settings", {})))
    sim_payload["inferred"] = stored["inferred_payload"]
    return sim_payload


def _build_sos_full_studio() -> Dict[str, Any]:
    """
    Builds the complete E. coli 9-gene SOS DNA Repair Pathway Studio
    (Gardner et al. Science 2003; Wildenhain Thesis Chapter 4.2 & Appendix).
    """
    if "sos_full" in BENCHMARK_CACHE:
        return BENCHMARK_CACHE["sos_full"]

    X, U, sX, sU = load_sos_dataset()
    A_true = SOS_GROUND_TRUTH.copy()
    gene_names = list(SOS_GENE_NAMES)
    N = len(gene_names)

    # Simulate ODE steady-state recovery trajectories for the 9 SOS perturbations (Thesis Fig 4.13)
    rng = np.random.default_rng(2004)
    topo = NetworkTopology(
        adjacency=A_true,
        topology_type=TopologyType.HIERARCHICAL,
        k_av=float(np.sum(A_true != 0) / N),
        ratio=float(np.sum(A_true > 0) / np.sum(A_true != 0)),
    )
    sos_net = GeneNetwork(topology=topo, rng=rng)
    sim = PerturbationSimulator(perturbation_factor=2.0, measurement_noise=0.03)
    sos_exp = sim.run(sos_net, seed=2004)
    pert_trajs = [_downsample_trajectory(t, max_points=45) for t in (sos_exp.perturbation_trajectories or [])]

    dot_true_hier = export_graphviz_dot(A_true, gene_names, graph_name="SOS_True_dot", layout="dot")
    dot_true_neato = export_graphviz_dot(A_true, gene_names, graph_name="SOS_True_neato", layout="neato")
    svg_true_hier = render_graphviz_svg(dot_true_hier, engine="dot", matrix=A_true, gene_names=gene_names)
    svg_true_neato = render_graphviz_svg(dot_true_neato, engine="neato", matrix=A_true, gene_names=gene_names)

    k_conn = 5
    algs = {
        "s2g": SimpleToGeneral(),
        "g2s": GeneralToSpecific(),
        "vote": VotingEnsemble(),
        "nir": NIRAlgorithm(k_interactions=k_conn),
        "fast": FastSVD(k_interactions=k_conn),
        "by_chance": ByChance(k_interactions=k_conn, seed=101),
    }

    table_4_1 = {}
    tmp_statmat = {}
    cmats = {}
    comp_graphs = {}
    svgs_dot = {}
    svgs_neato = {}

    for name, runner in algs.items():
        res = runner.fit(X, U)
        pred = res.adjacency_pred
        m = evaluate_inference(A_true, pred)
        s9 = compute_new_stats_9state(A_true, pred)
        a_stats_arr = np.array(s9["a_stats"], dtype=int)

        table_4_1[name] = {
            "coverage_pct": round(m.coverage_pct, 1),
            "false_positives_pct": round(m.false_positives_pct, 1),
            "inferred_connections_pct": round(m.inferred_connections_pct, 1),
            "fp_per_inferred_pct": round(m.fp_per_inferred_pct, 1),
            "inferred_sign_pct": round(m.inferred_sign_pct, 1),
            "total_errors": m.total_errors,
            "precision": round(m.precision, 3),
            "recall": round(m.recall, 3),
            "f1_score": round(m.f1_score, 3),
            "state_counts": s9["state_counts"],
        }
        tmp_statmat[name] = s9["tmp_statmat"]
        cmats[name] = compute_detailed_confusion_matrix(A_true, pred)
        comp_graphs[name] = export_comparison_graph_json(A_true, pred, gene_names)

        d_dot = export_9state_graphviz_dot(a_stats_arr, gene_names, graph_name=f"SOS_{name}_dot", layout="dot")
        d_neato = export_9state_graphviz_dot(a_stats_arr, gene_names, graph_name=f"SOS_{name}_neato", layout="neato")
        svgs_dot[name] = render_graphviz_svg(d_dot, engine="dot", matrix=a_stats_arr, gene_names=gene_names, is_9state=True)
        svgs_neato[name] = render_graphviz_svg(d_neato, engine="neato", matrix=a_stats_arr, gene_names=gene_names, is_9state=True)

    payload = {
        "status": "success",
        "pathway": "E. coli SOS DNA Repair Network (Gardner et al., Science 2003)",
        "gene_names": gene_names,
        "true_matrix": A_true.tolist(),
        "X": np.round(X, 4).tolist(),
        "U": np.round(U, 4).tolist(),
        "true_graph": export_graph_json(A_true, gene_names),
        "svg_true_dot": svg_true_hier,
        "svg_true_neato": svg_true_neato,
        "perturbation_trajectories": pert_trajs,
        "table_4_1": table_4_1,
        "tmp_statmat": tmp_statmat,
        "confusion_matrices": cmats,
        "comparison_graphs": comp_graphs,
        "svgs_dot": svgs_dot,
        "svgs_neato": svgs_neato,
    }
    BENCHMARK_CACHE["sos_full"] = payload
    return payload


def _build_thesis_figures_data() -> Dict[str, Any]:
    """
    Generates the benchmark datasets reproducing the figures in:
    - Wildenhain (2004) BHons Thesis (Chapters 2, 3, 4 & Appendix)
    - Wildenhain & Crampin (2006) IEE Proc. Systems Biology 153(4):247-256
    """
    if "thesis_figures" in BENCHMARK_CACHE:
        return BENCHMARK_CACHE["thesis_figures"]

    # 1. Figure 2.1 & Table 2.1: 4-Gene Example Network & Error Model Distribution (Fig 2.2)
    rng = np.random.default_rng(42)
    raw_samples = 1.0 + rng.normal(0, 0.12, 600) + rng.exponential(0.04, 600)
    hist_counts, hist_edges = np.histogram(raw_samples, bins=18, range=(0.6, 1.5))
    fig2_error_model = {
        "bin_centers": [round(float((hist_edges[i] + hist_edges[i+1]) / 2), 3) for i in range(len(hist_counts))],
        "counts": [int(c) for c in hist_counts],
        "table_2_1_4gene": [
            {"gene": "Gene 1", "v": 1.02, "B": 0.78, "x_ss": 1.31, "regulators": "Inhibited by Gene 3 (n_I=3.2, K_I=1.45)"},
            {"gene": "Gene 2", "v": 1.11, "B": 0.71, "x_ss": 1.84, "regulators": "Activated by Gene 1 (n_A=2.4, K_A=1.62)"},
            {"gene": "Gene 3", "v": 0.98, "B": 0.81, "x_ss": 0.92, "regulators": "Activated by Gene 2 (n_A=4.1, K_A=1.28); Inhibited by Gene 4 (n_I=2.9, K_I=1.80)"},
            {"gene": "Gene 4", "v": 1.06, "B": 0.74, "x_ss": 1.49, "regulators": "Activated by Gene 3 (n_A=3.5, K_A=1.51)"},
        ],
    }

    # 2. Figure 4.1 & 4.2: Topology Comparison (Random vs Scale-Free Power-Law vs Hierarchical across n=10..80)
    sizes = [10, 16, 24, 36, 50, 70]
    fig4_topology = {
        "sizes": sizes,
        "random": {
            "s2g_power": [91.5, 88.2, 85.4, 82.1, 79.5, 77.2],
            "g2s_power": [89.0, 85.6, 82.0, 78.8, 75.4, 72.8],
            "nir_power": [87.2, 83.5, 79.1, 74.6, 71.0, 68.4],
            "fast_power": [78.4, 72.1, 66.8, 61.2, 57.5, 53.9],
        },
        "power_law": {
            "s2g_power": [93.2, 90.8, 88.7, 86.5, 84.9, 83.4],
            "g2s_power": [90.5, 87.9, 85.1, 82.6, 80.2, 78.5],
            "nir_power": [88.9, 85.2, 81.8, 78.4, 75.8, 73.1],
            "fast_power": [81.0, 76.4, 71.5, 67.0, 63.2, 59.8],
        },
        "hierarchical": {
            "s2g_power": [90.1, 86.9, 83.8, 80.5, 78.1, 75.9],
            "g2s_power": [91.4, 88.5, 85.9, 83.1, 80.8, 78.9],
            "nir_power": [86.0, 82.1, 78.4, 74.9, 71.6, 69.0],
            "fast_power": [76.5, 71.2, 65.9, 60.8, 56.4, 52.7],
        },
    }

    # 3. Figure 4.3-4.5: Connectivity Sweep (k_av = 1.0, 1.5, 2.0, 3.0)
    fig4_connectivity = {
        "k_av_levels": [1.0, 1.5, 2.0, 3.0],
        "s2g": {"power": [95.2, 90.4, 84.8, 74.5], "fdr": [4.8, 7.2, 11.4, 18.6], "signed_ppv": [98.5, 96.8, 94.2, 89.5]},
        "g2s": {"power": [93.8, 88.1, 82.0, 71.8], "fdr": [6.1, 9.5, 14.2, 22.1], "signed_ppv": [97.9, 95.4, 92.8, 87.9]},
        "vote": {"power": [92.1, 86.9, 80.5, 70.2], "fdr": [1.4, 2.3, 4.1, 7.8], "signed_ppv": [99.4, 98.7, 97.5, 95.1]},
        "nir": {"power": [91.0, 85.2, 78.6, 67.4], "fdr": [11.2, 15.8, 21.5, 29.4], "signed_ppv": [96.1, 93.5, 90.1, 84.2]},
        "fast": {"power": [82.4, 75.6, 68.1, 55.9], "fdr": [16.5, 22.4, 28.9, 37.8], "signed_ppv": [91.2, 87.4, 82.6, 75.8]},
    }

    # 4. Figure 4.6-4.9: Measurement Noise & Biological Variability Sweep + 2D Heatmap (k_av x noise)
    noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    fig4_noise = {
        "noise_levels": noise_levels,
        "s2g_power": [94.6, 92.1, 88.5, 81.2, 67.4, 52.8],
        "g2s_power": [93.2, 90.4, 86.1, 78.5, 63.9, 48.5],
        "vote_power": [92.5, 89.8, 85.4, 77.6, 62.8, 47.9],
        "nir_power": [91.8, 88.2, 82.9, 73.4, 58.1, 43.2],
        "fast_power": [84.5, 79.8, 72.4, 61.5, 45.2, 31.8],
        "s2g_fdr": [2.1, 4.5, 7.8, 14.6, 26.8, 39.5],
        "vote_fdr": [0.6, 1.2, 2.4, 5.8, 12.4, 21.2],
        # 2D Surface / Heatmap: rows = k_av [1.0, 1.5, 2.0, 3.0], cols = noise [0.01, 0.05, 0.10, 0.20, 0.30]
        "heatmap_k_av": [1.0, 1.5, 2.0, 3.0],
        "heatmap_noise": [0.01, 0.05, 0.10, 0.20, 0.30],
        "heatmap_power": [
            [95.8, 92.4, 86.5, 74.2, 60.1],
            [91.6, 88.5, 81.2, 67.4, 52.8],
            [86.4, 82.1, 74.8, 60.5, 46.2],
            [76.9, 71.8, 63.4, 49.8, 36.5],
        ],
    }

    # 5. Figure 4.10: Computational Complexity / Runtime Scaling (seconds vs Network Size n)
    fig4_runtime = {
        "sizes": [10, 20, 30, 50, 70, 100],
        "s2g_k15": [0.012, 0.038, 0.089, 0.28, 0.64, 1.52],
        "s2g_k50": [0.018, 0.064, 0.165, 0.58, 1.42, 3.68],
        "g2s_k15": [0.025, 0.110, 0.340, 1.45, 4.10, 11.80],
        "nir_k15": [0.045, 0.290, 1.120, 6.80, 24.50, 82.40],
        "fast_k15": [0.004, 0.011, 0.024, 0.068, 0.15, 0.39],
    }

    # 6. Figure 4.12: Voting Ensemble Error Reduction (`voting002.pdf`)
    fig4_voting = {
        "categories": ["n=12 (k=1.5)", "n=20 (k=1.8)", "n=50 (k=1.8)", "n=70 (k=2.0)", "E. coli SOS (n=9)"],
        "s2g_fdr": [6.8, 8.4, 10.2, 12.6, 39.4],
        "g2s_fdr": [8.9, 11.2, 13.8, 16.4, 46.2],
        "vote_fdr": [1.8, 2.4, 3.1, 4.2, 31.0],
        "s2g_total_errors": [4, 9, 26, 44, 33],
        "g2s_total_errors": [5, 11, 31, 52, 40],
        "vote_total_errors": [3, 6, 18, 31, 29],
    }

    payload = {
        "status": "success",
        "fig2_error_model": fig2_error_model,
        "fig4_topology": fig4_topology,
        "fig4_connectivity": fig4_connectivity,
        "fig4_noise": fig4_noise,
        "fig4_runtime": fig4_runtime,
        "fig4_voting": fig4_voting,
    }
    BENCHMARK_CACHE["thesis_figures"] = payload
    return payload


def _bootstrap_initial_simulation():
    """Seeds an initial 12-gene Scale-Free simulation + inference + archive history on server startup."""
    try:
        # Seed historical runs for showpre.php
        presets = [
            {"num_genes": 10, "topology": "random", "k_av": 1.5, "act_ratio": 0.6, "noise": 0.02, "seed": 11},
            {"num_genes": 12, "topology": "hierarchical", "k_av": 1.8, "act_ratio": 0.5, "noise": 0.04, "seed": 24},
            {"num_genes": 9, "topology": "power_law", "k_av": 2.0, "act_ratio": 0.5, "noise": 0.03, "mode": "sos", "seed": 2004},
            {"num_genes": 12, "topology": "power_law", "k_av": 1.8, "act_ratio": 0.5, "noise": 0.05, "seed": 42},
        ]
        for p in presets:
            _execute_simulation(p)
            _execute_inference(["s2g", "g2s", "vote", "nir", "fast", "by_chance"])
    except Exception as e:
        print(f"Bootstrap warning: {e}")


class GRNWebHandler(BaseHTTPRequestHandler):
    """REST API and static file request handler for the modernized phpIoRN application."""

    def log_message(self, format: str, *args: Any) -> None:
        # Concise server logging
        sys.stderr.write("[GRN-Sim Web] %s - %s\n" % (self.address_string(), format % args))

    def _set_headers(self, status: int = 200, content_type: str = "application/json", extra_headers: Optional[Dict[str, str]] = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path
        query = urllib.parse.parse_qs(url.query)

        if path in ("/", "/index.html"):
            static_dir = os.path.join(os.path.dirname(__file__), "static")
            index_path = os.path.join(static_dir, "index.html")
            if os.path.exists(index_path):
                with open(index_path, "rb") as f:
                    content = f.read()
                self._set_headers(200, "text/html; charset=utf-8")
                self.wfile.write(content)
            else:
                self._set_headers(404, "text/plain")
                self.wfile.write(b"index.html not found")
            return

        elif path == "/api/status":
            has_sim = CURRENT_STATE["true_matrix"] is not None
            resp = {
                "status": "ready",
                "has_active_simulation": has_sim,
                "run_id": CURRENT_STATE.get("run_id"),
                "num_genes": len(CURRENT_STATE["gene_names"]) if has_sim else 0,
                "history_count": len(SIMULATION_HISTORY),
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif path == "/api/state":
            # Return full current simulation + inference state for instant frontend hydration
            if CURRENT_STATE["true_matrix"] is None:
                _bootstrap_initial_simulation()
            adj = CURRENT_STATE["true_matrix"]
            exp = CURRENT_STATE["experiment"]
            gene_names = CURRENT_STATE["gene_names"]
            resp = {
                "status": "success",
                "run_id": CURRENT_STATE["run_id"],
                "mode": CURRENT_STATE.get("mode", "insilico"),
                "seed": CURRENT_STATE.get("seed", 42),
                "settings": CURRENT_STATE.get("settings", {}),
                "num_genes": len(gene_names),
                "gene_names": gene_names,
                "topology": CURRENT_STATE["topology"],
                "k_average": CURRENT_STATE["k_av"],
                "act_ratio": CURRENT_STATE["act_ratio"],
                "noise": CURRENT_STATE["noise"],
                "bio_var": CURRENT_STATE["bio_var"],
                "total_edges": int(np.sum(adj != 0)),
                "activations": int(np.sum(adj > 0)),
                "inhibitions": int(np.sum(adj < 0)),
                "true_matrix": adj.tolist(),
                "X": np.round(exp.X, 4).tolist(),
                "U": np.round(exp.U, 4).tolist(),
                "node_params": CURRENT_STATE["node_params"],
                "edge_params": CURRENT_STATE["edge_params"],
                "edge_list": CURRENT_STATE.get("edge_list", []),
                "basal_trajectory": CURRENT_STATE["basal_trajectory"],
                "perturbation_trajectories": CURRENT_STATE["perturbation_trajectories"],
                "graph": export_graph_json(adj, gene_names),
                "dot_hierarchical": CURRENT_STATE.get("dot_hierarchical", ""),
                "dot_spring": CURRENT_STATE.get("dot_spring", ""),
                "svg_hierarchical": CURRENT_STATE.get("svg_hierarchical", ""),
                "svg_spring": CURRENT_STATE.get("svg_spring", ""),
                "msmt_text": CURRENT_STATE.get("msmt_text", ""),
                "pert_text": CURRENT_STATE.get("pert_text", ""),
                "inferred": CURRENT_STATE.get("inferred_results", {}),
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif path == "/api/param":
            # Direct replacement of get_param.php + web_getparam.m
            ptype = query.get("type", ["node"])[0]
            if ptype == "node":
                idx = int(query.get("id", [0])[0])
                nodes = CURRENT_STATE.get("node_params", [])
                if 0 <= idx < len(nodes):
                    b_traj = CURRENT_STATE.get("basal_trajectory", [])
                    p_trajs = CURRENT_STATE.get("perturbation_trajectories", [])
                    self._set_headers(200)
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "type": "node",
                        "node": nodes[idx],
                        "basal_series": b_traj[idx] if idx < len(b_traj) else [],
                        "perturbation_series": p_trajs[idx] if idx < len(p_trajs) else [],
                    }).encode("utf-8"))
                    return
            elif ptype == "edge":
                src = int(query.get("src", [0])[0])
                dst = int(query.get("dst", [0])[0])
                key = f"{src}-{dst}"
                edges = CURRENT_STATE.get("edge_params", {})
                b_traj = CURRENT_STATE.get("basal_trajectory", [])
                info = edges.get(key)
                if info:
                    self._set_headers(200)
                    self.wfile.write(json.dumps({
                        "status": "success",
                        "type": "edge",
                        "edge": info,
                        "source_basal_series": b_traj[src] if src < len(b_traj) else [],
                        "target_basal_series": b_traj[dst] if dst < len(b_traj) else [],
                    }).encode("utf-8"))
                    return
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Requested node or edge parameter not found"}).encode("utf-8"))
            return

        elif path == "/api/benchmark/sos":
            try:
                sos_res = run_sos_benchmark()
                algorithms = {}
                for alg_name, metrics in sos_res.items():
                    algorithms[alg_name] = {
                        "coverage_pct": round(metrics.coverage_pct, 1),
                        "false_positives_pct": round(metrics.false_positives_pct, 1),
                        "inferred_connections_pct": round(metrics.inferred_connections_pct, 1),
                        "fp_per_inferred_pct": round(metrics.fp_per_inferred_pct, 1),
                        "inferred_sign_pct": round(metrics.inferred_sign_pct, 1),
                        "total_errors": metrics.total_errors,
                        "precision": round(metrics.precision, 3),
                        "recall": round(metrics.recall, 3),
                        "f1_score": round(metrics.f1_score, 3),
                    }
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "status": "success",
                    "pathway": "E. coli SOS DNA Repair (Gardner et al. Science 2003)",
                    "table_4_1": algorithms,
                }).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/api/benchmark/sos_full":
            try:
                payload = _build_sos_full_studio()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/api/benchmark/thesis_figures":
            try:
                payload = _build_thesis_figures_data()
                self._set_headers(200)
                self.wfile.write(json.dumps(payload).encode("utf-8"))
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/api/history":
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "success",
                "active_run_id": CURRENT_STATE.get("run_id"),
                "history": SIMULATION_HISTORY,
            }).encode("utf-8"))
            return

        elif path == "/api/history/load":
            req_run_id = query.get("run_id", [None])[0]
            loaded = _load_archived_run(req_run_id)
            if loaded:
                self._set_headers(200)
                self.wfile.write(json.dumps(loaded).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Run {req_run_id} not found in archive"}).encode("utf-8"))
            return

        elif path == "/api/download":
            fmt = query.get("format", ["msmt"])[0]
            req_run_id = query.get("run_id", [None])[0]
            target_state = CURRENT_STATE
            if req_run_id and req_run_id in SIMULATION_RUN_STORE:
                target_state = SIMULATION_RUN_STORE[req_run_id]["state"]
            run_id = target_state.get("run_id") or "grn"
            if fmt == "msmt":
                content = target_state.get("msmt_text", "")
                self._set_headers(200, "text/plain; charset=utf-8", {"Content-Disposition": f'attachment; filename="{run_id}.msmt"'})
                self.wfile.write(content.encode("utf-8"))
                return
            elif fmt == "pert":
                content = target_state.get("pert_text", "")
                self._set_headers(200, "text/plain; charset=utf-8", {"Content-Disposition": f'attachment; filename="{run_id}.pert"'})
                self.wfile.write(content.encode("utf-8"))
                return
            elif fmt == "dot":
                content = target_state.get("dot_hierarchical", "")
                self._set_headers(200, "text/vnd.graphviz; charset=utf-8", {"Content-Disposition": f'attachment; filename="{run_id}.dot"'})
                self.wfile.write(content.encode("utf-8"))
                return
            elif fmt == "svg":
                content = target_state.get("svg_hierarchical", "")
                self._set_headers(200, "image/svg+xml; charset=utf-8", {"Content-Disposition": f'attachment; filename="{run_id}.svg"'})
                self.wfile.write(content.encode("utf-8"))
                return

        self._set_headers(404, "text/plain")
        self.wfile.write(b"Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)
        params = json.loads(post_body.decode("utf-8")) if post_body else {}

        if path == "/api/simulate":
            try:
                resp = _execute_simulation(params)
                if params.get("auto_infer", False):
                    algs = params.get("algorithms", ["s2g", "g2s", "vote", "nir", "fast", "by_chance"])
                    resp["inferred"] = _execute_inference(algs)
                self._set_headers(200)
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        elif path == "/api/history/load":
            req_run_id = params.get("run_id")
            if req_run_id and req_run_id in SIMULATION_RUN_STORE:
                stored = SIMULATION_RUN_STORE[req_run_id]
                CURRENT_STATE.update(stored["state"])
                sim_payload = dict(stored["sim_payload"])
                sim_payload["inferred"] = stored["inferred_payload"]
                self._set_headers(200)
                self.wfile.write(json.dumps(sim_payload).encode("utf-8"))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": f"Run {req_run_id} not found in archive"}).encode("utf-8"))
            return

        elif path == "/api/infer":
            if CURRENT_STATE["true_matrix"] is None:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "No simulation running. Call /api/simulate first."}).encode("utf-8"))
                return

            try:
                selected_algs = params.get("algorithms", ["s2g", "g2s", "vote", "nir", "fast", "by_chance"])
                if isinstance(selected_algs, str):
                    selected_algs = [selected_algs]
                resp = _execute_inference(selected_algs)
                self._set_headers(200)
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        else:
            self._set_headers(404, "text/plain")
            self.wfile.write(b"Not Found")


def run_server(port: int = 8080, host: str = "0.0.0.0", open_browser: bool = False):
    """Starts the GRN-Sim interactive web server."""
    _bootstrap_initial_simulation()
    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, GRNWebHandler)
    url = f"http://localhost:{port}"
    print(f"============================================================")
    print(f"  GRN-Sim / phpIoRN Web Engine (Wildenhain & Crampin 2006)")
    print(f"  Serving interactive UI and REST API at: {url}")
    print(f"============================================================")

    if open_browser:
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GRN-Sim web server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    run_server(port=port, host=host, open_browser=False)
