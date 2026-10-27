"""
generate_balanced_dynamics_corpus.py
Creates and balances the simulation archive to ~30 datasets per dynamics category:
  - 30 Flat (Insensitive / Damped, max amplitude < 5%)
  - 30 Oscillating (Persistent Limit Cycle, tail variance > 0.02)
  - 30 Responsive (Clear steady-state level changes, max amplitude >= 5%)

Executes full reverse-engineering inference across all algorithms and writes to SQLite rgn_db.sqlite.
"""

import sqlite3
import numpy as np

# Needs the package: `pip install -e .` from the repository root.
from grn_sim.paths import archive_db_path

from grn_sim.core.topology import TopologyType
from grn_sim.simulator.kinetics import GeneNetwork
from grn_sim.web.server import (
    run_simulation_with_engine,
    classify_perturbation_response,
    export_graphviz_dot,
    render_graphviz_svg,
    export_graph_json,
    format_matrix_tsv,
    _extract_kinetics_and_trajectories,
    _execute_inference,
    CURRENT_STATE,
    SIMULATION_RUN_STORE,
)

ALGORITHMS = ["s2g", "g2s", "vote", "nir", "fast", "by_chance", "ridge_cv", "genie3", "netdeconv", "cpa_lite", "atlasshift"]

def create_and_save_run(network, experiment, dynamics_preset, mode_name, topo_str, noise, seed, run_id_prefix):
    num_genes = network.n
    gene_names = [f"G{i+1}" for i in range(num_genes)]
    adj = network.topology.adjacency

    kin = _extract_kinetics_and_trajectories(network, experiment, gene_names)
    ss_diag = experiment.steady_state_diagnostics or {
        "is_valid_steady_state": bool(experiment.is_valid_steady_state),
        "oscillating_detected": not bool(experiment.is_valid_steady_state),
    }
    resp_dyn = classify_perturbation_response(
        experiment.X,
        ss_diag,
        perturbation_trajectories=experiment.perturbation_trajectories,
    )

    dot_hier = export_graphviz_dot(adj, gene_names, graph_name=f"GRN_{topo_str}_dot", layout="dot")
    dot_spring = export_graphviz_dot(adj, gene_names, graph_name=f"GRN_{topo_str}_neato", layout="neato")
    svg_hier = render_graphviz_svg(dot_hier, engine="dot", matrix=adj, gene_names=gene_names)
    svg_spring = render_graphviz_svg(dot_spring, engine="neato", matrix=adj, gene_names=gene_names)
    graph_json = export_graph_json(adj, gene_names)
    msmt_text = format_matrix_tsv(experiment.X)
    pert_text = format_matrix_tsv(experiment.U)

    # Deterministic clean run_id
    run_id = f"iorn-{run_id_prefix}-{seed:04d}"

    input_settings = {
        "mode": "insilico",
        "topology": topo_str,
        "kinetics_engine": "biophysical",
        "dynamics_preset": dynamics_preset,
        "loading_mode": "insilico",
        "is_empirical_direct": False,
        "is_biophysical_fit": False,
        "dataset_source": None,
        "response_type": resp_dyn["response_type"],
        "num_genes": num_genes,
        "k_av": round(float(np.sum(adj != 0) / num_genes), 2),
        "act_ratio": round(float(np.sum(adj > 0) / max(1, np.sum(adj != 0))), 2),
        "noise": round(float(noise), 3),
        "bio_var": 0.01,
        "seed": seed,
    }

    CURRENT_STATE.clear()
    CURRENT_STATE.update({
        "run_id": run_id,
        "mode": "insilico",
        "seed": seed,
        "settings": input_settings,
        "dynamics_preset": dynamics_preset,
        "loading_mode": "insilico",
        "is_empirical_direct": False,
        "is_biophysical_fit": False,
        "dataset_source": None,
        "provenance": {},
        "fitted_parameters": {},
        "response_type": resp_dyn["response_type"],
        "response_dynamics": resp_dyn,
        "network": network,
        "experiment": experiment,
        "true_matrix": adj,
        "gene_names": gene_names,
        "topology": topo_str,
        "k_av": input_settings["k_av"],
        "act_ratio": input_settings["act_ratio"],
        "noise": noise,
        "bio_var": 0.01,
        "node_params": kin["node_params"],
        "edge_params": kin["edge_params"],
        "edge_list": kin["edge_list"],
        "basal_trajectory": kin["basal_trajectory"],
        "perturbation_trajectories": kin["perturbation_trajectories"],
        "gene_profiles": kin.get("gene_profiles", {}),
        "time_points": kin.get("time_points", []),
        "steady_state_diagnostics": ss_diag,
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
        "mode": "insilico",
        "settings": input_settings,
        "dynamics_preset": dynamics_preset,
        "loading_mode": "insilico",
        "is_empirical_direct": False,
        "is_biophysical_fit": False,
        "dataset_source": None,
        "provenance": {},
        "fitted_parameters": {},
        "response_type": resp_dyn["response_type"],
        "response_dynamics": resp_dyn,
        "num_genes": num_genes,
        "gene_names": gene_names,
        "topology": topo_str,
        "k_average": input_settings["k_av"],
        "act_ratio": input_settings["act_ratio"],
        "noise": noise,
        "bio_var": 0.01,
        "seed": seed,
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
        "gene_profiles": kin.get("gene_profiles", {}),
        "time_points": kin.get("time_points", []),
        "steady_state_diagnostics": ss_diag,
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

    # Execute all algorithms and save to database
    _execute_inference(ALGORITHMS)
    return resp_dyn["response_type"]

def generate_flat_runs(target_count=30):
    print(f"=== Generating Flat Datasets (Target: {target_count}) ===")
    conn = sqlite3.connect(str(archive_db_path()))
    c = conn.cursor()
    c.execute("SELECT count(*) FROM nw WHERE response_type = 'flat' AND history_entry_json != '{}' AND history_entry_json != ''")
    current = c.fetchone()[0]
    conn.close()
    needed = max(0, target_count - current)
    print(f"Current flat in DB: {current}, generating {needed} more...")

    created = 0
    seed = 1000
    while created < needed:
        seed += 1
        n_genes = [8, 10, 12, 14][seed % 4]
        topo = [TopologyType.POWER_LAW, TopologyType.HIERARCHICAL, TopologyType.RANDOM][seed % 3]
        topo_str = ["power_law", "hierarchical", "random"][seed % 3]
        network = GeneNetwork.create(n=n_genes, k_av=1.5, net_type=topo, ratio=0.5, seed=seed)
        # Flat: minimal perturbation factor 0.003 - 0.005 so shifts are < 5%
        p_factor = 0.0035 + (seed % 5) * 0.0003
        noise = 0.001
        exp = run_simulation_with_engine(network, engine="biophysical", perturbation_factor=p_factor, measurement_noise=noise, seed=seed)
        r_type = create_and_save_run(network, exp, "flat", "insilico", topo_str, noise, seed, f"flt{n_genes}")
        if r_type == "flat":
            created += 1
            print(f"[{created}/{needed}] Created Flat run: seed {seed}, n={n_genes}, topo={topo_str}")

def generate_oscillating_runs(target_count=30):
    print(f"\n=== Generating Oscillating Datasets (Target: {target_count}) ===")
    conn = sqlite3.connect(str(archive_db_path()))
    c = conn.cursor()
    c.execute("SELECT count(*) FROM nw WHERE response_type = 'oscillating' AND history_entry_json != '{}' AND history_entry_json != ''")
    current = c.fetchone()[0]
    conn.close()
    needed = max(0, target_count - current)
    print(f"Current oscillating in DB: {current}, generating {needed} more...")

    created = 0
    seed = 2000
    while created < needed:
        seed += 1
        ring_len = 5 if (seed % 3 != 0) else 3
        extra_genes = (seed % 4) * 2  # 0, 2, 4, 6 extra genes
        n_genes = ring_len + extra_genes
        topo_str = "random" if extra_genes == 0 else ("power_law" if seed % 2 == 0 else "hierarchical")

        network = GeneNetwork.create(n=n_genes, k_av=1.8, net_type=TopologyType.RANDOM, ratio=0.0, seed=seed)
        network.topology.adjacency = np.zeros((n_genes, n_genes))
        for r in range(ring_len):
            network.topology.adjacency[r, (r + 1) % ring_len] = -1.0

        # If extra genes, connect them downstream of the ring
        if extra_genes > 0:
            for eg in range(ring_len, n_genes):
                source = eg % ring_len
                network.topology.adjacency[eg, source] = 1.0 if (eg % 2 == 0) else -1.0

        hill = 4.2 + (seed % 4) * 0.3
        network.params.n_i = np.where(network.topology.adjacency < 0, hill, 2.0)
        network.params.Ki = np.where(network.topology.adjacency < 0, 1.0, 1.0)
        network.params.v = np.full(n_genes, 10.0)
        network.params.B = np.full(n_genes, 1.0)

        exp = run_simulation_with_engine(network, engine="biophysical", perturbation_factor=1.5, measurement_noise=0.0, seed=seed)
        r_type = create_and_save_run(network, exp, "oscillating", "insilico", topo_str, 0.0, seed, f"osc{n_genes}")
        if r_type == "oscillating":
            created += 1
            print(f"[{created}/{needed}] Created Oscillating run: seed {seed}, n={n_genes}, ring={ring_len}")

def balance_responsive_runs(target_count=30):
    print(f"\n=== Balancing Responsive Datasets (Target: ~{target_count}) ===")
    conn = sqlite3.connect(str(archive_db_path()))
    c = conn.cursor()
    c.execute("SELECT count(*) FROM nw WHERE response_type = 'responsive' AND history_entry_json != '{}' AND history_entry_json != ''")
    current = c.fetchone()[0]
    print(f"Current valid responsive in DB: {current}")

    # If more than 35, prune down to target_count to achieve clean 30-30-30 balance
    if current > target_count:
        excess = current - target_count
        print(f"Pruning {excess} excess responsive runs to maintain ~{target_count} per category...")
        c.execute("""
            SELECT simid FROM nw
            WHERE response_type = 'responsive' AND history_entry_json != '{}' AND history_entry_json != ''
            ORDER BY rowid ASC
            LIMIT ?
        """, (excess,))
        prune_ids = [r[0] for r in c.fetchall()]
        for pid in prune_ids:
            c.execute("DELETE FROM nw WHERE simid = ?", (pid,))
            c.execute("DELETE FROM alg WHERE simid = ?", (pid,))
        conn.commit()
        print(f"Pruned {len(prune_ids)} runs successfully.")
    conn.close()

if __name__ == "__main__":
    generate_flat_runs(30)
    generate_oscillating_runs(30)
    balance_responsive_runs(30)

    # Print final counts
    conn = sqlite3.connect(str(archive_db_path()))
    c = conn.cursor()
    c.execute("SELECT response_type, count(*) FROM nw WHERE history_entry_json != '{}' AND history_entry_json != '' GROUP BY response_type")
    print("\n=== Final Archive Balance in SQLite rgn_db ===")
    for row in c.fetchall():
        print(f"  Category {row[0]:12s}: {row[1]} runs")
    conn.close()
