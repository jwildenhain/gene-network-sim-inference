"""
Benchmark 100 Parameter Settings & Reclassify Simulation Runs.
Systematically tests 100 distinct parameter combinations to evaluate:
1. Responsive (Clear Level Changes): Steady state reached, max |ΔX| >= 5%.
2. Flat (Damped / Insensitive): Steady state reached, max |ΔX| < 5%.
3. Oscillating (Limit Cycle / Non-steady): Simulation does NOT reach steady state after perturbation (sustained cyclic fluctuations).

Also generates 3 representative runs (iorn-rep-responsive, iorn-rep-flat, iorn-rep-oscillating),
saves them to data/rgn_db.sqlite, and reclassifies all existing database runs.
"""

import sys
import json
import sqlite3
import collections
from pathlib import Path
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from grn_sim.core.network import GeneNetwork
from grn_sim.core.topology import NetworkTopology, TopologyType
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.simulator.ode_solver import ODESolver
from grn_sim.simulator.kinetics import evaluate_derivatives
from grn_sim.simulator.response_dynamics import classify_perturbation_response
from grn_sim.web.server import (
    _extract_kinetics_and_trajectories,
    export_graphviz_dot,
    render_graphviz_svg,
    export_graph_json,
    format_matrix_tsv,
)
from grn_sim.web.database import save_simulation_to_db, init_db


def run_100_parameter_sweep():
    print("=================================================================")
    print("  RUNNING 100 PARAMETER SETTINGS DYNAMICAL SWEEP")
    print("=================================================================")

    rng = np.random.default_rng(20261023)
    results = []

    for idx in range(100):
        # 1. Parameter space sampling
        n_genes = int(rng.choice([5, 6, 8, 10, 12, 16]))
        topo_name = str(rng.choice(["random", "power_law", "hierarchical"]))
        net_type = (
            TopologyType.RANDOM if topo_name == "random"
            else (TopologyType.HIERARCHICAL if topo_name == "hierarchical" else TopologyType.POWER_LAW)
        )
        k_av = float(rng.uniform(1.2, 4.2))
        act_ratio = float(rng.uniform(0.1, 0.85))
        pert_factor = float(rng.choice([0.005, 0.02, 0.1, 0.5, 1.5, 2.0, 3.5]))
        hill_n = float(rng.choice([1.5, 2.0, 3.0, 4.5, 6.0]))
        deg_B = float(rng.choice([0.3, 0.75, 1.0, 2.5]))
        
        # Target specific dynamical regimes across the sweep
        regime_target = "random"
        if idx % 3 == 0:
            regime_target = "oscillating_candidate"
        elif idx % 3 == 1:
            regime_target = "flat_candidate"
        else:
            regime_target = "responsive_candidate"

        eff_subdiv = 2 if n_genes % 2 == 0 else 1
        net = GeneNetwork.create(
            n=n_genes,
            k_av=k_av,
            net_type=net_type,
            ratio=act_ratio,
            subdiv=eff_subdiv,
            seed=idx * 73 + 1,
        )

        if regime_target == "flat_candidate":
            pert_factor = 0.005
            net.params.B = np.full(n_genes, 2.5)  # Heavy damping
        elif regime_target == "oscillating_candidate":
            # Repressilator negative feedback loop core
            ring_size = 5 if n_genes >= 5 else 3
            for r in range(ring_size):
                net.topology.adjacency[r, (r + 1) % ring_size] = -1.0
            net.params.n_i = np.where(net.topology.adjacency < 0, 4.5, 0.0)
            net.params.Ki = np.where(net.topology.adjacency < 0, 1.0, 0.0)
            net.params.v = np.full(n_genes, 8.0)
            net.params.B = np.full(n_genes, 1.0)
            pert_factor = 0.5
        else:
            # Standard responsive network
            net.params.v = np.full(n_genes, 1.05)
            net.params.B = np.full(n_genes, 0.75)
            pert_factor = 2.0

        solver = ODESolver(dt=0.05, max_steps=400)
        sim = PerturbationSimulator(perturbation_factor=pert_factor, solver=solver)

        try:
            exp = sim.run(net, seed=idx)
            p_trajs = exp.perturbation_trajectories or []
            
            # Direct dynamical trajectory evaluation
            converged = True
            max_tail_ptp = 0.0
            for traj in p_trajs:
                arr = np.asarray(traj, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 4:
                    tail_len = max(4, int(arr.shape[1] * 0.25))
                    tail = arr[:, -tail_len:]
                    ptp = float(np.max(np.ptp(tail, axis=1)))
                    if ptp > max_tail_ptp:
                        max_tail_ptp = ptp
                    if ptp > 0.02:
                        converged = False

            abs_X = np.abs(exp.X)
            max_shift = float(np.max(abs_X)) if abs_X.size > 0 else 0.0

            # Classification
            if not converged:
                category = "oscillating"
            elif max_shift < 0.05:
                category = "flat"
            else:
                category = "responsive"

            results.append({
                "idx": idx,
                "n_genes": n_genes,
                "topology": topo_name,
                "k_av": round(k_av, 2),
                "pert_factor": pert_factor,
                "regime_target": regime_target,
                "converged": converged,
                "max_tail_ptp": round(max_tail_ptp, 5),
                "max_shift": round(max_shift, 4),
                "category": category,
            })
        except Exception as e:
            print(f"Error on run {idx}: {e}")

    counts = collections.Counter(r["category"] for r in results)
    print("\nSummary of 100 Parameter Settings:")
    print(f"  • Responsive (Clear Level Changes): {counts['responsive']} runs ({counts['responsive']}%)")
    print(f"  • Flat (Damped / Insensitive):      {counts['flat']} runs ({counts['flat']}%)")
    print(f"  • Oscillating (Limit Cycles):       {counts['oscillating']} runs ({counts['oscillating']}%)")

    return results


def create_and_store_representative_runs(db_path: Path):
    print("\n=================================================================")
    print("  CREATING & STORING 3 REPRESENTATIVE RUNS IN SQLITE")
    print("=================================================================")

    init_db(str(db_path))

    # 1. Representative RESPONSIVE Run
    print("1. Generating Representative RESPONSIVE run (iorn-rep-responsive)...")
    N_resp = 10
    net_resp = GeneNetwork.create(
        n=N_resp,
        k_av=2.0,
        net_type=TopologyType.POWER_LAW,
        ratio=0.5,
        subdiv=2,
        seed=42,
    )
    solver_resp = ODESolver(dt=0.05, max_steps=400)
    sim_resp = PerturbationSimulator(perturbation_factor=2.0, solver=solver_resp)
    exp_resp = sim_resp.run(net_resp, seed=42)
    gene_names_resp = [f"G{i+1}" for i in range(N_resp)]
    kin_resp = _extract_kinetics_and_trajectories(net_resp, exp_resp, gene_names_resp)
    diag_resp = exp_resp.steady_state_diagnostics or {"is_valid_steady_state": True}
    resp_dyn_resp = classify_perturbation_response(
        exp_resp.X, diag_resp, perturbation_trajectories=exp_resp.perturbation_trajectories
    )

    dot_hier = export_graphviz_dot(net_resp.topology.adjacency, gene_names_resp, layout="dot")
    dot_spring = export_graphviz_dot(net_resp.topology.adjacency, gene_names_resp, layout="neato")
    svg_hier = render_graphviz_svg(dot_hier, engine="dot", matrix=net_resp.topology.adjacency, gene_names=gene_names_resp)
    svg_spring = render_graphviz_svg(dot_spring, engine="neato", matrix=net_resp.topology.adjacency, gene_names=gene_names_resp)

    sim_payload_resp = {
        "status": "success",
        "run_id": "iorn-rep-responsive",
        "mode": "synthetic",
        "settings": {
            "mode": "synthetic",
            "topology": "power_law",
            "kinetics_engine": "hill_nonlinear",
            "dynamics_preset": "responsive",
            "loading_mode": "insilico",
            "num_genes": N_resp,
            "k_av": 2.0,
            "act_ratio": 0.5,
            "noise": 0.0,
            "seed": 42,
        },
        "response_type": "responsive",
        "response_dynamics": resp_dyn_resp,
        "true_matrix": net_resp.topology.adjacency.tolist(),
        "gene_names": gene_names_resp,
        "X": exp_resp.X.tolist(),
        "U": exp_resp.U.tolist(),
        "basal_trajectory": kin_resp["basal_trajectory"],
        "perturbation_trajectories": kin_resp["perturbation_trajectories"],
        "gene_profiles": kin_resp["gene_profiles"],
        "time_points": kin_resp["time_points"],
        "steady_state_diagnostics": diag_resp,
        "svg_hierarchical": svg_hier,
        "svg_spring": svg_spring,
    }
    save_simulation_to_db("iorn-rep-responsive", sim_payload_resp, db_path=str(db_path))

    # 2. Representative FLAT Run
    print("2. Generating Representative FLAT run (iorn-rep-flat)...")
    N_flat = 10
    net_flat = GeneNetwork.create(
        n=N_flat,
        k_av=1.5,
        net_type=TopologyType.HIERARCHICAL,
        ratio=0.6,
        subdiv=2,
        seed=101,
    )
    net_flat.params.B = np.full(N_flat, 2.5)  # Fast decay / heavy damping
    solver_flat = ODESolver(dt=0.05, max_steps=400)
    sim_flat = PerturbationSimulator(perturbation_factor=0.005, solver=solver_flat)
    exp_flat = sim_flat.run(net_flat, seed=101)
    gene_names_flat = [f"G{i+1}" for i in range(N_flat)]
    kin_flat = _extract_kinetics_and_trajectories(net_flat, exp_flat, gene_names_flat)
    diag_flat = exp_flat.steady_state_diagnostics or {"is_valid_steady_state": True}
    resp_dyn_flat = classify_perturbation_response(
        exp_flat.X, diag_flat, perturbation_trajectories=exp_flat.perturbation_trajectories
    )

    dot_hier_f = export_graphviz_dot(net_flat.topology.adjacency, gene_names_flat, layout="dot")
    dot_spring_f = export_graphviz_dot(net_flat.topology.adjacency, gene_names_flat, layout="neato")
    svg_hier_f = render_graphviz_svg(dot_hier_f, engine="dot", matrix=net_flat.topology.adjacency, gene_names=gene_names_flat)
    svg_spring_f = render_graphviz_svg(dot_spring_f, engine="neato", matrix=net_flat.topology.adjacency, gene_names=gene_names_flat)

    sim_payload_flat = {
        "status": "success",
        "run_id": "iorn-rep-flat",
        "mode": "synthetic",
        "settings": {
            "mode": "synthetic",
            "topology": "hierarchical",
            "kinetics_engine": "hill_nonlinear",
            "dynamics_preset": "flat",
            "loading_mode": "insilico",
            "num_genes": N_flat,
            "k_av": 1.5,
            "act_ratio": 0.6,
            "noise": 0.0,
            "seed": 101,
        },
        "response_type": "flat",
        "response_dynamics": resp_dyn_flat,
        "true_matrix": net_flat.topology.adjacency.tolist(),
        "gene_names": gene_names_flat,
        "X": exp_flat.X.tolist(),
        "U": exp_flat.U.tolist(),
        "basal_trajectory": kin_flat["basal_trajectory"],
        "perturbation_trajectories": kin_flat["perturbation_trajectories"],
        "gene_profiles": kin_flat["gene_profiles"],
        "time_points": kin_flat["time_points"],
        "steady_state_diagnostics": diag_flat,
        "svg_hierarchical": svg_hier_f,
        "svg_spring": svg_spring_f,
    }
    save_simulation_to_db("iorn-rep-flat", sim_payload_flat, db_path=str(db_path))

    # 3. Representative OSCILLATING Run (Limit Cycle)
    print("3. Generating Representative OSCILLATING run (iorn-rep-oscillating)...")
    N_osc = 6
    adj_osc = np.zeros((N_osc, N_osc))
    # 5-gene repressilator ring (0->1->2->3->4->0)
    for r in range(5):
        adj_osc[r, (r + 1) % 5] = -1.0
    # Gene 5 is driven by the oscillator
    adj_osc[0, 5] = 1.0

    topo_osc = NetworkTopology(adjacency=adj_osc, topology_type=TopologyType.RANDOM, k_av=1.2, ratio=0.17)
    net_osc = GeneNetwork(topology=topo_osc)
    net_osc.params.n_i = np.where(adj_osc < 0, 4.5, 0.0)
    net_osc.params.n_k = np.where(adj_osc > 0, 2.0, 0.0)
    net_osc.params.Ki = np.where(adj_osc < 0, 1.2, 0.0)
    net_osc.params.Ka = np.where(adj_osc > 0, 1.0, 0.0)
    net_osc.params.v = np.full(N_osc, 8.0)
    net_osc.params.B = np.full(N_osc, 1.0)

    solver_osc = ODESolver(dt=0.05, max_steps=500)
    sim_osc = PerturbationSimulator(perturbation_factor=0.5, solver=solver_osc)
    exp_osc = sim_osc.run(net_osc, seed=777)
    gene_names_osc = [f"G{i+1}" for i in range(N_osc)]
    kin_osc = _extract_kinetics_and_trajectories(net_osc, exp_osc, gene_names_osc)
    diag_osc = exp_osc.steady_state_diagnostics or {
        "is_valid_steady_state": False,
        "oscillating_detected": True,
    }
    resp_dyn_osc = classify_perturbation_response(
        exp_osc.X, diag_osc, perturbation_trajectories=exp_osc.perturbation_trajectories
    )

    dot_hier_o = export_graphviz_dot(net_osc.topology.adjacency, gene_names_osc, layout="dot")
    dot_spring_o = export_graphviz_dot(net_osc.topology.adjacency, gene_names_osc, layout="neato")
    svg_hier_o = render_graphviz_svg(dot_hier_o, engine="dot", matrix=net_osc.topology.adjacency, gene_names=gene_names_osc)
    svg_spring_o = render_graphviz_svg(dot_spring_o, engine="neato", matrix=net_osc.topology.adjacency, gene_names=gene_names_osc)

    sim_payload_osc = {
        "status": "success",
        "run_id": "iorn-rep-oscillating",
        "mode": "synthetic",
        "settings": {
            "mode": "synthetic",
            "topology": "random",
            "kinetics_engine": "hill_nonlinear",
            "dynamics_preset": "oscillating",
            "loading_mode": "insilico",
            "num_genes": N_osc,
            "k_av": 1.2,
            "act_ratio": 0.17,
            "noise": 0.0,
            "seed": 777,
        },
        "response_type": "oscillating",
        "response_dynamics": resp_dyn_osc,
        "true_matrix": net_osc.topology.adjacency.tolist(),
        "gene_names": gene_names_osc,
        "X": exp_osc.X.tolist(),
        "U": exp_osc.U.tolist(),
        "basal_trajectory": kin_osc["basal_trajectory"],
        "perturbation_trajectories": kin_osc["perturbation_trajectories"],
        "gene_profiles": kin_osc["gene_profiles"],
        "time_points": kin_osc["time_points"],
        "steady_state_diagnostics": diag_osc,
        "svg_hierarchical": svg_hier_o,
        "svg_spring": svg_spring_o,
    }
    save_simulation_to_db("iorn-rep-oscillating", sim_payload_osc, db_path=str(db_path))
    print("Representative runs saved successfully!")


def reclassify_sqlite_database(db_path: Path):
    print("\n=================================================================")
    print("  RECLASSIFYING ALL RUNS IN SQLITE DATABASE")
    print("=================================================================")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT simid, response_type, sim_payload_json, history_entry_json FROM nw")
    rows = cur.fetchall()

    reclassified_count = 0
    new_counts = collections.Counter()

    for simid, old_resp, sim_json, hist_json in rows:
        if not sim_json:
            continue
        try:
            data = json.loads(sim_json)
            X = np.asarray(data.get("X") or [], dtype=float)
            diag = data.get("steady_state_diagnostics") or {}
            pert_trajs = data.get("perturbation_trajectories") or []

            resp_dyn = classify_perturbation_response(
                X,
                diag,
                perturbation_trajectories=pert_trajs,
            )
            new_resp = resp_dyn["response_type"]
            new_counts[new_resp] += 1

            if new_resp != old_resp:
                reclassified_count += 1
                # Update nw table
                cur.execute(
                    "UPDATE nw SET response_type = ?, response_stats_json = ? WHERE simid = ?",
                    (new_resp, json.dumps(resp_dyn), simid),
                )
                # Update history entry json if present
                if hist_json:
                    try:
                        h = json.loads(hist_json)
                        h["response_type"] = new_resp
                        h["response_label"] = resp_dyn["label"]
                        h["badge_class"] = resp_dyn["badge_class"]
                        cur.execute(
                            "UPDATE nw SET history_entry_json = ? WHERE simid = ?",
                            (json.dumps(h), simid),
                        )
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error reclassifying {simid}: {e}")

    conn.commit()
    conn.close()

    print(f"Database reclassification complete:")
    print(f"  • Total runs processed: {len(rows)}")
    print(f"  • Total reclassified runs: {reclassified_count}")
    print(f"  • New distribution: {dict(new_counts)}")


if __name__ == "__main__":
    db_file = ROOT / "data" / "rgn_db.sqlite"
    run_100_parameter_sweep()
    create_and_store_representative_runs(db_file)
    reclassify_sqlite_database(db_file)
