"""
Interactive Web Application Server for GRN-Sim.
Fully replaces phpIoRN (runsim.php, global_conf.inc.php, get_results.php, springgraph.pl)
with an asynchronous, real-time simulation and network inference engine.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import json
import os
import sys
import urllib.parse
from typing import Dict, Any, Optional
import numpy as np

from grn_sim.core.topology import generate_topology, TopologyType
from grn_sim.core.network import GeneNetwork
from grn_sim.simulator.experiment import PerturbationSimulator
from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.inference.by_chance import ByChance
from grn_sim.evaluation.metrics import evaluate_inference
from grn_sim.benchmarks.sos_pathway import run_sos_benchmark
from grn_sim.visualization.graph_export import (
    export_graph_json,
    export_comparison_graph_json,
    export_graphviz_dot,
    compute_detailed_confusion_matrix,
)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server for handling concurrent simulation requests."""
    daemon_threads = True


# In-memory session state for active simulation
CURRENT_STATE: Dict[str, Any] = {
    "network": None,
    "experiment": None,
    "true_matrix": None,
    "gene_names": None,
    "inferred_results": {},
}


class GRNWebHandler(BaseHTTPRequestHandler):
    """REST API and static file request handler."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

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
                "num_genes": len(CURRENT_STATE["gene_names"]) if has_sim else 0,
            }
            self._set_headers(200)
            self.wfile.write(json.dumps(resp).encode("utf-8"))
            return

        elif path == "/api/benchmark/sos":
            # Replicate SOS benchmark
            try:
                sos_res = run_sos_benchmark()
                # Format for web
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

        else:
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
                num_genes = int(params.get("num_genes", 12))
                topo_str = params.get("topology", "power_law")
                if topo_str == "random":
                    net_type = TopologyType.RANDOM
                elif topo_str == "hierarchical":
                    net_type = TopologyType.HIERARCHICAL
                else:
                    net_type = TopologyType.POWER_LAW

                k_av = float(params.get("k_av", 1.8))
                act_ratio = float(params.get("act_ratio", 0.5))
                noise_level = float(params.get("noise", 0.05))
                seed = params.get("seed", 42)
                if seed is not None:
                    seed = int(seed)

                # Generate network and run perturbation simulation
                subdiv = 2 if num_genes % 2 == 0 else 1
                network = GeneNetwork.create(
                    n=num_genes,
                    k_av=k_av,
                    net_type=net_type,
                    ratio=act_ratio,
                    subdiv=subdiv,
                    seed=seed,
                )
                simulator = PerturbationSimulator(
                    perturbation_factor=2.0,
                    measurement_noise=noise_level,
                )
                experiment = simulator.run(network, seed=seed)

                adj = network.topology.adjacency
                gene_names = [f"G{i+1}" for i in range(num_genes)]

                CURRENT_STATE["network"] = network
                CURRENT_STATE["experiment"] = experiment
                CURRENT_STATE["true_matrix"] = adj
                CURRENT_STATE["gene_names"] = gene_names
                CURRENT_STATE["inferred_results"] = {}

                # Format graph JSON
                graph_json = export_graph_json(adj, gene_names)
                dot_str = export_graphviz_dot(adj, gene_names, graph_name=f"GRN_{topo_str}")

                resp = {
                    "status": "success",
                    "num_genes": num_genes,
                    "topology": topo_str,
                    "k_average": round(float(np.sum(adj != 0) / num_genes), 2),
                    "total_edges": int(np.sum(adj != 0)),
                    "activations": int(np.sum(adj > 0)),
                    "inhibitions": int(np.sum(adj < 0)),
                    "graph": graph_json,
                    "dot": dot_str,
                    "expression_ratios_shape": list(experiment.X.shape),
                }
                self._set_headers(200)
                self.wfile.write(json.dumps(resp).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
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

                inferred_dict = {}
                metrics_dict = {}
                cmat_dict = {}
                comp_graphs = {}
                dots_dict = {}

                for alg in selected_algs:
                    if alg in alg_runners:
                        res = alg_runners[alg].fit(X, U)
                        pred = res.adjacency_pred
                        inferred_dict[alg] = pred
                        m = evaluate_inference(true_matrix, pred)
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
                        }
                        cmat_dict[alg] = compute_detailed_confusion_matrix(true_matrix, pred)
                        comp_graphs[alg] = export_comparison_graph_json(true_matrix, pred, gene_names)
                        dots_dict[alg] = export_graphviz_dot(pred, gene_names, graph_name=f"Inferred_{alg.upper()}")

                primary_alg = selected_algs[0] if selected_algs else "s2g"

                resp = {
                    "status": "success",
                    "primary_algorithm": primary_alg,
                    "metrics": metrics_dict,
                    "confusion_matrix": cmat_dict,
                    "comparison_graph": comp_graphs.get(primary_alg, {}),
                    "comparison_graphs": comp_graphs,
                    "dot": dots_dict.get(primary_alg, ""),
                    "dots": dots_dict,
                }
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


def run_server(port: int = 8080, open_browser: bool = True):
    """Starts the GRN-Sim interactive web server."""
    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, GRNWebHandler)
    url = f"http://localhost:{port}"
    print(f"============================================================")
    print(f"  GRN-Sim Web Engine (Modernized phpIoRN Simulator)")
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_server(port=port)
