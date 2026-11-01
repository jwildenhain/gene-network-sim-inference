"""
Command-line interface (CLI) for GRN-Sim.
"""

import argparse
import json

from grn_sim.benchmarks.sos_pathway import run_sos_benchmark
from grn_sim.agents.multi_agent_system import CoordinatorAgent


def format_table(headers, rows):
    """Formats ASCII table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            widths[idx] = max(widths[idx], len(str(val)))

    sep = "+-" + "-+-".join(["-" * w for w in widths]) + "-+"
    lines = [sep]
    header_line = "| " + " | ".join([f"{h:<{widths[i]}}" for i, h in enumerate(headers)]) + " |"
    lines.append(header_line)
    lines.append(sep)
    for row in rows:
        row_line = "| " + " | ".join([f"{str(val):<{widths[i]}}" for i, val in enumerate(row)]) + " |"
        lines.append(row_line)
    lines.append(sep)
    return "\n".join(lines)


def cmd_sos_benchmark(args):
    print("================================================================================")
    print("REPLICATING E. COLI SOS PATHWAY BENCHMARK (Gardner et al. 2003 / Thesis Table 4.1)")
    print("================================================================================")
    metrics_by_algo = run_sos_benchmark(by_chance_runs=100)

    headers = [
        "Statistic", "s2g", "g2s", "fast", "NIR", "Voting", "By Chance"
    ]
    stat_rows = [
        ("Coverage %", lambda m: f"{m.coverage_pct:.2f}%"),
        ("False positives %", lambda m: f"{m.false_positives_pct:.2f}%"),
        ("Inferred connections %", lambda m: f"{m.inferred_connections_pct:.2f}%"),
        ("FP per inferred conn %", lambda m: f"{m.fp_per_inferred_pct:.2f}%"),
        ("Inferred sign %", lambda m: f"{m.inferred_sign_pct:.2f}%"),
        ("Total number of errors", lambda m: f"{m.total_errors}"),
        ("F1-Score", lambda m: f"{m.f1_score:.4f}"),
        ("Precision", lambda m: f"{m.precision:.4f}"),
        ("Recall", lambda m: f"{m.recall:.4f}"),
    ]

    table_data = []
    for label, fn in stat_rows:
        row = [label]
        for algo_key in ["s2g", "g2s", "fast", "NIR", "voting", "by_chance"]:
            m = metrics_by_algo[algo_key]
            row.append(fn(m))
        table_data.append(row)

    print(format_table(headers, table_data))


def cmd_multi_agent(args):
    print("================================================================================")
    print("AUTONOMOUS MULTI-AGENT GRN SIMULATION & INFERENCE STUDY")
    print("================================================================================")
    coordinator = CoordinatorAgent()
    print(f"Launching study: n={args.genes}, topology={args.topology}, k_av={args.k_av}, noise={args.noise}")
    result = coordinator.run_study(
        n=args.genes,
        k_av=args.k_av,
        net_type=args.topology,
        noise=args.noise,
        seed=args.seed,
    )

    print("\n--- Network Characteristics ---")
    print(json.dumps(result["network_parameters"], indent=2))

    print("\n--- Multi-Agent Benchmark Evaluation ---")
    headers = ["Algorithm", "Coverage %", "FP %", "Inferred %", "FP/Inferred %", "Total Errors", "F1"]
    rows = []
    for algo, m in result["metrics"].items():
        rows.append([
            algo,
            f"{m['Coverage %']:.1f}%",
            f"{m['False Positives %']:.1f}%",
            f"{m['Inferred Connections %']:.1f}%",
            f"{m['FP per Inferred %']:.1f}%",
            m['Total Errors'],
            f"{m['F1-Score']:.4f}",
        ])
    print(format_table(headers, rows))


def cmd_perturbseq(args):
    """Simulates a Perturb-seq screen on a preset network, writes an .h5ad and prints the realism report."""
    from grn_sim.perturbseq.calibration import PRESETS
    from grn_sim.perturbseq.pipeline import build_network, run_screen
    from grn_sim.perturbseq.realism import format_report

    print(f"Building a {args.genes}-gene '{args.preset}' network (seed {args.seed}) ...")
    net = build_network(args.preset, args.genes, seed=args.seed)
    run = run_screen(
        net, mode=args.mode, pairs=args.pairs, cells_per_condition=args.cells, n_control_cells=args.controls,
        cell_model=args.noise, measurement=not args.no_measurement, seed=args.seed,
        guide_efficiency_mean=PRESETS[args.preset].guide_efficiency_mean,
    )
    print(f"Simulated {run.adata.n_obs} cells x {run.adata.n_vars} genes  {run.seconds}")
    if args.out:
        run.adata.write_h5ad(args.out, compression="gzip")
        print(f"Wrote {args.out}")
    if not args.no_report:
        result = run.realism()
        print("\n--- Realism vs genome-scale Perturb-seq targets (score 0 = all inside) ---")
        print(format_report(result["report"]))
        print(f"score {result['score']:.2f}")
        if args.report:
            with open(args.report, "w") as fh:
                json.dump(result, fh, indent=2, default=float)
            print(f"Wrote {args.report}")


def cmd_web(args):
    from grn_sim.web.server import run_server
    run_server(port=args.port, open_browser=not args.no_browser)


def main():
    parser = argparse.ArgumentParser(
        description="GRN-Sim: Gene Network Perturbation Simulator & Inference Suite"
    )
    subparsers = parser.add_subparsers(dest="command")

    # sos-benchmark
    subparsers.add_parser("sos-benchmark", help="Run 9-gene SOS pathway benchmark replication")

    # multi-agent
    p_ma = subparsers.add_parser("multi-agent", help="Run autonomous multi-agent simulation and inference loop")
    p_ma.add_argument("--genes", "-n", type=int, default=12, help="Number of genes")
    p_ma.add_argument("--topology", "-t", type=str, default="power_law", choices=["random", "power_law", "hierarchical"])
    p_ma.add_argument("--k-av", "-k", type=float, default=1.5, help="Average degree")
    p_ma.add_argument("--noise", type=float, default=0.0, help="Measurement noise variance")
    p_ma.add_argument("--seed", type=int, default=42, help="Random seed")

    # perturbseq
    p_ps = subparsers.add_parser("perturbseq", help="Simulate a Perturb-seq screen (h5ad + realism report)")
    p_ps.add_argument("--genes", "-n", type=int, default=200, help="Number of genes (default 200)")
    p_ps.add_argument("--preset", default="replogle_like", choices=["replogle_like", "thesis"],
                      help="Network regime (default replogle_like)")
    p_ps.add_argument("--mode", default="knockdown", choices=["knockdown", "knockout", "input"])
    p_ps.add_argument("--pairs", type=int, default=0, help="Random double perturbations")
    p_ps.add_argument("--cells", type=int, default=100, help="Cells per perturbation")
    p_ps.add_argument("--controls", type=int, default=None, help="Non-targeting cells (default: --cells)")
    p_ps.add_argument("--noise", default="lna", choices=["lna", "sde", "legacy"], help="Cell noise model")
    p_ps.add_argument("--no-measurement", action="store_true", help="Skip doublets/ambient/batches/guide calling")
    p_ps.add_argument("--seed", type=int, default=1)
    p_ps.add_argument("--out", "-o", default=None, help="Write the AnnData to this .h5ad file")
    p_ps.add_argument("--report", default=None, help="Write the realism report to this JSON file")
    p_ps.add_argument("--no-report", action="store_true", help="Skip the realism statistics")

    # web
    p_web = subparsers.add_parser("web", help="Launch interactive phpIoRN-equivalent web simulation & inference engine")
    p_web.add_argument("--port", "-p", type=int, default=8080, help="Port to bind server (default: 8080)")
    p_web.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")

    args = parser.parse_args()

    if args.command == "sos-benchmark":
        cmd_sos_benchmark(args)
    elif args.command == "multi-agent":
        cmd_multi_agent(args)
    elif args.command == "perturbseq":
        cmd_perturbseq(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        # Default behavior: run sos benchmark and multi-agent demonstration
        cmd_sos_benchmark(args)
        print("\n")
        parser.parse_args(["multi-agent"])
        args.genes = 12
        args.topology = "power_law"
        args.k_av = 1.5
        args.noise = 0.0
        args.seed = 42
        cmd_multi_agent(args)


if __name__ == "__main__":
    main()

