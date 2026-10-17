#!/usr/bin/env python3
"""
GRN-Sim Prototype Demonstration Script.
Replicates the Bachelor Honours Thesis experiments:
1. E. coli SOS Pathway Benchmark (Table 4.1)
2. In Silico Scale-Free Network Simulation & Multi-Algorithm Inference
3. Autonomous Multi-Agent Simulation & Inference Workflow
"""

import sys
from pathlib import Path

# Add src to pythonpath
src_dir = str(Path(__file__).resolve().parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from grn_sim.cli import cmd_sos_benchmark, cmd_multi_agent
from grn_sim.benchmarks.synthetic_sweep import run_topology_comparison, run_noise_sweep
from grn_sim.agents.multi_agent_system import CoordinatorAgent


def main():
    print("\n" + "=" * 80)
    print("      GENE REGULATORY NETWORK PERTURBATION SIMULATOR & INFERENCE SUITE")
    print("         Reverse-Engineered from Bachelor Honours Thesis (Jan Wildenhain)")
    print("=" * 80 + "\n")

    # 1. Biological Benchmark
    class Args:
        pass

    args = Args()
    cmd_sos_benchmark(args)

    print("\n" + "=" * 80)
    print("            AUTONOMOUS MULTI-AGENT SIMULATION & INFERENCE STUDY")
    print("=" * 80 + "\n")

    args.genes = 12
    args.topology = "power_law"
    args.k_av = 1.5
    args.noise = 0.02
    args.seed = 42
    cmd_multi_agent(args)

    print("\n" + "=" * 80)
    print("            SYNTHETIC NOISE SENSITIVITY SWEEP (Section 4.3)")
    print("=" * 80 + "\n")
    print("Testing recovery under increasing measurement noise (0%, 2%, 5%, 10%)...")
    noise_results = run_noise_sweep(network_size=10, noise_levels=[0.0, 0.02, 0.05, 0.10], repeats=2)
    print("\nMean Coverage % across noise levels [0.0, 0.02, 0.05, 0.10]:")
    for algo, covs in noise_results.items():
        print(f"  {algo:<15}: {covs}")

    print("\n[SUCCESS] End-to-end prototype demonstration completed successfully.")


if __name__ == "__main__":
    main()
