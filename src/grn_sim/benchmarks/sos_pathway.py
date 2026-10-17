"""
E. coli SOS Pathway Benchmark Replication.
Replicates the 9-gene biological network reconstruction benchmark from Gardner et al. (Science 2003)
and thesis Chapter 4 (Table 4.1).
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import scipy.io

from grn_sim.inference.s2g import SimpleToGeneral
from grn_sim.inference.g2s import GeneralToSpecific
from grn_sim.inference.nir import NIRAlgorithm
from grn_sim.inference.fast_svd import FastSVD
from grn_sim.inference.by_chance import ByChance
from grn_sim.inference.ensemble import VotingEnsemble
from grn_sim.evaluation.metrics import evaluate_inference, BenchmarkMetrics


GENE_NAMES = [
    "recA", "lexA", "ssb", "recF", "dinI", "umuDC", "rpoD", "rpoH", "rpoS"
]

# Ground truth signed adjacency matrix from thesis / SOS_statistic.m
SOS_GROUND_TRUTH = np.array([
    [ 1, -1, -1,  1,  1, -1,  1,  0,  0],
    [ 1, -1, -1,  1,  1, -1,  1,  0,  0],
    [ 1, -1, -1,  1,  1, -1,  1,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1],
    [ 1, -1, -1,  1,  1, -1,  1,  0,  0],
    [ 1, -1, -1,  1,  1, -1,  1,  0,  0],
    [ 1, -1, -1,  1,  1, -1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1],
], dtype=int)


def load_sos_dataset(data_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads results.r (expression ratios) and results.p (perturbation matrix).

    Returns:
        X: (9, 9) expression response matrix.
        U: (9, 9) perturbation matrix.
        sX: (9, 9) expression error matrix.
        sU: (9, 9) perturbation error matrix.
    """
    if data_path is None:
        # Default project path
        base_dir = Path(__file__).resolve().parent.parent.parent.parent / "data"
        mat_path = base_dir / "results_gapa_20021209_and_mmc_dbl.mat"
        if not mat_path.exists():
            # Backup path on zpool
            mat_path = Path("/mnt/zfspool/Backups/Jan/512GB_usb_backup_ugreen/Jan Wildenhain/614txci/EigDaten/Projects/BHonsThesis/matlab/farina/results_gapa_20021209_and_mmc_dbl.mat")
    else:
        mat_path = Path(data_path)

    if not mat_path.exists():
        raise FileNotFoundError(f"SOS dataset not found at {mat_path}")

    mat = scipy.io.loadmat(str(mat_path))
    results = mat["results"]

    X = np.array(results["r"][0, 0], dtype=float)
    U = np.array(results["p"][0, 0], dtype=float)
    sX = np.array(results["sr"][0, 0], dtype=float)
    sU = np.array(results["sp"][0, 0], dtype=float)

    return X, U, sX, sU


def run_sos_benchmark(
    data_path: Optional[str] = None,
    k_connections: int = 5,
    by_chance_runs: int = 500,
) -> Dict[str, BenchmarkMetrics]:
    """
    Executes all reverse engineering algorithms on the E. coli SOS dataset
    and returns comparative metrics table directly corresponding to Table 4.1.

    Algorithms evaluated:
        - S2G (Simple to General)
        - G2S (General to Specific)
        - FAST (Farina SVD)
        - NIR (Gardner et al.)
        - Voting Ensemble (S2G + G2S consensus)
        - By Chance (averaged over runs)
    """
    X, U, sX, sU = load_sos_dataset(data_path)
    A_true = SOS_GROUND_TRUTH

    results: Dict[str, BenchmarkMetrics] = {}

    # 1. Simple-to-General
    s2g = SimpleToGeneral()
    res_s2g = s2g.fit(X, U)
    results["s2g"] = evaluate_inference(A_true, res_s2g.adjacency_pred)

    # 2. General-to-Specific
    g2s = GeneralToSpecific()
    res_g2s = g2s.fit(X, U)
    results["g2s"] = evaluate_inference(A_true, res_g2s.adjacency_pred)

    # 3. FAST Algorithm
    fast = FastSVD(k_interactions=k_connections)
    res_fast = fast.fit(X, U)
    results["fast"] = evaluate_inference(A_true, res_fast.adjacency_pred)

    # 4. NIR Algorithm
    nir = NIRAlgorithm(k_interactions=k_connections)
    res_nir = nir.fit(X, U)
    results["NIR"] = evaluate_inference(A_true, res_nir.adjacency_pred)

    # 5. Voting Ensemble
    ensemble = VotingEnsemble(s2g=s2g, g2s=g2s)
    res_ens = ensemble.fit(X, U)
    results["voting"] = evaluate_inference(A_true, res_ens.adjacency_pred)

    # 6. By Chance (averaged)
    by_chance = ByChance(k_interactions=k_connections)
    bc_metrics_list = []
    for r in range(by_chance_runs):
        res_bc = by_chance.fit(X, U, seed=1000 + r)
        bc_metrics_list.append(evaluate_inference(A_true, res_bc.adjacency_pred))

    # Average By Chance
    avg_coverage = float(np.mean([m.coverage_pct for m in bc_metrics_list]))
    avg_fp = float(np.mean([m.false_positives_pct for m in bc_metrics_list]))
    avg_inf = float(np.mean([m.inferred_connections_pct for m in bc_metrics_list]))
    avg_fp_inf = float(np.mean([m.fp_per_inferred_pct for m in bc_metrics_list]))
    avg_sign = float(np.mean([m.inferred_sign_pct for m in bc_metrics_list]))
    avg_errors = int(np.round(np.mean([m.total_errors for m in bc_metrics_list])))

    results["by_chance"] = BenchmarkMetrics(
        coverage_pct=avg_coverage,
        false_positives_pct=avg_fp,
        inferred_connections_pct=avg_inf,
        fp_per_inferred_pct=avg_fp_inf,
        inferred_sign_pct=avg_sign,
        total_errors=avg_errors,
        true_positives=int(np.round(np.mean([m.true_positives for m in bc_metrics_list]))),
        false_positives=int(np.round(np.mean([m.false_positives for m in bc_metrics_list]))),
        true_negatives=int(np.round(np.mean([m.true_negatives for m in bc_metrics_list]))),
        false_negatives=int(np.round(np.mean([m.false_negatives for m in bc_metrics_list]))),
        sign_errors=int(np.round(np.mean([m.sign_errors for m in bc_metrics_list]))),
        precision=float(np.mean([m.precision for m in bc_metrics_list])),
        recall=float(np.mean([m.recall for m in bc_metrics_list])),
        f1_score=float(np.mean([m.f1_score for m in bc_metrics_list])),
        accuracy=float(np.mean([m.accuracy for m in bc_metrics_list])),
    )

    return results
