"""
Single-Cell Perturb-seq Simulator & AnnData (.h5ad) Exporter.
Bridges phpIoRN ODE steady-state perturbation dynamics with single-cell genomics,
generating realistic 400-cell Perturb-seq datasets matching Arc Institute Virtual Cell Challenge (VCC 2026).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import anndata as ad
    from scipy import sparse
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False


def simulate_perturb_seq_counts(
    gene_names: List[str],
    mean_expression_matrix: np.ndarray,
    cells_per_condition: int = 400,
    mean_library_depth: int = 20000,
    depth_dispersion: float = 0.35,
    biological_dispersion: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Simulates single-cell integer count profiles from ODE steady-state expression levels.

    Args:
        gene_names: List of gene symbols (n_genes)
        mean_expression_matrix: (n_genes, n_conditions) array where col 0 is unperturbed control,
                                and cols 1..n_conditions are single-gene perturbations.
        cells_per_condition: Number of single cells per condition (default 400, matching VCC 2026)
        mean_library_depth: Mean total UMI count per cell (default 20,000)
        depth_dispersion: Log-normal standard deviation of cell sequencing depth
        biological_dispersion: Gamma-Poisson biological overdispersion
        seed: Random seed for reproducibility

    Returns:
        (counts, cpm, obs_df, var_df)
    """
    rng = np.random.default_rng(seed)
    n_genes, n_conditions = mean_expression_matrix.shape
    total_cells = n_conditions * cells_per_condition

    # 1. Condition labels
    condition_names = ["control"] + [f"{gene_names[j-1]}" for j in range(1, n_conditions)]

    obs_records = []
    cell_idx = 0
    all_counts = np.zeros((total_cells, n_genes), dtype=np.int32)
    all_cpm = np.zeros((total_cells, n_genes), dtype=np.float32)

    for cond_idx in range(n_conditions):
        cond_name = condition_names[cond_idx]
        is_ctrl = (cond_idx == 0)
        pert_target = "non-targeting" if is_ctrl else gene_names[cond_idx - 1]

        # Gene expression probability distribution for this condition
        base_expr = np.maximum(mean_expression_matrix[:, cond_idx], 1e-6)
        base_prob = base_expr / base_expr.sum()

        # Cell sequencing depths (log-normal distribution matching 10x Chromium)
        depths = rng.lognormal(
            mean=np.log(mean_library_depth) - 0.5 * depth_dispersion**2,
            sigma=depth_dispersion,
            size=cells_per_condition,
        ).astype(np.int64)
        depths = np.clip(depths, 1000, 150000)

        # Gamma-Poisson (Negative Binomial) single-cell sampling
        alpha = 1.0 / max(1e-4, biological_dispersion)
        for c in range(cells_per_condition):
            cell_depth = depths[c]
            # Biological gamma shock per gene
            cell_lambda = rng.gamma(shape=alpha, scale=base_prob / alpha)
            cell_lambda /= cell_lambda.sum()

            # Poisson sampling with total sequencing depth
            cell_counts = rng.poisson(cell_lambda * cell_depth).astype(np.int32)
            all_counts[cell_idx] = cell_counts
            all_cpm[cell_idx] = (cell_counts / max(1, cell_counts.sum())) * 1e6

            barcode = f"{cond_name}_cell_{c+1:04d}"
            obs_records.append({
                "cell_barcode": barcode,
                "perturbation": pert_target,
                "condition": cond_name,
                "is_control": is_ctrl,
                "total_counts": int(cell_counts.sum()),
                "detected_genes": int(np.sum(cell_counts > 0)),
            })
            cell_idx += 1

    obs_df = pd.DataFrame(obs_records)
    obs_df.index = obs_df["cell_barcode"].values

    var_df = pd.DataFrame({
        "gene_name": gene_names,
        "is_perturbed_target": [g in condition_names[1:] for g in gene_names],
        "mean_control_cpm": np.round(
            (mean_expression_matrix[:, 0] / mean_expression_matrix[:, 0].sum()) * 1e6, 2
        ),
    })
    var_df.index = gene_names

    return all_counts, all_cpm, obs_df, var_df


def create_perturb_seq_anndata(
    gene_names: List[str],
    mean_expression_matrix: np.ndarray,
    ground_truth_network: Optional[np.ndarray] = None,
    perturbation_matrix: Optional[np.ndarray] = None,
    cells_per_condition: int = 400,
    mean_library_depth: int = 20000,
    seed: int = 42,
    as_sparse: bool = True,
) -> Any:
    """
    Creates an AnnData (.h5ad compatible) single-cell dataset from simulation results.
    """
    counts, cpm, obs_df, var_df = simulate_perturb_seq_counts(
        gene_names=gene_names,
        mean_expression_matrix=mean_expression_matrix,
        cells_per_condition=cells_per_condition,
        mean_library_depth=mean_library_depth,
        seed=seed,
    )

    if not HAS_ANNDATA:
        return {
            "counts": counts,
            "cpm": cpm,
            "obs": obs_df,
            "var": var_df,
            "ground_truth_network": ground_truth_network,
            "perturbation_matrix": perturbation_matrix,
        }

    X_data = sparse.csr_matrix(counts) if as_sparse else counts
    cpm_data = sparse.csr_matrix(cpm) if as_sparse else cpm

    adata = ad.AnnData(
        X=X_data,
        obs=obs_df,
        var=var_df,
        layers={"cpm": cpm_data, "raw_counts": X_data},
    )

    if ground_truth_network is not None:
        adata.uns["ground_truth_network"] = np.asarray(ground_truth_network, dtype=float)
    if perturbation_matrix is not None:
        adata.uns["perturbation_matrix"] = np.asarray(perturbation_matrix, dtype=float)

    adata.uns["vcc_challenge"] = {
        "framework": "phpIoRN / GRN-Sim 2.0",
        "cells_per_group": cells_per_condition,
        "mean_library_depth": mean_library_depth,
        "n_conditions": mean_expression_matrix.shape[1],
    }

    return adata


def export_perturb_seq_h5ad(
    adata: Any,
    output_path: Union[str, Path],
    compression: str = "gzip",
) -> Path:
    """Saves AnnData object to .h5ad file format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if HAS_ANNDATA and isinstance(adata, ad.AnnData):
        adata.write_h5ad(out_path, compression=compression)
    else:
        # Fallback to npz container if anndata is not installed
        np.savez_compressed(
            out_path.with_suffix(".npz"),
            counts=adata["counts"],
            cpm=adata["cpm"],
            obs_json=adata["obs"].to_json(),
            var_json=adata["var"].to_json(),
            ground_truth=adata.get("ground_truth_network"),
        )
    return out_path
