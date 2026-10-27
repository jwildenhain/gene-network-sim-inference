"""
Extract and build real proportional gene expression datasets from Norman et al. 2019 Perturb-seq.
Generates:
1. grn_sim/data/benchmarks/norman_2019_real_16tf.json (16 core TF subcircuit)
2. grn_sim/data/benchmarks/norman_2019_real_105gene.json (Full 105-gene square single-perturbation matrix)
"""

import json
import anndata as ad
import numpy as np

from grn_sim.paths import bundled_path, downloads_dir, source_checkout_root, user_data_dir

def build_real_norman_datasets():
    h5ad_path = downloads_dir() / "norman" / "perturb_processed.h5ad"
    # In a source checkout the rebuilt files replace the bundled ones; otherwise they go to the user's data folder,
    # which takes precedence over the bundled copy when the benchmarks are loaded.
    out_dir = bundled_path("benchmarks") if source_checkout_root() else user_data_dir() / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path)
    gene_symbols = list(adata.var["gene_name"])
    sym_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    # Identify single-gene perturbation conditions
    single_conds = {}
    for c in adata.obs["condition"].unique():
        if c == "ctrl":
            continue
        parts = c.split("+")
        if len(parts) == 2:
            if parts[0] == "ctrl" and parts[1] != "ctrl":
                single_conds[parts[1]] = c
            elif parts[1] == "ctrl" and parts[0] != "ctrl":
                single_conds[parts[0]] = c

    valid_targets = sorted([g for g in single_conds.keys() if g in sym_to_idx])
    print(f"Total valid single targets: {len(valid_targets)}")

    # Compute control baseline (7,353 cells)
    ctrl_mask = (adata.obs["condition"] == "ctrl").values
    mean_ctrl = np.array(adata.X[ctrl_mask].mean(axis=0)).flatten()

    # -------------------------------------------------------------
    # 1. 16-Gene Core Regulatory Circuit
    # -------------------------------------------------------------
    core_16 = [
        "KLF1", "SPI1", "CEBPA", "CEBPE", "ETS2", "DUSP9", "EGR1", "FOSB",
        "JUN", "RUNX1T1", "TBX2", "TBX3", "CDKN1A", "CNN1", "FEV", "MAPK1"
    ]
    idx_16 = [sym_to_idx[g] for g in core_16]
    c_16 = mean_ctrl[idx_16]

    pert_16 = np.zeros((16, 16))
    cells_16 = []
    for j, g in enumerate(core_16):
        cond = single_conds[g]
        cond_mask = (adata.obs["condition"] == cond).values
        cells_16.append(int(cond_mask.sum()))
        pert_16[:, j] = np.array(adata.X[cond_mask].mean(axis=0)).flatten()[idx_16]

    eps = 0.05
    # Proportional expression shift: (pert - ctrl) / (ctrl + eps)
    X_prop_16 = (pert_16 - c_16[:, None]) / (c_16[:, None] + eps)
    # Log2 fold change: log2((pert + 0.01) / (ctrl + 0.01))
    X_logfc_16 = np.log2((pert_16 + 0.01) / (c_16[:, None] + 0.01))
    # On-target CRISPRa activation diagonal
    U_diag_16 = np.maximum(np.diag(X_prop_16), 0.5)
    U_mat_16 = np.diag(U_diag_16)

    # Literature-known TF-TF ground truth regulatory wiring (16 genes)
    # Directed: A[i, j] indicates effect of regulator j on target i
    # Key known biology:
    # - KLF1 activates erythroid program, represses myeloid SPI1
    # - SPI1 (PU.1) represses KLF1, activates CEBPA/CEBPE
    # - CEBPA activates CEBPE, CDKN1A
    # - DUSP9 dephosphorylates and represses MAPK1/ERK activity
    # - JUN / FOSB form AP-1 heterodimer, regulate EGR1
    # - RUNX1T1 (ETO) acts as transcriptional repressor
    A_known_16 = np.zeros((16, 16), dtype=int)
    g_to_idx = {g: i for i, g in enumerate(core_16)}

    # Positive activations (+1)
    activations = [
        ("KLF1", "CNN1"), ("KLF1", "FEV"),
        ("SPI1", "CEBPA"), ("SPI1", "CEBPE"), ("SPI1", "EGR1"),
        ("CEBPA", "CEBPE"), ("CEBPA", "CDKN1A"),
        ("JUN", "FOSB"), ("FOSB", "JUN"), ("JUN", "EGR1"), ("FOSB", "EGR1"),
        ("ETS2", "JUN"), ("ETS2", "EGR1"),
        ("MAPK1", "ETS2"), ("MAPK1", "JUN"), ("MAPK1", "EGR1"),
        ("TBX2", "TBX3"), ("TBX3", "TBX2")
    ]
    # Negative repressions (-1)
    repressions = [
        ("KLF1", "SPI1"), ("SPI1", "KLF1"),
        ("DUSP9", "MAPK1"), ("DUSP9", "ETS2"), ("DUSP9", "JUN"),
        ("RUNX1T1", "CEBPA"), ("RUNX1T1", "CEBPE"),
        ("CDKN1A", "JUN"), ("CNN1", "ETS2")
    ]
    for src, dst in activations:
        if src in g_to_idx and dst in g_to_idx:
            A_known_16[g_to_idx[dst], g_to_idx[src]] = 1
    for src, dst in repressions:
        if src in g_to_idx and dst in g_to_idx:
            A_known_16[g_to_idx[dst], g_to_idx[src]] = -1

    ds_16 = {
        "id": "norman_2019_real_16tf",
        "name": "Norman et al. 2019 Real CRISPRa Perturb-seq (16 TFs, K562)",
        "category": "Real-World Single-Cell CRISPR Screen",
        "gene_names": core_16,
        "X": X_prop_16.tolist(),
        "X_logfc": X_logfc_16.tolist(),
        "U": U_mat_16.tolist(),
        "true_matrix": A_known_16.tolist(),
        "mean_ctrl": c_16.tolist(),
        "cells_per_condition": cells_16,
        "num_genes": 16,
        "num_experiments": 16,
        "metadata": {
            "source": "Norman et al., Science 2019 (DOI: 10.1126/science.aax4438)",
            "data_type": "Real Experimental Perturb-seq Proportional Shift",
            "cell_line": "K562 (human erythroleukemia)",
            "perturbation_type": "CRISPRa (dCas9-VPR overexpression)",
            "cells_control": int(ctrl_mask.sum()),
            "total_cells_analyzed": int(sum(cells_16)),
        }
    }

    out_file_16 = out_dir / "norman_2019_real_16tf.json"
    with open(out_file_16, "w") as f:
        json.dump(ds_16, f, indent=2)
    print(f"Saved 16-TF real dataset to {out_file_16}")

    # -------------------------------------------------------------
    # 2. 105-Gene Full Single-Perturbation Matrix
    # -------------------------------------------------------------
    idx_105 = [sym_to_idx[g] for g in valid_targets]
    c_105 = mean_ctrl[idx_105]

    pert_105 = np.zeros((105, 105))
    cells_105 = []
    for j, g in enumerate(valid_targets):
        cond = single_conds[g]
        cond_mask = (adata.obs["condition"] == cond).values
        cells_105.append(int(cond_mask.sum()))
        pert_105[:, j] = np.array(adata.X[cond_mask].mean(axis=0)).flatten()[idx_105]

    X_prop_105 = (pert_105 - c_105[:, None]) / (c_105[:, None] + eps)
    U_diag_105 = np.maximum(np.diag(X_prop_105), 0.5)
    U_mat_105 = np.diag(U_diag_105)

    ds_105 = {
        "id": "norman_2019_real_105gene",
        "name": "Norman et al. 2019 Real CRISPRa Perturb-seq (105 Targets, Full Matrix)",
        "category": "Real-World Single-Cell CRISPR Screen",
        "gene_names": valid_targets,
        "X": X_prop_105.tolist(),
        "U": U_mat_105.tolist(),
        "mean_ctrl": c_105.tolist(),
        "cells_per_condition": cells_105,
        "num_genes": 105,
        "num_experiments": 105,
        "metadata": {
            "source": "Norman et al., Science 2019 (DOI: 10.1126/science.aax4438)",
            "data_type": "Real Experimental Perturb-seq Proportional Shift",
            "cell_line": "K562",
            "perturbation_type": "CRISPRa (dCas9-VPR)",
            "cells_control": int(ctrl_mask.sum()),
            "total_cells_analyzed": int(sum(cells_105)),
        }
    }

    out_file_105 = out_dir / "norman_2019_real_105gene.json"
    with open(out_file_105, "w") as f:
        json.dump(ds_105, f, indent=2)
    print(f"Saved 105-gene real dataset to {out_file_105}")

if __name__ == "__main__":
    build_real_norman_datasets()
