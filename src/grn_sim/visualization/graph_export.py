"""
Graph visualization, export, and formatting utilities for GRN-Sim.
Directly reproduces the Graphviz DOT and spring layout generation from
phpIoRN (springgraph.pl, draw_dot.m, and php_and_stats.m).
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import io


def export_graphviz_dot(
    matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
    graph_name: str = "GRN",
    layout: str = "neato",
) -> str:
    """
    Exports interaction matrix to Graphviz DOT format compatible with dot/neato.
    Activation (+1) edges are green with standard arrowhead ('vee').
    Inhibition (-1) edges are red with tee arrowhead ('tee').

    Args:
        matrix: (N, N) interaction matrix where matrix[i, j] is effect of j on i.
        gene_names: Optional list of N gene labels.
        graph_name: Name of the digraph.
        layout: 'neato' (spring) or 'dot' (hierarchical).
    """
    N = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    lines = [
        f'digraph "{graph_name}" {{',
        f'  layout="{layout}";',
        '  overlap=false;',
        '  splines=true;',
        '  node [shape=circle, style=filled, fillcolor="#EBF5FB", color="#2980B9", fontname="Helvetica", fontsize=11, width=0.6];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    for name in gene_names:
        lines.append(f'  "{name}";')

    for i in range(N):
        for j in range(N):
            val = matrix[i, j]
            if val != 0:
                source = gene_names[j]
                target = gene_names[i]
                if val > 0:
                    color = "#27AE60"   # Green activation
                    arrowhead = "vee"
                    style = "solid"
                else:
                    color = "#E74C3C"   # Red inhibition
                    arrowhead = "tee"
                    style = "solid"
                
                # Check for self-loop
                if i == j:
                    lines.append(f'  "{source}" -> "{target}" [color="{color}", arrowhead="{arrowhead}", style="{style}", dir=forward];')
                else:
                    lines.append(f'  "{source}" -> "{target}" [color="{color}", arrowhead="{arrowhead}", style="{style}"];')

    lines.append('}')
    return "\n".join(lines)


def export_graph_json(
    matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Exports network to JSON nodes and links format for modern HTML5 Canvas / D3 / Cytoscape visualizers.
    """
    N = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    # Calculate degrees
    in_degree = [int(np.sum(matrix[i, :] != 0)) for i in range(N)]
    out_degree = [int(np.sum(matrix[:, j] != 0)) for j in range(N)]

    nodes = []
    for i in range(N):
        nodes.append({
            "id": gene_names[i],
            "index": i,
            "label": gene_names[i],
            "in_degree": in_degree[i],
            "out_degree": out_degree[i],
            "degree": in_degree[i] + out_degree[i],
        })

    links = []
    for i in range(N):
        for j in range(N):
            val = float(matrix[i, j])
            if val != 0:
                links.append({
                    "source": gene_names[j],
                    "target": gene_names[i],
                    "source_idx": j,
                    "target_idx": i,
                    "value": val,
                    "sign": 1 if val > 0 else -1,
                    "type": "activation" if val > 0 else "inhibition",
                    "color": "#10b981" if val > 0 else "#ef4444",
                })

    return {"nodes": nodes, "links": links}


def export_comparison_graph_json(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Exports a comparison graph where edges are classified into:
    - TP_CORRECT: Inferred edge present with correct sign (Green solid)
    - TP_SIGN_ERR: Inferred edge present with wrong sign (Orange solid)
    - FP: Inferred edge that does not exist in ground truth (Red dashed)
    - FN: True edge missed by the inference algorithm (Gray dotted)
    """
    N = true_matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    nodes = [{"id": name, "index": idx, "label": name} for idx, name in enumerate(gene_names)]
    links = []

    for i in range(N):
        for j in range(N):
            t_val = true_matrix[i, j]
            p_val = pred_matrix[i, j]

            source = gene_names[j]
            target = gene_names[i]

            if t_val != 0 and p_val != 0:
                if np.sign(t_val) == np.sign(p_val):
                    status = "TP_CORRECT"
                    color = "#10b981" # Emerald green
                    desc = f"True Positive ({'+' if t_val > 0 else '-'})"
                else:
                    status = "TP_SIGN_ERR"
                    color = "#f59e0b" # Amber
                    desc = f"Sign Mismatch (True {'+' if t_val > 0 else '-'}, Pred {'+' if p_val > 0 else '-'})"
                links.append({
                    "source": source,
                    "target": target,
                    "source_idx": j,
                    "target_idx": i,
                    "status": status,
                    "color": color,
                    "style": "solid",
                    "true_sign": int(np.sign(t_val)),
                    "pred_sign": int(np.sign(p_val)),
                    "description": desc,
                })
            elif t_val == 0 and p_val != 0:
                # False positive
                links.append({
                    "source": source,
                    "target": target,
                    "source_idx": j,
                    "target_idx": i,
                    "status": "FP",
                    "color": "#ef4444", # Red
                    "style": "dashed",
                    "true_sign": 0,
                    "pred_sign": int(np.sign(p_val)),
                    "description": f"False Positive ({'+' if p_val > 0 else '-'})",
                })
            elif t_val != 0 and p_val == 0:
                # False negative (missed edge)
                links.append({
                    "source": source,
                    "target": target,
                    "source_idx": j,
                    "target_idx": i,
                    "status": "FN",
                    "color": "#94a3b8", # Slate/gray
                    "style": "dotted",
                    "true_sign": int(np.sign(t_val)),
                    "pred_sign": 0,
                    "description": f"False Negative / Missed ({'+' if t_val > 0 else '-'})",
                })

    return {"nodes": nodes, "links": links}


def compute_detailed_confusion_matrix(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    ignore_diagonal: bool = False,
) -> Dict[str, Any]:
    """
    Computes the 4x4 Observed vs Predicted interaction table matching phpIoRN's stats_obs_pred.m.
    Rows:
      1: Observed Activation (+1)
      2: Observed Inhibition (-1)
      3: Observed No Interaction (0)
      4: Prediction Error Rates
    Columns:
      1: Predicted Activation (+1)
      2: Predicted Inhibition (-1)
      3: Predicted No Interaction (0)
      4: Observation Error Rates / Overall
    """
    T = true_matrix.copy()
    P = pred_matrix.copy()
    N = T.shape[0]

    if ignore_diagonal:
        np.fill_diagonal(T, 0)
        np.fill_diagonal(P, 0)

    # 3x3 core counts
    # Row 0: Obs +
    # Row 1: Obs -
    # Row 2: Obs 0
    # Col 0: Pred +
    # Col 1: Pred -
    # Col 2: Pred 0
    cmat = np.zeros((3, 3), dtype=int)

    for i in range(N):
        for j in range(N):
            if ignore_diagonal and i == j:
                continue
            t = T[i, j]
            p = P[i, j]

            r = 0 if t > 0 else (1 if t < 0 else 2)
            c = 0 if p > 0 else (1 if p < 0 else 2)
            cmat[r, c] += 1

    # Error rates per row (Observation Error Rate = proportion of incorrect predictions for that true class)
    row_errors = []
    for r in range(3):
        total_in_row = np.sum(cmat[r, :])
        correct = cmat[r, r]
        err_pct = (1.0 - (correct / total_in_row)) * 100.0 if total_in_row > 0 else 0.0
        row_errors.append(round(err_pct, 1))

    # Error rates per col (Prediction Error Rate = proportion of false discoveries for that predicted class)
    col_errors = []
    for c in range(3):
        total_in_col = np.sum(cmat[:, c])
        correct = cmat[c, c]
        err_pct = (1.0 - (correct / total_in_col)) * 100.0 if total_in_col > 0 else 0.0
        col_errors.append(round(err_pct, 1))

    total_elements = np.sum(cmat)
    total_correct = np.trace(cmat)
    overall_error_pct = (1.0 - (total_correct / total_elements)) * 100.0 if total_elements > 0 else 0.0

    return {
        "matrix": cmat.tolist(),
        "row_labels": ["Observed Activation (+)", "Observed Inhibition (-)", "Observed No Interaction (0)"],
        "col_labels": ["Predicted Activation (+)", "Predicted Inhibition (-)", "Predicted No Interaction (0)"],
        "row_errors": row_errors,
        "col_errors": col_errors,
        "overall_error_pct": round(overall_error_pct, 2),
    }
