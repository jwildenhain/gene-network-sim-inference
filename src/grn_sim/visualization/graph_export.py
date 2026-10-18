"""
Graph visualization, export, and formatting utilities for GRN-Sim.
Directly reproduces the Graphviz DOT, spring layout, 9-state comparison graphs,
and confusion matrices from phpIoRN (springgraph.pl, draw_dot.m, statsgraph_to_dot.m,
new_stats.m, and php_and_stats.m).
"""

from typing import Dict, List, Any, Optional, Tuple
import shutil
import subprocess
import math
import numpy as np


# Exact 9-state edge style specification from statsgraph_to_dot.m
NINE_STATE_STYLES: Dict[int, Dict[str, str]] = {
    1: {
        "code": "+1",
        "status": "TP_ACT",
        "label": "True Positive Activator (+1)",
        "color": "#10b981",          # Emerald / black in paper
        "dot_color": "#111827",
        "arrowhead": "normal",
        "style": "solid",
    },
    -1: {
        "code": "-1",
        "status": "TP_INH",
        "label": "True Positive Inhibitor (-1)",
        "color": "#ef4444",          # Red solid tee
        "dot_color": "#dc2626",
        "arrowhead": "tee",
        "style": "solid",
    },
    2: {
        "code": "+2",
        "status": "FN_ACT",
        "label": "False Negative / Missed Activator (+2)",
        "color": "#94a3b8",          # Grey solid normal
        "dot_color": "#9ca3af",
        "arrowhead": "normal",
        "style": "solid",
    },
    -2: {
        "code": "-2",
        "status": "FN_INH",
        "label": "False Negative / Missed Inhibitor (-2)",
        "color": "#fda4af",          # Light pink solid tee
        "dot_color": "#f472b6",
        "arrowhead": "tee",
        "style": "solid",
    },
    3: {
        "code": "+3",
        "status": "FP_ACT",
        "label": "False Positive Activator (+3)",
        "color": "#64748b",          # Grey dotted normal
        "dot_color": "#6b7280",
        "arrowhead": "normal",
        "style": "dotted",
    },
    -3: {
        "code": "-3",
        "status": "FP_INH",
        "label": "False Positive Inhibitor (-3)",
        "color": "#FA6378",          # Salmon dotted tee (exact hex from statsgraph_to_dot.m)
        "dot_color": "#FA6378",
        "arrowhead": "tee",
        "style": "dotted",
    },
    4: {
        "code": "+4",
        "status": "SIGN_ACT",
        "label": "Sign Error on True Activator (+4)",
        "color": "#F5C491",          # Sand dashed normal (exact hex from statsgraph_to_dot.m)
        "dot_color": "#d97706",
        "arrowhead": "normal",
        "style": "dashed",
    },
    -4: {
        "code": "-4",
        "status": "SIGN_INH",
        "label": "Sign Error on True Inhibitor (-4)",
        "color": "#B7673C",          # Brick-red dashed tee (exact hex from statsgraph_to_dot.m)
        "dot_color": "#B7673C",
        "arrowhead": "tee",
        "style": "dashed",
    },
}


def compute_new_stats_9state(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    Reproduces MATLAB `new_stats.m` and `php_and_stats.m` (`$TMP_STATMAT`).
    Classifies every pair (i, j) into the 9-state matrix A_stats:
      +1: True Positive Activator (A_orig > 0 and A_new > 0)
      -1: True Positive Inhibitor (A_orig < 0 and A_new < 0)
      +2: False Negative Activator (A_orig > 0 and A_new == 0)
      -2: False Negative Inhibitor (A_orig < 0 and A_new == 0)
      +3: False Positive Activator (A_orig == 0 and A_new > 0)
      -3: False Positive Inhibitor (A_orig == 0 and A_new < 0)
      +4: Sign Error on True Activator (A_orig > 0 and A_new < 0)
      -4: Sign Error on True Inhibitor (A_orig < 0 and A_new > 0)
       0: True Negative (A_orig == 0 and A_new == 0)
    Note: new_stats.m conventions use A(source, target) or A(target, source) consistently.
    Here true_matrix[i, j] is effect of gene j on gene i (row=target i, col=source j).
    """
    N = true_matrix.shape[0]
    a_stats = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            orig = true_matrix[i, j]
            pred = pred_matrix[i, j]
            if orig > 0 and pred > 0:
                a_stats[i, j] = 1
            elif orig < 0 and pred < 0:
                a_stats[i, j] = -1
            elif orig > 0 and pred == 0:
                a_stats[i, j] = 2
            elif orig < 0 and pred == 0:
                a_stats[i, j] = -2
            elif orig == 0 and pred > 0:
                a_stats[i, j] = 3
            elif orig == 0 and pred < 0:
                a_stats[i, j] = -3
            elif orig > 0 and pred < 0:
                a_stats[i, j] = 4
            elif orig < 0 and pred > 0:
                a_stats[i, j] = -4
            else:
                a_stats[i, j] = 0

    total_cells = N * N
    true_edges = int(np.sum(true_matrix != 0))
    absent_edges = total_cells - true_edges
    pred_edges = int(np.sum(pred_matrix != 0))

    tp_signed = int(np.sum((a_stats == 1) | (a_stats == -1)))
    sign_err = int(np.sum((a_stats == 4) | (a_stats == -4)))
    tp_unsigned = tp_signed + sign_err
    fn_count = int(np.sum((a_stats == 2) | (a_stats == -2)))
    fp_count = int(np.sum((a_stats == 3) | (a_stats == -3)))
    tn_count = int(np.sum(a_stats == 0))

    # Matching new_stats.m & php_and_stats.m $TMP_STATMAT definitions:
    fp_rate_pct = (fp_count / absent_edges * 100.0) if absent_edges > 0 else 0.0
    specificity_pct = 100.0 - fp_rate_pct
    power_pct = (tp_unsigned / true_edges * 100.0) if true_edges > 0 else 0.0
    fn_rate_pct = 100.0 - power_pct
    fdr_pct = (fp_count / pred_edges * 100.0) if pred_edges > 0 else 0.0
    ppv_pct = 100.0 - fdr_pct if pred_edges > 0 else 0.0
    signed_ppv_pct = (tp_signed / tp_unsigned * 100.0) if tp_unsigned > 0 else 0.0
    total_errors = fp_count + fn_count + sign_err
    coverage_pct = ((tn_count + tp_signed) / total_cells * 100.0) if total_cells > 0 else 0.0

    # Breakdown of 9 states
    state_counts = {
        "+1": int(np.sum(a_stats == 1)),
        "-1": int(np.sum(a_stats == -1)),
        "+2": int(np.sum(a_stats == 2)),
        "-2": int(np.sum(a_stats == -2)),
        "+3": int(np.sum(a_stats == 3)),
        "-3": int(np.sum(a_stats == -3)),
        "+4": int(np.sum(a_stats == 4)),
        "-4": int(np.sum(a_stats == -4)),
        "0": tn_count,
    }

    return {
        "a_stats": a_stats.tolist(),
        "state_counts": state_counts,
        "tmp_statmat": {
            "false_positive_rate": round(fp_rate_pct, 1),
            "false_negative_rate": round(fn_rate_pct, 1),
            "specificity": round(specificity_pct, 1),
            "power": round(power_pct, 1),
            "ppv": round(ppv_pct, 1),
            "fdr": round(fdr_pct, 1),
            "signed_ppv": round(signed_ppv_pct, 1),
            "total_errors": total_errors,
            "coverage": round(coverage_pct, 1),
        },
    }


def export_graphviz_dot(
    matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
    graph_name: str = "GRN",
    layout: str = "neato",
) -> str:
    """
    Exports interaction matrix to Graphviz DOT format compatible with dot/neato.
    Matches phpIoRN's draw_dot.m including clickable node and edge URLs/Basic IDs.
    Activation (+1) edges have standard arrowhead ('normal').
    Inhibition (-1) edges are red with tee arrowhead ('tee').
    """
    N = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    lines = [
        f'digraph "{graph_name}" {{',
        f'  layout="{layout}";',
        '  overlap=false;',
        '  splines=true;',
        '  bgcolor="transparent";',
        '  node [shape=ellipse, style=filled, fillcolor="#1e293b", color="#38bdf8", fontcolor="#f8fafc", fontname="Helvetica", fontsize=11, width=0.65, height=0.45, penwidth=1.8];',
        '  edge [fontname="Helvetica", fontsize=9, penwidth=1.8];',
    ]

    for idx, name in enumerate(gene_names):
        lines.append(
            f'  "{name}" [id="node_{idx}", URL="javascript:window.inspectNode({idx})", tooltip="Click to inspect {name} ODE kinetics & basal steady state"];'
        )

    for i in range(N):
        for j in range(N):
            val = matrix[i, j]
            if val != 0:
                source = gene_names[j]
                target = gene_names[i]
                if val > 0:
                    color = "#10b981"   # Green activation
                    arrowhead = "normal"
                    style = "solid"
                    act_str = "Activation (+1)"
                else:
                    color = "#ef4444"   # Red inhibition
                    arrowhead = "tee"
                    style = "solid"
                    act_str = "Inhibition (-1)"

                lines.append(
                    f'  "{source}" -> "{target}" [id="edge_{j}_{i}", color="{color}", arrowhead="{arrowhead}", style="{style}", '
                    f'URL="javascript:window.inspectEdge({j},{i})", tooltip="{source} -> {target}: {act_str} (Click for Hill kinetics)"];'
                )

    lines.append('}')
    return "\n".join(lines)


def export_9state_graphviz_dot(
    a_stats: np.ndarray,
    gene_names: Optional[List[str]] = None,
    graph_name: str = "GRN_Comparison",
    layout: str = "dot",
) -> str:
    """
    Reproduces MATLAB `statsgraph_to_dot.m` 9-state differential Graphviz DOT export.
    """
    N = a_stats.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    lines = [
        f'digraph "{graph_name}" {{',
        f'  layout="{layout}";',
        '  overlap=false;',
        '  splines=true;',
        '  bgcolor="transparent";',
        '  node [shape=ellipse, style=filled, fillcolor="#1e293b", color="#6366f1", fontcolor="#f8fafc", fontname="Helvetica", fontsize=11, width=0.65, height=0.45, penwidth=1.8];',
        '  edge [fontname="Helvetica", fontsize=9, penwidth=1.8];',
    ]

    for idx, name in enumerate(gene_names):
        lines.append(
            f'  "{name}" [id="node_{idx}", URL="javascript:window.inspectNode({idx})", tooltip="Gene {name}"];'
        )

    for i in range(N):
        for j in range(N):
            code = int(a_stats[i, j])
            if code != 0 and code in NINE_STATE_STYLES:
                st = NINE_STATE_STYLES[code]
                source = gene_names[j]
                target = gene_names[i]
                lines.append(
                    f'  "{source}" -> "{target}" [id="edge_{j}_{i}", color="{st["color"]}", arrowhead="{st["arrowhead"]}", '
                    f'style="{st["style"]}", tooltip="{source} -> {target}: {st["label"]}"];'
                )

    lines.append('}')
    return "\n".join(lines)


def render_graphviz_svg(
    dot_source: str,
    engine: str = "dot",
    matrix: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
    is_9state: bool = False,
) -> str:
    """
    Renders a DOT string to an interactive SVG using the system Graphviz binary (`dot` or `neato`),
    exactly like phpIoRN's `draw_dot.m` (`dot -Tsvg` / `neato -Tsvg`).
    Falls back to a clean native SVG generator if the Graphviz binary is not installed.
    """
    bin_path = shutil.which(engine) or shutil.which("dot")
    if bin_path:
        try:
            cmd = [bin_path, f"-K{engine}", "-Tsvg"]
            proc = subprocess.run(
                cmd,
                input=dot_source.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=8,
                check=True,
            )
            svg_out = proc.stdout.decode("utf-8", errors="replace")
            # Strip XML doctype header for clean inline HTML5 embedding
            idx = svg_out.find("<svg")
            if idx != -1:
                svg_out = svg_out[idx:]
            return svg_out
        except Exception:
            pass

    # Pure-Python SVG fallback when Graphviz CLI is absent
    if matrix is None:
        return '<svg viewBox="0 0 400 100" xmlns="http://www.w3.org/2000/svg"><text x="20" y="50" fill="#94a3b8">Graphviz SVG unavailable</text></svg>'

    N = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    width, height = 560, 420
    cx, cy = width / 2, height / 2
    positions: List[Tuple[float, float]] = []

    if engine == "dot":
        # Hierarchical layered layout
        cols = max(2, int(math.ceil(math.sqrt(N * 1.4))))
        rows = max(1, int(math.ceil(N / cols)))
        for idx in range(N):
            r = idx // cols
            c = idx % cols
            x = 80 + (c + 0.5 * (r % 2)) * ((width - 160) / max(1, cols - 0.5))
            y = 65 + r * ((height - 130) / max(1, rows - 1 if rows > 1 else 1))
            positions.append((x, y))
    else:
        # Spring / circular layout
        radius = min(width, height) * 0.37
        for idx in range(N):
            angle = (2 * math.pi * idx) / N - math.pi / 2
            positions.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    svg = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="background:transparent;font-family:Inter,sans-serif;">'
    ]
    # Edges
    for i in range(N):
        for j in range(N):
            val = int(matrix[i, j])
            if val == 0:
                continue
            x1, y1 = positions[j]
            x2, y2 = positions[i]
            if is_9state and val in NINE_STATE_STYLES:
                st = NINE_STATE_STYLES[val]
                color = st["color"]
                dash = 'stroke-dasharray="6,4"' if st["style"] == "dashed" else ('stroke-dasharray="2,4"' if st["style"] == "dotted" else "")
            else:
                color = "#10b981" if val > 0 else "#ef4444"
                dash = ""
            if i == j:
                svg.append(
                    f'<circle cx="{x1}" cy="{y1-24}" r="14" fill="none" stroke="{color}" stroke-width="2" {dash} '
                    f'style="cursor:pointer" onclick="window.inspectEdge && window.inspectEdge({j},{i})"/>'
                )
            else:
                svg.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2" {dash} '
                    f'style="cursor:pointer" onclick="window.inspectEdge && window.inspectEdge({j},{i})"/>'
                )

    # Nodes
    for idx, (x, y) in enumerate(positions):
        name = gene_names[idx]
        svg.append(
            f'<g style="cursor:pointer" onclick="window.inspectNode && window.inspectNode({idx})">'
            f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="26" ry="18" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" fill="#f8fafc" font-size="11" font-weight="600">{name}</text>'
            f'</g>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def export_graph_json(
    matrix: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Exports network to JSON nodes and links format for HTML5 Canvas / SVG visualizers.
    """
    N = matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

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
    Exports a comparison graph where edges carry both the 4-class summary and
    the full 9-state classification from `new_stats.m` / `statsgraph_to_dot.m`.
    """
    N = true_matrix.shape[0]
    if gene_names is None:
        gene_names = [f"G{i+1}" for i in range(N)]

    stats_9 = compute_new_stats_9state(true_matrix, pred_matrix)
    a_stats = np.array(stats_9["a_stats"], dtype=int)

    nodes = [{"id": name, "index": idx, "label": name} for idx, name in enumerate(gene_names)]
    links = []

    for i in range(N):
        for j in range(N):
            code = int(a_stats[i, j])
            if code == 0:
                continue
            t_val = true_matrix[i, j]
            p_val = pred_matrix[i, j]
            st = NINE_STATE_STYLES[code]

            if code in (1, -1):
                legacy_status = "TP_CORRECT"
            elif code in (4, -4):
                legacy_status = "TP_SIGN_ERR"
            elif code in (3, -3):
                legacy_status = "FP"
            else:
                legacy_status = "FN"

            links.append({
                "source": gene_names[j],
                "target": gene_names[i],
                "source_idx": j,
                "target_idx": i,
                "status": legacy_status,
                "state_code": code,
                "state_str": st["code"],
                "state_status": st["status"],
                "color": st["color"],
                "style": st["style"],
                "arrowhead": st["arrowhead"],
                "true_sign": int(np.sign(t_val)),
                "pred_sign": int(np.sign(p_val)),
                "description": st["label"],
            })

    return {
        "nodes": nodes,
        "links": links,
        "state_counts": stats_9["state_counts"],
        "tmp_statmat": stats_9["tmp_statmat"],
    }


def compute_detailed_confusion_matrix(
    true_matrix: np.ndarray,
    pred_matrix: np.ndarray,
    ignore_diagonal: bool = False,
) -> Dict[str, Any]:
    """
    Computes the 4x4 Observed vs Predicted interaction table matching phpIoRN's
    `stats_obs_pred.m` and `$TMP_CMAT` in `get_results.php`.
    Rows:
      1: Obs + (Observed Activation)
      2: Obs - (Observed Inhibition)
      3: Obs no (Observed No Interaction)
      4: Error rates (Prediction Error Rates per column)
    Columns:
      1: Prediction +
      2: Prediction -
      3: Prediction no
      4: Overall error (Observation Error Rates per row)
    """
    T = true_matrix.copy()
    P = pred_matrix.copy()
    N = T.shape[0]

    if ignore_diagonal:
        np.fill_diagonal(T, 0)
        np.fill_diagonal(P, 0)

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

    row_errors = []
    for r in range(3):
        total_in_row = np.sum(cmat[r, :])
        correct = cmat[r, r]
        err_pct = (1.0 - (correct / total_in_row)) * 100.0 if total_in_row > 0 else 0.0
        row_errors.append(round(err_pct, 1))

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
        "row_labels": ["Obs + (Activation)", "Obs - (Inhibition)", "Obs no (0)"],
        "col_labels": ["Prediction +", "Prediction -", "Prediction no"],
        "row_errors": row_errors,
        "col_errors": col_errors,
        "overall_error_pct": round(overall_error_pct, 2),
    }


def format_matrix_tsv(matrix: np.ndarray) -> str:
    """Formats an (N, M) matrix into space/tab-separated text matching .msmt and .pert files."""
    lines = []
    for row in matrix:
        lines.append("\t".join(f"{float(v):.6f}" for v in row))
    return "\n".join(lines) + "\n"
