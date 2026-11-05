"""
Writes docs/algorithms/ (an index and one short page per algorithm) from ``grn_sim.inference.descriptions`` and the registry.

    python scripts/generate_algorithm_docs.py            # write the pages
    python scripts/generate_algorithm_docs.py --check    # exit 1 when the committed pages are out of date

Edit the descriptions in src/grn_sim/inference/descriptions.py, never the generated pages.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

from grn_sim.inference import orientation, registry
from grn_sim.inference.descriptions import D, FAMILIES, AlgorithmDescription

OUT = Path(__file__).resolve().parent.parent / "docs" / "algorithms"
STATUS = {"core": "core (thesis)", "extended": "extended", "baseline": "baseline / control", "null": "null model",
          "experimental": "experimental (not in the registry)"}
PARTIAL = {"native": "runs on a partial screen as it is", "adapter": "needs `PartialAdapter` (`registry.build_partial`)"}


def orientation_text(key: str) -> str:
    if key in ("bicycle_lite", "dotears_lite", "interventional_consensus"):
        return "`A_est[regulator, target]` (direct effects among the perturbed genes, extended to the unperturbed ones)"
    target_row = orientation.output_is_target_regulator(key, orientation.HILL)
    additive = orientation.output_is_target_regulator(key, orientation.ADDITIVE_GARDNER)
    if key in orientation.ORIENTATION_FREE:
        return "no orientation (random)"
    if target_row:
        return "`A_est[target, regulator]` on every data model"
    if additive:
        return "`A_est[target, regulator]` on additive data, `[regulator, target]` on Hill data (`orient_prediction` handles both)"
    return "`A_est[regulator, target]`"


def page(d: AlgorithmDescription) -> str:
    spec = registry.get(d.key) if registry.is_registered(d.key) else None
    lines = [f"# {d.name}", "", f"`{d.key}` · {d.family} · status: {STATUS[d.status]}", "", d.summary, "",
             "## How it works", "", d.how, "", "## Interface", ""]
    lines.append("- Input: `fit(X, U)`: X is genes × perturbations, U is zero except for each perturbed gene's own response.")
    lines.append(f"- Output: {orientation_text(d.key)}.")
    if spec is not None:
        lines.append(f"- Partial screens (m perturbed genes of n measured): {PARTIAL[spec.partial]}.")
        lines.append(f"- Registry key: `{d.key}`; listed by `/api/algorithms` and the studio.")
    else:
        lines.append("- Not registered: called from Python on a calibrated effect estimate (`grn_sim.perturbseq.ace.estimate_ace`).")
    lines += ["", "## Measured here", "", d.measured, "",
              "Simulator results, 3-4 seeds, wide spreads: orders of magnitude, not rankings. Sources: "
              "[`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md), [`docs/results/`](../results/).", "",
              "## Reference and code", "", f"- {d.reference}", f"- Code: [`src/grn_sim/{d.module}`](../../src/grn_sim/{d.module})", ""]
    return "\n".join(lines)


def index() -> str:
    lines = ["# Inference algorithms", "",
             "One page per algorithm. All of them take the same input, `fit(X, U)`, and are scored against the simulator's ground truth; "
             "see [`docs/INFERENCE_GAP_ANALYSIS.md`](../INFERENCE_GAP_ANALYSIS.md) for how they compare and "
             "[`docs/PERTURBSEQ.md`](../PERTURBSEQ.md) for the screens they are run on.", "",
             "- **Input.** X is genes × perturbations (responses, e.g. log fold changes), U is zero except for each perturbed gene's own response. "
             "Fewer perturbation columns than genes is a *partial screen*.",
             "- **Output.** `A_est` (continuous scores) and `adjacency_pred` (signed -1/0/+1). The orientation differs between algorithms and is "
             "normalised for scoring by `grn_sim.inference.orientation`.",
             "- **Partial screens.** `native` algorithms accept them as they are; `adapter` ones need `registry.build_partial` "
             "(`PartialAdapter`: fit on the perturbed block, extended to the unperturbed genes).",
             "- **Adding one.** Write the class, `register(AlgorithmSpec(...))` in `inference/registry.py`, add a description in "
             "`inference/descriptions.py`, run `python scripts/generate_algorithm_docs.py`. A test fails until all three exist.", ""]
    by_family: Dict[str, list] = {f: [] for f in FAMILIES}
    for d in D:
        by_family[d.family].append(d)
    for fam in FAMILIES:
        lines += [f"## {fam}", "", "| algorithm | status | partial | in one line |", "|---|---|---|---|"]
        for d in by_family[fam]:
            spec = registry.get(d.key) if registry.is_registered(d.key) else None
            partial = spec.partial if spec else "experimental"
            lines.append(f"| [{d.name}]({d.key}.md) | {STATUS[d.status].split(' (')[0]} | {partial} | {d.summary.split('. ')[0].rstrip('.')}. |")
        lines.append("")
    return "\n".join(lines)


def render() -> Dict[str, str]:
    files = {"README.md": index()}
    for d in D:
        files[f"{d.key}.md"] = page(d)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    files = render()
    if args.check:
        stale = [n for n, t in files.items() if not (OUT / n).exists() or (OUT / n).read_text() != t]
        extra = [p.name for p in OUT.glob("*.md") if p.name not in files] if OUT.exists() else []
        if stale or extra:
            print("out of date:", stale + extra)
            return 1
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (OUT / n).write_text(t)
    print(f"wrote {len(files)} pages to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
