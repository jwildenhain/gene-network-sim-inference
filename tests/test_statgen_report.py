"""docs/results/statgen_calibration.md is rendered from the calibrated parameters and the stored verification results."""
import json
from pathlib import Path

from grn_sim.statgen.calibration import CALIBRATED
from grn_sim.statgen.report import render

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_document_matches_code_and_verification_data():
    results = json.loads((ROOT / "docs" / "results" / "statgen_noise_verification.json").read_text())
    doc = (ROOT / "docs" / "results" / "statgen_calibration.md").read_text()
    assert doc == render(results), "regenerate with scripts/verify_statgen_noise.py --out ... --doc ..."
    for g, p in CALIBRATED.items():
        assert f"| {g} | {p.effect_scale:g} | {p.noise:g} | {p.weight_spread:g} |" in doc
    assert {r["generator"] for r in results} == set(CALIBRATED) | {"mechanistic"}
