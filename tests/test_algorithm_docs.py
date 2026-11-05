"""Every registered algorithm is described, and docs/algorithms/ matches what the generator writes."""
import importlib.util
from pathlib import Path

from grn_sim.inference import registry
from grn_sim.inference.descriptions import BY_KEY, D, FAMILIES

ROOT = Path(__file__).resolve().parent.parent


def _generator():
    spec = importlib.util.spec_from_file_location("generate_algorithm_docs", ROOT / "scripts" / "generate_algorithm_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_registered_algorithm_has_a_description():
    missing = sorted(set(registry.keys()) - set(BY_KEY))
    assert not missing, f"describe these algorithms in inference/descriptions.py: {missing}"
    assert all(d.family in FAMILIES and d.summary and d.how and d.measured and d.module for d in D)
    assert len(BY_KEY) == len(D)                               # no duplicate keys


def test_descriptions_point_at_existing_code():
    for d in D:
        assert (ROOT / "src" / "grn_sim" / d.module).exists(), (d.key, d.module)


def test_generated_pages_are_up_to_date():
    gen = _generator()
    files = gen.render()
    out = ROOT / "docs" / "algorithms"
    stale = [n for n, t in files.items() if not (out / n).exists() or (out / n).read_text() != t]
    assert not stale, f"run python scripts/generate_algorithm_docs.py (out of date: {stale})"
    assert {p.name for p in out.glob("*.md")} == set(files)


def test_relative_links_in_the_pages_resolve():
    import re
    out = ROOT / "docs" / "algorithms"
    for page in out.glob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text()):
            if target.startswith("http"):
                continue
            assert (page.parent / target).resolve().exists(), (page.name, target)
