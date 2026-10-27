# Contributing to GRN-Sim

Thanks for helping. This page is the short version of how to set up, check your change and send it in.
How the code is organised: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Set-up

```bash
git clone https://github.com/jwildenhain/gene-network-sim-inference.git
cd gene-network-sim-inference
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                    # add ",browser" for the Playwright tests, ",torch" for PyTorch
```
Python 3.9 or newer. [Graphviz](https://graphviz.org/) is optional (the studio has a built-in layout without it).

## Before you open a pull request

```bash
ruff check .                                   # lint (configuration in pyproject.toml)
mypy                                           # types: the package is expected to stay clean
pytest                                         # unit, simulation, inference, server and in-process browser tests
```
`pytest` runs everything that needs no external service. Browser tests start their own server and skip themselves when
Chromium is missing (`pip install -e ".[browser]" && playwright install chromium`).

Tests marked `live_server` drive a studio that is **already running** (default `http://localhost:8085`, change with
`GRN_SIM_URL`) and write screenshots to `$ARTIFACT_DIR`; they are deselected by default. Run them on purpose with
`pytest -m live_server`. They add simulations to that studio's archive, so point them at a scratch instance:

```bash
RGN_DB_PATH=/tmp/scratch.sqlite python -m grn_sim.web.server 8097 &
GRN_SIM_URL=http://localhost:8097 pytest -m live_server
```

Some of them also save the screenshots shown in `docs/screenshots/`; they only overwrite those tracked images with
`GRN_SIM_UPDATE_DOCS=1`, otherwise the pictures go to `$ARTIFACT_DIR/docs_screenshots`.

Every in-process test uses a temporary archive (see `tests/conftest.py`), so running the suite never touches your own
`rgn_db.sqlite`.

## Conventions

- **Tests with the change.** A bug fix comes with a test that fails without it; a new algorithm or benchmark comes with
  a test of its contract (shape, determinism, held-out columns never used, ...).
- **Compare methods with paired, multi-network studies.** A handful of random networks is noise: the mean-profile null
  alone ranges from 0 to 0.4 between networks. `scripts/holdout_study.py` shows the design; report differences with
  their standard errors.
- **No machine-specific paths.** Locate files through `grn_sim.paths` (bundled data, user data, archive); tests that
  need a folder use `tmp_path`.
- **No new top-level dependencies** without a reason in the pull request; heavy or optional ones go behind an extra in
  `pyproject.toml` and are imported lazily with a fallback (see `inference/cellbox.py` for PyTorch).
- Style is whatever `ruff` accepts; long lines are allowed. Add type hints to new public functions.

## Data

Read-only datasets ship inside the package (`src/grn_sim/data/`, about 2.5 MB). Everything the studio writes (the
SQLite archive, raw downloads, regenerated exports) goes to a user data folder: `$GRN_SIM_HOME`, else `<checkout>/data`
when running from a clone, else the platform's per-user data folder. `python scripts/download_data.py` checks the
bundled data and can fetch the optional 160 MB raw Norman 2019 matrix.

## Releasing (maintainers)

```bash
# 1. bump __version__ in src/grn_sim/__init__.py and move the CHANGELOG entries under the new version
python -m build                                # sdist + wheel in dist/
twine check dist/*
# 2. smoke test the wheel in a clean environment
python -m venv /tmp/grn-check && /tmp/grn-check/bin/pip install dist/grn_sim-*.whl && /tmp/grn-check/bin/grn-sim sos-benchmark
# 3. tag and upload
git tag v<version> && git push --tags && twine upload dist/*
```
The CI workflow performs the build, `twine check` and the clean-environment install on every push.

## Docker

```bash
docker build -t grn-sim-phpiorn:latest .       # installs the package non-editable and runs the tests during the build
docker run -d --name grn-sim-phpiorn -p 8085:8080 -v "$PWD/data:/app/data" grn-sim-phpiorn:latest
```
`docker-compose.yml` runs the same container (`docker compose up -d --build`).
