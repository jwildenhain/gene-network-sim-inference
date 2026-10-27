"""Settings shared by the tests that drive an already running web studio (marker: live_server)."""

import os
import tempfile
from pathlib import Path

# The studio under test: `grn-sim web --port 8085`, `docker run -p 8085:8080 ...`, or any other deployment.
APP_URL = os.environ.get("GRN_SIM_URL", "http://localhost:8085")

# Where these tests save screenshots (override with $ARTIFACT_DIR).
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR") or Path(tempfile.gettempdir()) / "grn-sim-artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Several of these tests also save the screenshots used in docs/screenshots. They only overwrite those tracked images
# when asked to (GRN_SIM_UPDATE_DOCS=1); otherwise the pictures go to a folder inside ARTIFACT_DIR.
if os.environ.get("GRN_SIM_UPDATE_DOCS") == "1":
    DOCS_SCREENSHOTS = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
else:
    DOCS_SCREENSHOTS = ARTIFACT_DIR / "docs_screenshots"
DOCS_SCREENSHOTS.mkdir(parents=True, exist_ok=True)
