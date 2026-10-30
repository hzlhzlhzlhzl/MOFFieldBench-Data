"""Portable MOF ED/ESP training, inference and paper-figure reproduction."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
# The upstream model retains its original `src.*` namespace.
for path in (ROOT / "models/charge3net", ROOT / "third_party/cp3d/src"):
    sys.path.insert(0, str(path))
