#!/usr/bin/env python3
"""Single entry point; works without installing the project itself."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from mof_fields.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
