#!/usr/bin/env python3
"""Stable package launcher; works from any current working directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from mof_multimodal.train import main  # noqa: E402


if __name__ == "__main__":
    main()
