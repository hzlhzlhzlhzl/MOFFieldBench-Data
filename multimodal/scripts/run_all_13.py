#!/usr/bin/env python3
"""Launch every registered task using the same public training entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", default="cif_ed_esp", choices=("cif_ed", "cif_esp", "cif_ed_esp"))
    parser.add_argument("--data-manifest", default="data/dataset_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()
    tasks = json.loads((ROOT / "configs/tasks.json").read_text(encoding="utf-8"))["tasks"]
    for task in tasks:
        command = [sys.executable, str(ROOT / "scripts/run.py"), "--config", "configs/base.yaml", "--task", task,
                   "--modalities", args.modalities, "--data-manifest", args.data_manifest, *args.extra]
        if args.dry_run:
            command.append("--dry-run")
        print("[run]", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
