#!/usr/bin/env python3
"""Launch the same training entry point once for each of the 13 properties."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="ED or ESP base config")
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-set", nargs="*", default=[])
    args = ap.parse_args()
    tasks = yaml.safe_load((ROOT / "configs/tasks.yaml").read_text())["tasks"]
    for task in tasks:
        command = [sys.executable, "train.py", "--config", str(args.config), "--set",
                   f"data.target_column={task}",
                   f"training.output_dir={args.output_root / task}", *args.extra_set]
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
