#!/usr/bin/env python3
"""Create a verified, task/modality-organized local checkpoint collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Local weight collection organized as <task>/<modalities>/best_model.pt")
    args = parser.parse_args()
    records = json.loads((ROOT / "checkpoints/manifest.json").read_text(encoding="utf-8"))["checkpoints"]
    for item in records:
        source = args.source_root / item["source_checkpoint"]
        destination = ROOT / "checkpoints/artifacts" / item["task_id"] / item["modalities"] / "best_model.pt"
        if not source.is_file():
            raise FileNotFoundError(f"Missing source checkpoint: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        actual = digest(destination)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {destination}")
        print(f"ok  {item['task_id']:28s} {item['modalities']:10s} {destination}")


if __name__ == "__main__":
    main()
