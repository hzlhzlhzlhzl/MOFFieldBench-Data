#!/usr/bin/env python3
"""Materialize canonical per-task labels and a reproducible 80/10/10 split."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "ASA", "Df", "Di", "POAVAf", "Density",
    "post-combustion-vsa__selectivity", "post-combustion-vsa__wt%_working_capacity",
    "natural-gas-purification__selectivity", "natural-gas-purification__purity",
    "landfill-gas-vpsa__selectivity", "methane-storage-psa__v/v_uptake",
    "methane-storage-psa__wt%_uptake", "pre-combustion-40-40__selectivity",
]


def main() -> None:
    source = ROOT / "data/labels/selected_mofs_wide.csv"
    output = ROOT / "data/property_13tasks"
    output.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["base"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("base IDs must be unique")
    rng = random.Random(42)
    rng.shuffle(ids)
    n = len(ids); n_train = int(n * .8); n_val = int(n * .1)
    split = {sample_id: "train" if i < n_train else "val" if i < n_train + n_val else "test"
             for i, sample_id in enumerate(ids)}
    with (output / "split_seed42.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "split"])
        writer.writeheader(); writer.writerows({"id": x, "split": split[x]} for x in ids)
    stats = {"seed": 42, "source": str(source.relative_to(ROOT)), "samples": n,
             "split_counts": {name: list(split.values()).count(name) for name in ("train", "val", "test")},
             "tasks": {}}
    for task in TASKS:
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        selected = [row for row in rows if row.get(task, "") not in {"", "nan", "NaN"}]
        # CGCNN-compatible files: id_prop.csv has no header, splits.csv has one.
        with (task_dir / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows((row["base"], row[task]) for row in selected)
        with (task_dir / "splits.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "split"])
            writer.writeheader(); writer.writerows({"id": row["base"], "split": split[row["base"]]} for row in selected)
        # Same information with an explicit header for point-cloud scripts.
        with (task_dir / "point_labels.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "target", "filename", "cif_source_path", "split"])
            writer.writeheader()
            for row in selected:
                writer.writerow({"id": row["base"], "target": row[task], "filename": row["filename"],
                                 "cif_source_path": row["cif_source_path"], "split": split[row["base"]]})
        stats["tasks"][task] = {"samples": len(selected), "id_prop": f"{task}/id_prop.csv", "splits": f"{task}/splits.csv", "point_labels": f"{task}/point_labels.csv"}
    (output / "dataset_card.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
