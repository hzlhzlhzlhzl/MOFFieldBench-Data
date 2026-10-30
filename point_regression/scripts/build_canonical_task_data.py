#!/usr/bin/env python3
"""Build canonical Fig. 1 task directories: id_prop.csv + splits.csv."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/tasks"


def write_point_labels(task_dir, labels, splits, source):
    with (task_dir / "point_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "target", "split", "source"])
        writer.writeheader()
        writer.writerows({"id": r["id"], "target": labels[r["id"]], "split": r["split"], "source": source}
                         for r in splits if r["id"] in labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Local archived task data root")
    WORK = parser.parse_args().source_root
    OUT.mkdir(parents=True, exist_ok=True)
    merged_path = WORK / "output_merged_2w_qmof/data/combined_geometry_labels.csv"
    with merged_path.open(encoding="utf-8", newline="") as handle:
        merged = {row["base"]: row for row in csv.DictReader(handle)}
    card = {"datasets": {}, "bandgap_manifest": "../manifests/qmof_bandgap_manifest.csv"}
    for task in ("Df", "Di", "Density"):
        task_dir = OUT / task; task_dir.mkdir(exist_ok=True)
        split_file = WORK / "output_merged_2w_qmof/splits" / f"{task}.csv"
        with split_file.open(encoding="utf-8", newline="") as handle:
            splits = [r for r in csv.DictReader(handle) if r["id"] in merged]
        labels = {r["id"]: merged[r["id"]][task] for r in splits}
        with (task_dir / "id_prop.csv").open("w", encoding="utf-8", newline="") as h:
            csv.writer(h).writerows(labels.items())
        with (task_dir / "splits.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=["id", "split"]); w.writeheader(); w.writerows(splits)
        write_point_labels(task_dir, labels, splits, "output_merged_2w_qmof")
        card["datasets"][task] = {"samples": len(labels), "split_counts": {x: sum(r["split"] == x for r in splits) for x in ("train", "val", "test")}, "origin": "merged_40k"}
    strict_root = WORK / "data/cif_all_strict_e50"
    for source in sorted(strict_root.iterdir()):
        if not source.is_dir() or source.name in {"Df", "Di", "Density"} or not (source / "id_prop.csv").is_file():
            continue
        task_dir = OUT / source.name; task_dir.mkdir(exist_ok=True)
        for filename in ("id_prop.csv", "splits.csv"):
            shutil.copy2(source / filename, task_dir / filename)
        with (task_dir / "id_prop.csv").open(encoding="utf-8", newline="") as h:
            labels = {r[0]: r[1] for r in csv.reader(h)}
        with (task_dir / "splits.csv").open(encoding="utf-8", newline="") as h:
            splits = list(csv.DictReader(h))
        write_point_labels(task_dir, labels, splits, "data/cif_all_strict_e50")
        card["datasets"][source.name] = {"samples": len(labels), "split_counts": {x: sum(r["split"] == x for r in splits) for x in ("train", "val", "test")}, "origin": "strict_task_split"}
    (OUT / "DATASET_CARD.json").write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
