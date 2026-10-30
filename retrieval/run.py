#!/usr/bin/env python3
"""Single entry point for the three frozen retrieval experiments.

Examples:
  python run.py figure
  python run.py evaluate --task all
  python run.py train --task cif_ed --set training.epochs=200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TASKS = {
    "cif_ed": {"config": "configs/cif_ed.yaml", "train": "train_retrieval.py", "checkpoint": "checkpoints/cif_ed_best.pt", "manifest": "data/manifests/cif_ed_test_split.csv", "metrics": "results/cif_ed_metrics.json"},
    "cif_esp": {"config": "configs/cif_esp.yaml", "train": "train_retrieval.py", "checkpoint": "checkpoints/cif_esp_best.pt", "manifest": "data/manifests/cif_esp_test_split.csv", "metrics": "results/cif_esp_metrics.json"},
    "ed_esp": {"config": "configs/ed_esp.yaml", "train": "train_ed_esp_retrieval.py", "checkpoint": "checkpoints/ed_esp_best.pt", "manifest": "data/manifests/ed_esp_test_split.csv", "metrics": "results/ed_esp_metrics.json"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the packaged MOF retrieval experiments.")
    parser.add_argument("command", choices=("train", "evaluate", "figure", "reproduce"))
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="Dataset root (default: package data/).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", nargs="*", default=[], help="Additional config overrides: key=value.")
    return parser.parse_args()


def task_names(task: str) -> list[str]:
    return list(TASKS) if task == "all" else [task]


def local_manifest(task: str, data_root: Path, output_dir: Path) -> Path:
    """Make absolute paths from the portable, source-path manifest.

    CIFs are intentionally flat by filename. ED/ESP retain their path below
    the corresponding source ``ED_fps``/``EPS_fps`` directory.
    """
    source = ROOT / TASKS[task]["manifest"]
    target = output_dir / "manifests" / f"{task}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as read_handle, target.open("w", newline="", encoding="utf-8") as write_handle:
        reader = csv.DictReader(read_handle)
        fields = reader.fieldnames or []
        writer = csv.DictWriter(write_handle, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            for column in ("cif_path", "point_path", "ed_path", "esp_path"):
                if not row.get(column):
                    continue
                original = Path(row[column])
                if column == "cif_path":
                    mapped = data_root / "cifs" / original.name
                else:
                    marker = "ED_fps" if column in ("point_path", "ed_path") and "ED_fps" in original.parts else "EPS_fps"
                    modality = "ed" if marker == "ED_fps" else "esp"
                    suffix = Path(*original.parts[original.parts.index(marker) + 1:])
                    mapped = data_root / "point_clouds" / modality / suffix
                row[column] = str(mapped)
            writer.writerow(row)
    return target


def run_task(args: argparse.Namespace, task: str) -> None:
    spec = TASKS[task]
    run_dir = args.output_dir / task
    manifest = local_manifest(task, args.data_root, args.output_dir)
    overrides = [f"data.index_csv={manifest}", f"training.output_dir={run_dir}", f"training.device={args.device}", *args.set]
    command = [sys.executable, str(SRC / spec["train"]), "--config", str(ROOT / spec["config"]), "--set", *overrides]
    if args.command == "evaluate":
        command.extend(["--checkpoint", str(ROOT / spec["checkpoint"]), "--evaluate-only"])
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def make_figure(output_dir: Path, metrics_dir: Path | None = None) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    items = [("ED ↔ CIF", "cif_ed", "ED density 0.02 · 6,144 points · Fourier k=2", "#D69A55"), ("ESP ↔ CIF", "cif_esp", "ESP 3% · 4,096 points · Fourier k=2", "#9381B2"), ("ED ↔ ESP", "ed_esp", "ED 0.03 · ESP 3% · 4,096 points · xyz + scalar + Fourier k=2", "#4C9B8F")]
    values = []
    for _, key, _, _ in items:
        metric_path = (metrics_dir / key / "metrics.json") if metrics_dir else (ROOT / TASKS[key]["metrics"])
        payload = json.loads(metric_path.read_text())
        metric = payload.get("best_test_metrics", payload.get("test_metrics"))
        if metric is None:
            raise KeyError(f"No test metrics in {metric_path}")
        directions = list(metric.values())
        values.append({name: float(np.mean([x[name] for x in directions])) for name in ("Recall@1", "Recall@5", "Recall@10", "MRR")})
    plt.rcParams.update({"font.family": "serif", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2), dpi=220)
    labels, _, descriptions, colors = map(list, zip(*items))
    x = np.arange(3)
    bars = axes[0].bar(x, [x["MRR"] for x in values], color=colors, edgecolor="#222", alpha=.9)
    axes[0].set(title="MRR across retrieval tasks", ylabel="MRR", xticks=x, xticklabels=labels, ylim=(0, 1.02))
    for bar, value in zip(bars, values): axes[0].text(bar.get_x() + bar.get_width()/2, value["MRR"]+.018, f'{value["MRR"]:.3f}', ha="center", fontweight="bold")
    for label, value, color in zip(labels, values, colors): axes[1].plot(x, [value[k] for k in ("Recall@1", "Recall@5", "Recall@10")], "o-", label=label, color=color, linewidth=2.8, markersize=8.5)
    axes[1].set(title="Recall@K across retrieval tasks", ylabel="Recall", xticks=x, xticklabels=("R@1", "R@5", "R@10"), ylim=(0, 1.02)); axes[1].legend(loc="lower right")
    for axis in axes: axis.grid(True, linestyle="--", alpha=.55); axis.set_axisbelow(True)
    fig.text(.5, .09, "Selected configurations:   " + "    |    ".join(f"{a}: {b}" for a, b in zip(labels, descriptions)), ha="center", fontsize=9.5)
    fig.text(.5, .045, "Values are bidirectional averages on the test set from the best validation checkpoint.", ha="center", fontsize=9.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "best_retrieval_tasks.png"
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    print(f"saved={target}")


def main() -> None:
    args = parse_args()
    if args.command == "figure":
        make_figure(args.output_dir)
    elif args.command == "reproduce":
        # Evaluate released weights on frozen manifests, then plot fresh metrics.
        for task in task_names(args.task):
            run_task(argparse.Namespace(**{**vars(args), "command": "evaluate"}), task)
        if args.task == "all":
            make_figure(args.output_dir / "figure", args.output_dir)
    else:
        for task in task_names(args.task): run_task(args, task)


if __name__ == "__main__":
    main()
