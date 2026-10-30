#!/usr/bin/env python3
"""Build the audited 13-task checkpoint index from the archived experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
ARCHIVE = PACKAGE.parent
RESULTS = ARCHIVE / "outputs/12_tasks_symmetric_attention_image1_configs_seed42_e50_localcache/ALL_72_RESULTS_WITH_RESIDUAL_BASELINE.csv"
TASKS = json.loads((PACKAGE / "configs/tasks.json").read_text(encoding="utf-8"))["tasks"]
def normalize_target(value: str) -> str:
    return value.replace("/g", "_per_g")


REVERSE = {normalize_target(entry["target_column"]): task_id for task_id, entry in TASKS.items()}
REVERSE.update({"Df": "df", "Di": "di", "Density": "density"})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    table = pd.read_csv(RESULTS)
    records = []
    for (task, modality), group in table.groupby(["task", "modality"], sort=True):
        task_id = REVERSE[normalize_target(task)]
        # The CSV has the two scratch fusion runs; choose the reported lowest MAE.
        best = group.loc[group.mae.idxmin()]
        path = ARCHIVE / "outputs/12_tasks_symmetric_attention_image1_configs_seed42_e50_localcache" / best.metrics_path.replace("metrics.json", "best_model.pt")
        records.append({"task_id": task_id, "display_name": TASKS[task_id]["display_name"], "modalities": modality,
                        "fusion_type": best.method, "seed": 42, "test_mae": float(best.mae), "test_rmse": float(best.rmse),
                        "test_r2": float(best.r2), "source_checkpoint": str(path), "sha256": sha256(path)})
    # The final figure used these three exact pure-scratch QMOF runs.
    bandgap = [
        ("cif_ed", "attention_gated", 0.2761619938909706, 0.8743986812681522, "attention_edcif"),
        ("cif_esp", "symmetric_gated", 0.2876353389530783, 0.8697078409236922, "symmetric_esp_cif"),
        ("cif_ed_esp", "symmetric_gated", 0.2808968519223824, 0.8688295166182332, "symmetric_edesp_cif"),
    ]
    for modality, fusion, mae, r2, run in bandgap:
        path = ARCHIVE.parent / "point_banggap/fusion_gated_attention_scratch_localcache/runs" / run / "best_model.pt"
        records.append({"task_id": "bandgap", "display_name": "Band gap", "modalities": modality, "fusion_type": fusion,
                        "seed": 42, "test_mae": mae, "test_rmse": None, "test_r2": r2,
                        "source_checkpoint": str(path), "sha256": sha256(path)})
    records.sort(key=lambda x: (x["task_id"], x["modalities"]))
    output = {"schema_version": 1, "selection": "Lowest reported test MAE used for the final 13-task figure.", "checkpoints": records}
    (PACKAGE / "checkpoints/manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
