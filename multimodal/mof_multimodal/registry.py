"""Task and modality registry helpers used by the public CLI."""

from __future__ import annotations

import copy
import json
from pathlib import Path


MODALITY_ALIASES = {
    "cif": "cif_only",
    "cif_ed": "cif_ed",
    "cif_esp": "cif_esp",
    "cif_ed_esp": "cif_ed_esp",
    "cif_esp_ed": "cif_ed_esp",
}


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def apply_registry_selection(config: dict, tasks_path: str, data_path: str, task_id: str, modalities: str) -> dict:
    """Apply one task plus one of the three multimodal combinations."""
    tasks = load_json(tasks_path)["tasks"]
    data = load_json(data_path)
    if task_id not in tasks:
        raise KeyError(f"Unknown task {task_id!r}; use --list-tasks")
    mode = MODALITY_ALIASES.get(modalities.lower())
    if mode is None:
        raise KeyError(f"Unknown modality combination {modalities!r}")
    task = tasks[task_id]
    dataset = copy.deepcopy(data["datasets"][task["dataset"]])
    # One label table can serve all tasks, but CIFData is task-specific.  A
    # manifest may therefore specify either explicit roots per task or a
    # template using {task_id}, {target_column}, and {safe_target}.
    roots = dataset.pop("cif_data_root_by_task", {})
    template = dataset.pop("cif_data_root_template", None)
    if task_id in roots:
        dataset["cif_data_root"] = roots[task_id]
    elif template:
        dataset["cif_data_root"] = template.format(
            task_id=task_id,
            target_column=task["target_column"],
            safe_target=task["target_column"].replace("/", "_per_"),
        )
    variants = data["point_cloud_variants"]
    result = copy.deepcopy(config)
    result["data"].update(dataset)
    result["data"]["target_column"] = task["target_column"]
    result["experiment"] = {"task_id": task_id, "display_name": task["display_name"], "modalities": mode}
    if mode == "cif_only":
        primary = variants["ed_default"]
        result["fusion"]["type"] = "cif_only"
    elif mode == "cif_ed":
        primary = variants["ed_default"]
    elif mode == "cif_esp":
        primary = variants["esp_default"]
    else:
        primary = variants["esp_default"]
        secondary = variants["ed_default"]
        result["data"].update({f"secondary_{key}": value for key, value in secondary.items() if key in {"point_pack_path", "feature_blocks", "frac_pe_frequencies"}})
    result["data"].update(primary)
    return result
