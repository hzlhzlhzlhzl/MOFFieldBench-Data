from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairedRecord:
    sample_id: str
    point_path: Path
    secondary_point_path: Path | None
    cif_index: int
    target: float
    split: str


def add_source_paths(config: dict) -> None:
    point_root = str(Path(config["paths"]["point_project"]).resolve())
    cif_vendor = str(Path(config["paths"]["cif_project"]).resolve() / "vendor")
    for path in (cif_vendor, point_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def sanitize_target(target: str) -> str:
    return target.replace("/", "_per_")


def resolve_cif_data_root(config: dict) -> Path:
    explicit = config["data"].get("cif_data_root")
    if explicit:
        root = Path(explicit)
    else:
        project = Path(config["paths"]["cif_project"])
        target = config["data"]["target_column"]
        safe_target = sanitize_target(target)
        geom_targets = {"ASA", "Df", "Di", "POAVAf", "Density"}
        group = "data_recommended_geom" if target in geom_targets else "data_esp_highest_lowest_3pct_shared"
        root = project / group / safe_target
    if not root.is_dir():
        raise FileNotFoundError(f"CIF dataset directory not found: {root}")
    return root


def build_paired_records(config: dict, cif_dataset) -> tuple[dict[str, list[PairedRecord]], dict]:
    data_cfg = config["data"]
    target_column = data_cfg["target_column"]
    variant = data_cfg["point_cloud_variant"]
    num_points = int(data_cfg["num_points"])
    point_pack = data_cfg.get("point_pack_path")
    secondary_pack = data_cfg.get("secondary_point_pack_path")
    point_lookup = {}
    packed_ids: set[str] = set()
    if point_pack:
        pack_paths = point_pack if isinstance(point_pack, list) else [point_pack]
        for pack_path in pack_paths:
            ids_path = Path(pack_path).with_suffix(".ids.txt")
            with ids_path.open("r", encoding="utf-8") as handle:
                packed_ids.update(line.strip() for line in handle if line.strip())
    else:
        point_root = Path(data_cfg["point_root"]) / variant
        if not point_root.is_dir():
            raise FileNotFoundError(f"Point-cloud variant directory not found: {point_root}")
        point_lookup = {
            folder.name.removesuffix("_repeat"): folder
            for folder in point_root.iterdir()
            if folder.is_dir()
        }
    variant_suffix = variant.split("_", 1)[1] if "_" in variant else variant
    point_name = data_cfg.get(
        "point_file_template", "{variant}_nonperiodic_fps_{num_points}.ply"
    ).format(variant=variant, variant_suffix=variant_suffix, num_points=num_points)
    point_subdir = data_cfg.get("point_subdir", "nonperiodic_fps")

    labels: dict[str, float] = {}
    label_id_column = str(data_cfg.get("label_id_column", "base"))
    label_csv = Path(data_cfg["label_csv"])
    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if target_column not in (reader.fieldnames or []):
            raise KeyError(f"Target {target_column!r} not found in {label_csv}")
        for row in reader:
            try:
                value = float(row[target_column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                labels[row[label_id_column]] = value

    cif_index = {row[0]: index for index, row in enumerate(cif_dataset.id_prop_data)}
    split_path = data_cfg.get("split_csv") or (resolve_cif_data_root(config) / "splits.csv")
    split_frame = pd.read_csv(split_path)
    split_id_column = str(data_cfg.get("split_id_column", "id"))
    split_by_id = dict(zip(split_frame[split_id_column].astype(str), split_frame["split"].astype(str)))

    secondary_ids: set[str] = set()
    if secondary_pack:
        for pack_path in (secondary_pack if isinstance(secondary_pack, list) else [secondary_pack]):
            with Path(pack_path).with_suffix(".ids.txt").open("r", encoding="utf-8") as handle:
                secondary_ids.update(line.strip() for line in handle if line.strip())

    splits: dict[str, list[PairedRecord]] = {"train": [], "val": [], "test": []}
    missing = {"point_folder": 0, "label": 0, "cif": 0}
    for sample_id, split in split_by_id.items():
        if split not in splits:
            continue
        if point_pack:
            if sample_id not in packed_ids:
                missing["point_folder"] += 1
                continue
            point_path = Path(point_pack[0] if isinstance(point_pack, list) else point_pack)
        else:
            folder = point_lookup.get(sample_id)
            if folder is None:
                missing["point_folder"] += 1
                continue
            point_path = folder / point_subdir / point_name
        if sample_id not in labels:
            missing["label"] += 1
            continue
        if sample_id not in cif_index:
            missing["cif"] += 1
            continue
        if secondary_pack and sample_id not in secondary_ids:
            missing["secondary_point"] = missing.get("secondary_point", 0) + 1
            continue
        secondary_path = Path(secondary_pack[0] if isinstance(secondary_pack, list) else secondary_pack) if secondary_pack else None
        splits[split].append(
            PairedRecord(sample_id, point_path, secondary_path, cif_index[sample_id], labels[sample_id], split)
        )

    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise RuntimeError(f"Empty paired split after alignment: { {k: len(v) for k, v in splits.items()} }")
    # Avoid 17k individual stat calls on the network filesystem. Missing PLY files,
    # if any, are reported naturally with their exact path when that sample is loaded.
    for split_records in splits.values():
        if split_records and not split_records[0].point_path.is_file():
            raise FileNotFoundError(f"Representative point file not found: {split_records[0].point_path}")
    report = {
        "cif_split_sizes": split_frame["split"].value_counts().to_dict(),
        "paired_split_sizes": {key: len(value) for key, value in splits.items()},
        "missing": {**missing, "point_file": "checked_lazily_during_loading"},
    }
    return splits, report


class PairedMOFDataset(Dataset):
    def __init__(
        self,
        records: list[PairedRecord],
        cif_dataset,
        feature_blocks: Iterable[str],
        augment: bool,
        normalize_xyz: bool,
        frac_pe_frequencies: int,
        point_pack_path=None,
        secondary_point_pack_path=None,
        secondary_feature_blocks=None,
        secondary_frac_pe_frequencies=None,
    ) -> None:
        from mof_point.data import MOFPointCloudDataset, SampleRecord

        self.records = records
        self.cif_dataset = cif_dataset
        point_records = [SampleRecord(r.sample_id, r.point_path, r.target) for r in records]
        self.point_dataset = MOFPointCloudDataset(
            point_records,
            feature_blocks=feature_blocks,
            augment=augment,
            normalize_xyz=normalize_xyz,
            frac_pe_frequencies=frac_pe_frequencies,
            point_pack_path=point_pack_path,
        )
        self.secondary_point_dataset = None
        if secondary_point_pack_path:
            secondary_records = [SampleRecord(r.sample_id, r.secondary_point_path, r.target) for r in records]
            self.secondary_point_dataset = MOFPointCloudDataset(
                secondary_records,
                feature_blocks=secondary_feature_blocks or feature_blocks,
                augment=augment,
                normalize_xyz=normalize_xyz,
                frac_pe_frequencies=(
                    frac_pe_frequencies
                    if secondary_frac_pe_frequencies is None
                    else secondary_frac_pe_frequencies
                ),
                point_pack_path=secondary_point_pack_path,
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        point = self.point_dataset[index]
        cif = self.cif_dataset[self.records[index].cif_index]
        if cif[2] != self.records[index].sample_id:
            raise RuntimeError(f"CIF/point ID mismatch: {cif[2]} != {self.records[index].sample_id}")
        if self.secondary_point_dataset is None:
            return point, cif
        return point, self.secondary_point_dataset[index], cif


def paired_collate(items):
    from cgcnn.data import collate_pool

    triple = len(items[0]) == 3
    if triple:
        point_items, secondary_items, cif_items = zip(*items)
    else:
        point_items, cif_items = zip(*items)
    point_batch = {
        "features": torch.stack([item["features"] for item in point_items]),
        "xyz": torch.stack([item["xyz"] for item in point_items]),
        "target": torch.stack([item["target"] for item in point_items]),
        "sample_id": [item["sample_id"] for item in point_items],
    }
    cif_batch = collate_pool(list(cif_items))
    if point_batch["sample_id"] != list(cif_batch[2]):
        raise RuntimeError("CIF and point-cloud batches are not aligned")
    if not triple:
        return point_batch, cif_batch[0]
    secondary_batch = {
        "features": torch.stack([item["features"] for item in secondary_items]),
        "xyz": torch.stack([item["xyz"] for item in secondary_items]),
        "target": torch.stack([item["target"] for item in secondary_items]),
        "sample_id": [item["sample_id"] for item in secondary_items],
    }
    if secondary_batch["sample_id"] != point_batch["sample_id"]:
        raise RuntimeError("Primary and secondary point-cloud batches are not aligned")
    return point_batch, secondary_batch, cif_batch[0]


def make_datasets(config: dict):
    add_source_paths(config)
    from cgcnn.data import CIFData

    cif_root = resolve_cif_data_root(config)
    cif_dataset = CIFData(
        str(cif_root),
        max_num_nbr=int(config["cif_model"].get("max_num_nbr", 12)),
        radius=float(config["cif_model"].get("radius", 8.0)),
        random_seed=int(config["seed"]),
    )
    splits, report = build_paired_records(config, cif_dataset)
    data_cfg = config["data"]
    datasets = {
        split: PairedMOFDataset(
            records,
            cif_dataset,
            feature_blocks=data_cfg["feature_blocks"],
            augment=(split == "train" and bool(data_cfg.get("augment", False))),
            normalize_xyz=bool(data_cfg.get("normalize_xyz", True)),
            frac_pe_frequencies=int(data_cfg.get("frac_pe_frequencies", 1)),
            point_pack_path=data_cfg.get("point_pack_path"),
            secondary_point_pack_path=data_cfg.get("secondary_point_pack_path"),
            secondary_feature_blocks=data_cfg.get("secondary_feature_blocks"),
            secondary_frac_pe_frequencies=int(
                data_cfg.get(
                    "secondary_frac_pe_frequencies",
                    data_cfg.get("frac_pe_frequencies", 1),
                )
            ),
        )
        for split, records in splits.items()
    }
    return datasets, report, cif_root
