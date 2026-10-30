from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


PLY_PROPERTY_DTYPES = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    point_path: Path
    target: float


def load_records(config: dict) -> list[SampleRecord]:
    configured_packs = config["data"].get("point_pack_path")
    if configured_packs:
        pack_paths = configured_packs if isinstance(configured_packs, list) else [configured_packs]
        packed_ids: set[str] = set()
        for pack in pack_paths:
            ids_path = Path(pack).with_suffix(".ids.txt")
            with ids_path.open("r", encoding="utf-8") as handle:
                packed_ids.update(line.rstrip("\n") for line in handle if line.strip())
        label_csv = Path(config["data"]["label_csv"])
        target_column = config["data"]["target_column"]
        records = []
        with label_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if target_column not in reader.fieldnames:
                raise KeyError(f"Target column {target_column!r} not found in {label_csv}")
            for row in reader:
                sample_id = row["base"]
                if sample_id not in packed_ids:
                    continue
                raw_target = row[target_column]
                if raw_target in {"", "nan", "NaN", None}:
                    continue
                try:
                    target = float(raw_target)
                except ValueError:
                    continue
                if np.isfinite(target):
                    records.append(SampleRecord(sample_id=sample_id, point_path=Path(pack_paths[0]), target=target))
        if not records:
            raise RuntimeError("No matched samples were found between point packs and labels.")
        return records

    configured_roots = config["data"].get("point_roots")
    if configured_roots is None:
        point_roots = [Path(config["data"]["point_root"])]
    else:
        if not isinstance(configured_roots, list) or not configured_roots:
            raise ValueError("data.point_roots must be a non-empty list")
        point_roots = [Path(root) for root in configured_roots]
    variant = config["data"]["point_cloud_variant"]
    num_points = int(config["data"]["num_points"])
    label_csv = Path(config["data"]["label_csv"])
    target_column = config["data"]["target_column"]
    nonperiodic_dir = config["data"].get("nonperiodic_dirname", "nonperiodic_fps")
    point_file_template = config["data"].get(
        "point_file_template",
        "{variant}_nonperiodic_fps_{num_points}.ply",
    )
    variant_suffix = variant
    if "_" in variant:
        variant_suffix = variant.split("_", 1)[1]

    point_lookup = {}
    for point_root in point_roots:
        variant_root = point_root / variant
        if not variant_root.is_dir():
            raise FileNotFoundError(f"Point-cloud variant directory not found: {variant_root}")
        for folder in variant_root.iterdir():
            if folder.is_dir():
                point_lookup[folder.name.removesuffix("_repeat")] = folder
    if not point_lookup:
        raise RuntimeError(f"No sample folders found under {variant_root}")

    file_name = point_file_template.format(
        variant=variant,
        variant_suffix=variant_suffix,
        num_points=num_points,
    )
    records: list[SampleRecord] = []
    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if target_column not in reader.fieldnames:
            raise KeyError(f"Target column {target_column!r} not found in {label_csv}")
        for row in reader:
            base = row["base"]
            folder = point_lookup.get(base)
            if folder is None:
                continue
            raw_target = row[target_column]
            if raw_target in {"", "nan", "NaN", None}:
                continue
            point_path = folder / nonperiodic_dir / file_name
            if not point_path.is_file():
                continue
            try:
                target = float(raw_target)
            except ValueError:
                continue
            records.append(SampleRecord(sample_id=base, point_path=point_path, target=target))
    if not records:
        raise RuntimeError("No matched samples were found between point clouds and labels.")
    return records


def split_records(records: list[SampleRecord], config: dict) -> dict[str, list[SampleRecord]]:
    split_csv = config["data"].get("split_csv")
    if split_csv:
        split_by_id: dict[str, str] = {}
        with Path(split_csv).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                split_by_id[row["id"]] = row["split"]
        result = {"train": [], "val": [], "test": []}
        for record in records:
            split = split_by_id.get(record.sample_id)
            if split in result:
                result[split].append(record)
        if not all(result.values()):
            raise RuntimeError(f"Fixed split produced an empty partition: {split_csv}")
        return result

    split_cfg = config["data"]["split"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["val_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    shuffled = list(records)
    random.Random(int(config["seed"])).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_records = shuffled[:n_train]
    val_records = shuffled[n_train : n_train + n_val]
    test_records = shuffled[n_train + n_val :]
    return {"train": train_records, "val": val_records, "test": test_records}


class MOFPointCloudDataset(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        feature_blocks: Iterable[str],
        augment: bool = False,
        normalize_xyz: bool = True,
        frac_pe_frequencies: int = 1,
        point_pack_path: str | Path | list[str | Path] | None = None,
    ) -> None:
        self.records = records
        self.feature_blocks = list(feature_blocks)
        self.augment = augment
        self.normalize_xyz = normalize_xyz
        self.frac_pe_frequencies = int(frac_pe_frequencies)
        if isinstance(point_pack_path, list):
            self.point_pack_paths = [Path(path) for path in point_pack_path]
        elif point_pack_path:
            self.point_pack_paths = [Path(point_pack_path)]
        else:
            self.point_pack_paths = []
        self.point_pack_path = self.point_pack_paths[0] if len(self.point_pack_paths) == 1 else None
        self._point_packs: list[np.ndarray | None] = [None] * len(self.point_pack_paths)
        self._point_pack_rows: dict[str, tuple[int, int]] = {}
        for pack_index, pack_path in enumerate(self.point_pack_paths):
            ids_path = pack_path.with_suffix(".ids.txt")
            with ids_path.open("r", encoding="utf-8") as handle:
                for row, sample_id in enumerate(handle):
                    sample_id = sample_id.rstrip("\n")
                    if sample_id in self._point_pack_rows:
                        raise ValueError(f"Duplicate sample ID across point packs: {sample_id}")
                    self._point_pack_rows[sample_id] = (pack_index, row)
        if self.frac_pe_frequencies < 0:
            raise ValueError("frac_pe_frequencies must be non-negative")
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        points = self._load_point_array(record.point_path, record.sample_id).copy()
        if points.shape[1] < 7:
            raise ValueError(
                f"Expected at least 7 point features in {record.point_path}, got shape {points.shape}"
            )

        xyz = points[:, 0:3]
        scalar = points[:, 3:4]
        frac = points[:, 4:7]

        if self.normalize_xyz:
            xyz = normalize_xyz(xyz)
        if self.augment:
            xyz = augment_xyz(xyz)

        blocks = []
        for block in self.feature_blocks:
            if block == "xyz":
                blocks.append(xyz)
            elif block in {"esp", "density", "scalar"}:
                blocks.append(scalar)
            elif block == "frac":
                blocks.append(frac)
            elif block == "frac_pe":
                if self.frac_pe_frequencies < 1:
                    raise ValueError("frac_pe_frequencies must be at least 1 when using frac_pe")
                blocks.append(fourier_encode_frac(frac, self.frac_pe_frequencies))
            else:
                raise ValueError(f"Unsupported feature block: {block}")
        features = np.concatenate(blocks, axis=1).astype(np.float32)
        return {
            "features": torch.from_numpy(features),
            "xyz": torch.from_numpy(xyz.astype(np.float32)),
            "target": torch.tensor(record.target, dtype=torch.float32),
            "sample_id": record.sample_id,
        }

    def _load_point_array(self, path: Path, sample_id: str | None = None) -> np.ndarray:
        if self.point_pack_paths:
            if sample_id is None:
                raise ValueError("sample_id is required when using a point pack")
            pack_index, row = self._point_pack_rows[sample_id]
            if self._point_packs[pack_index] is None:
                self._point_packs[pack_index] = np.load(self.point_pack_paths[pack_index], mmap_mode="r")
            return self._point_packs[pack_index][row]
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        array = read_binary_little_endian_ply(path)
        self._cache[path] = array
        return array


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius > 0:
        centered = centered / radius
    return centered


def fourier_encode_frac(frac: np.ndarray, num_frequencies: int) -> np.ndarray:
    """Encode each fractional axis with sin/cos at frequencies 2**k."""
    frequencies = np.power(2.0, np.arange(num_frequencies, dtype=np.float32))
    angles = frac[:, :, None] * frequencies[None, None, :] * (2.0 * np.pi)
    encoded = np.stack((np.sin(angles), np.cos(angles)), axis=-1)
    return encoded.reshape(frac.shape[0], -1).astype(np.float32, copy=False)


def augment_xyz(xyz: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    xyz = xyz @ rotation.T
    xyz = xyz + np.random.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)
    scale = np.random.uniform(0.95, 1.05)
    return xyz * scale


def read_binary_little_endian_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        fmt = None
        num_vertices = None
        property_names: list[str] = []
        property_types: list[str] = []
        while True:
            line = handle.readline().decode("ascii").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                num_vertices = int(line.split()[2])
            elif line.startswith("property "):
                parts = line.split()
                if len(parts) == 3:
                    property_names.append(parts[2])
                    property_types.append(parts[1])
            elif line == "end_header":
                break

        if fmt != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian PLY is supported, got {fmt!r}")
        if num_vertices is None:
            raise ValueError(f"Vertex count missing in {path}")
        if any(dtype not in PLY_PROPERTY_DTYPES for dtype in property_types):
            unsupported = [dtype for dtype in property_types if dtype not in PLY_PROPERTY_DTYPES]
            raise ValueError(f"Unsupported PLY property types: {unsupported}")

        dtype = np.dtype(
            [
                (name, np.dtype(PLY_PROPERTY_DTYPES[kind]).newbyteorder("<"))
                for name, kind in zip(property_names, property_types)
            ]
        )
        rows = np.fromfile(handle, dtype=dtype, count=num_vertices)
    return np.column_stack([rows[name] for name in property_names]).astype(np.float32, copy=False)
