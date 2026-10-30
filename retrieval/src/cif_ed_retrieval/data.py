from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset

ATOM_INIT_PATH = Path(__file__).resolve().with_name("atom_init.json")

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
class PairRecord:
    sample_id: str
    cif_path: Path
    point_path: Path
    split: str


@dataclass(frozen=True)
class PointPairRecord:
    sample_id: str
    ed_path: Path
    esp_path: Path
    split: str


class GaussianDistance:
    def __init__(self, dmin: float, dmax: float, step: float, var: float | None = None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var or step

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomCustomJSONInitializer:
    def __init__(self, elem_embedding_file: str | Path):
        with Path(elem_embedding_file).open("r", encoding="utf-8") as handle:
            elem_embedding = json.load(handle)
        self.embedding = {
            int(key): np.asarray(value, dtype=float) for key, value in elem_embedding.items()
        }

    def get_atom_fea(self, atom_type: int) -> np.ndarray:
        return self.embedding[atom_type]


def build_index(config: dict) -> list[PairRecord]:
    data_cfg = config["data"]
    index_csv = data_cfg.get("index_csv")
    if index_csv:
        records: list[PairRecord] = []
        with Path(index_csv).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(PairRecord(
                    sample_id=row["sample_id"],
                    cif_path=Path(row["cif_path"]),
                    point_path=Path(row["point_path"]),
                    split=row["split"],
                ))
        if not records:
            raise RuntimeError(f"No records found in data.index_csv: {index_csv}")
        return records
    cif_roots = _paths(data_cfg, "cif_roots", "cif_root")
    point_root_value = data_cfg.get("point_roots", data_cfg.get("point_root"))
    point_variant_value = config["data"].get("point_variant")
    if point_root_value is None:
        point_root_value = data_cfg.get("ed_roots", data_cfg.get("ed_root"))
    if point_variant_value is None:
        point_variant_value = config["data"]["density_dir"]
    point_roots = [Path(x) for x in point_root_value] if isinstance(point_root_value, list) else [Path(point_root_value)]
    point_variant = str(point_variant_value)
    num_points = int(config["data"]["num_points"])
    seed = int(config["seed"])
    fps_dirname = str(config["data"].get("fps_dirname", "nonperiodic_fps"))
    default_template = "{variant}_{fps_dirname}_{num_points}.ply"
    if point_variant.startswith("density_"):
        default_template = "density_gt_{variant_suffix}_{fps_dirname}_{num_points}.ply"
    point_file_template = str(config["data"].get("point_file_template", default_template))

    cif_lookup = _cif_lookup(cif_roots)
    point_name = point_file_template.format(
        variant=point_variant,
        variant_suffix=_variant_suffix(point_variant),
        fps_dirname=fps_dirname,
        num_points=num_points,
    )
    pairs_by_id: dict[str, tuple[Path, Path]] = {}
    for point_root in point_roots:
        variant_root = point_root / point_variant
        if not variant_root.is_dir() and (point_root / "non_periodic" / point_variant).is_dir():
            variant_root = point_root / "non_periodic" / point_variant
        for folder in sorted(variant_root.iterdir()):
            if not folder.is_dir():
                continue
            sample_id = folder.name.removesuffix("_repeat")
            cif_path = cif_lookup.get(sample_id)
            point_path = folder / fps_dirname / point_name
            if cif_path is not None and point_path.is_file():
                pairs_by_id.setdefault(sample_id, (cif_path, point_path))
    pairs = [(sample_id, *paths) for sample_id, paths in sorted(pairs_by_id.items())]
    if not pairs:
        raise RuntimeError("No matched CIF/point-cloud pairs were found.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_ratio = float(config["data"]["split"]["train_ratio"])
    val_ratio = float(config["data"]["split"]["val_ratio"])
    test_ratio = float(config["data"]["split"]["test_ratio"])
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    records: list[PairRecord] = []
    for idx, (sample_id, cif_path, point_path) in enumerate(pairs):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"
        records.append(PairRecord(sample_id=sample_id, cif_path=cif_path, point_path=point_path, split=split))
    return records


def build_point_pair_index(config: dict) -> list[PointPairRecord]:
    """Build a deterministic ED/ESP index using only shared sample IDs."""
    data_cfg = config["data"]
    index_csv = data_cfg.get("index_csv")
    if index_csv:
        records: list[PointPairRecord] = []
        with Path(index_csv).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(PointPairRecord(
                    row["sample_id"], Path(row["ed_path"]), Path(row["esp_path"]), row["split"]
                ))
        if not records:
            raise RuntimeError(f"No records found in data.index_csv: {index_csv}")
        return records
    ed_roots = _paths(data_cfg, "ed_roots", "ed_root")
    ed_variant = str(data_cfg.get("density_dir", "density_0.03"))
    esp_value = data_cfg.get("esp_roots", data_cfg.get("esp_root", data_cfg.get("point_root", data_cfg.get("ed_root"))))
    esp_roots = [Path(x) for x in esp_value] if isinstance(esp_value, list) else [Path(esp_value)]
    esp_variant = str(data_cfg.get("point_variant", "esp_highest_lowest_3pct"))
    num_points = int(data_cfg["num_points"])
    fps_dirname = str(data_cfg.get("fps_dirname", "nonperiodic_fps"))

    ed_template = str(data_cfg.get("ed_point_file_template", "density_gt_{variant_suffix}_{fps_dirname}_{num_points}.ply"))
    esp_template = str(data_cfg.get("esp_point_file_template", "{variant}_{fps_dirname}_{num_points}.ply"))
    ed_lookup = _merged_point_lookup(ed_roots, ed_variant, ed_template, fps_dirname, num_points)
    esp_lookup = _merged_point_lookup(esp_roots, esp_variant, esp_template, fps_dirname, num_points)
    shared_ids = sorted(set(ed_lookup).intersection(esp_lookup))
    if not shared_ids:
        raise RuntimeError("No matched ED/ESP point-cloud pairs were found.")

    pairs = [(sample_id, ed_lookup[sample_id], esp_lookup[sample_id]) for sample_id in shared_ids]
    random.Random(int(config["seed"])).shuffle(pairs)
    split_cfg = data_cfg["split"]
    ratios = [float(split_cfg["train_ratio"]), float(split_cfg["val_ratio"]), float(split_cfg["test_ratio"])]
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
    n_train = int(len(pairs) * ratios[0])
    n_val = int(len(pairs) * ratios[1])
    records: list[PointPairRecord] = []
    for idx, (sample_id, ed_path, esp_path) in enumerate(pairs):
        split = "train" if idx < n_train else "val" if idx < n_train + n_val else "test"
        records.append(PointPairRecord(sample_id, ed_path, esp_path, split))
    return records


def _point_lookup(
    point_root: Path,
    variant: str,
    template: str,
    fps_dirname: str,
    num_points: int,
) -> dict[str, Path]:
    if (point_root / "non_periodic").is_dir():
        point_root = point_root / "non_periodic"
    variant_root = point_root / variant
    if not variant_root.is_dir():
        raise FileNotFoundError(f"Point-cloud variant directory not found: {variant_root}")
    point_name = template.format(
        variant=variant,
        variant_suffix=_variant_suffix(variant),
        fps_dirname=fps_dirname,
        num_points=num_points,
    )
    result: dict[str, Path] = {}
    for folder in sorted(variant_root.iterdir()):
        if not folder.is_dir():
            continue
        sample_id = folder.name.removesuffix("_repeat")
        path = folder / fps_dirname / point_name
        if path.is_file():
            result[sample_id] = path
    return result


def _merged_point_lookup(roots, variant, template, fps_dirname, num_points):
    merged = {}
    for root in roots:
        merged.update({k: v for k, v in _point_lookup(root, variant, template, fps_dirname, num_points).items() if k not in merged})
    return merged


def _paths(data_cfg, plural, singular):
    value = data_cfg.get(plural, data_cfg.get(singular))
    if value is None:
        raise KeyError(f"Missing data.{plural} or data.{singular}")
    return [Path(x) for x in value] if isinstance(value, list) else [Path(value)]


def _cif_lookup(roots):
    result = {}
    for root in roots:
        for path in root.glob("*.cif"):
            result.setdefault(path.stem, path)
    return result


def save_index(records: list[PairRecord], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "cif_path", "point_path", "split"])
        for record in records:
            writer.writerow([record.sample_id, str(record.cif_path), str(record.point_path), record.split])


class CIFPointCloudDataset(Dataset):
    def __init__(
        self,
        records: list[PairRecord],
        max_num_nbr: int = 12,
        radius: float = 8.0,
        dmin: float = 0.0,
        step: float = 0.2,
        feature_blocks: Iterable[str] = ("xyz", "scalar", "frac"),
        frac_pe_frequencies: int = 1,
        normalize_xyz: bool = True,
        augment: bool = False,
        cache_data: bool = True,
        cif_cache_dir: str | Path | None = None,
        point_pack_path: str | Path | None = None,
    ) -> None:
        self.records = records
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.feature_blocks = list(feature_blocks)
        self.frac_pe_frequencies = int(frac_pe_frequencies)
        if self.frac_pe_frequencies < 1:
            raise ValueError("frac_pe_frequencies must be at least 1")
        self.normalize_xyz = normalize_xyz
        self.augment = augment
        self.cache_data = cache_data
        self.cif_cache_dir = Path(cif_cache_dir) if cif_cache_dir else None
        if self.cif_cache_dir is not None:
            self.cif_cache_dir.mkdir(parents=True, exist_ok=True)
        self.point_pack_path = Path(point_pack_path) if point_pack_path else None
        self._point_pack: np.ndarray | None = None
        self._point_pack_rows: dict[str, int] = {}
        if self.point_pack_path is not None:
            ids_path = self.point_pack_path.with_suffix(".ids.txt")
            with ids_path.open("r", encoding="utf-8") as handle:
                self._point_pack_rows = {
                    sample_id.rstrip("\n"): row for row, sample_id in enumerate(handle)
                }
        self.atom_init = AtomCustomJSONInitializer(ATOM_INIT_PATH)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)
        self._cif_cache: dict[Path, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._point_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        atom_fea, nbr_fea, nbr_fea_idx = self._load_cif(record.cif_path)
        xyz, scalar, frac = self._load_point_cloud(record.point_path, record.sample_id)
        xyz = xyz.copy()
        if self.normalize_xyz:
            xyz = normalize_xyz(xyz)
        if self.augment:
            xyz = augment_xyz(xyz)
        blocks: list[np.ndarray] = []
        for block in self.feature_blocks:
            if block == "xyz":
                blocks.append(xyz)
            elif block in {"esp", "density", "scalar"}:
                blocks.append(scalar)
            elif block == "frac":
                blocks.append(frac)
            elif block == "frac_pe":
                blocks.append(fourier_encode_frac(frac, self.frac_pe_frequencies))
            else:
                raise ValueError(f"Unsupported feature block: {block}")
        if not blocks:
            raise ValueError("data.feature_blocks must contain at least one feature block")
        features = np.concatenate(blocks, axis=1).astype(np.float32)
        return {
            "sample_id": record.sample_id,
            "atom_fea": atom_fea,
            "nbr_fea": nbr_fea,
            "nbr_fea_idx": nbr_fea_idx,
            "xyz": torch.from_numpy(xyz.astype(np.float32)),
            "point_features": torch.from_numpy(features.astype(np.float32)),
        }

    def _load_cif(self, cif_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._cif_cache.get(cif_path) if self.cache_data else None
        if cached is not None:
            return cached

        disk_cache = self._cif_cache_path(cif_path)
        if disk_cache is not None and disk_cache.is_file():
            try:
                result = torch.load(disk_cache, map_location="cpu")
                if self.cache_data:
                    self._cif_cache[cif_path] = result
                return result
            except (EOFError, OSError, RuntimeError):
                disk_cache.unlink(missing_ok=True)

        crystal = Structure.from_file(cif_path)
        atom_fea = np.vstack(
            [self.atom_init.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))]
        )
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx: list[list[int]] = []
        nbr_fea: list[list[float]] = []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_path.name} does not have enough neighbors within radius {self.radius}.",
                    stacklevel=2,
                )
                nbr_fea_idx.append([item[2] for item in nbr] + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append([item[1] for item in nbr] + [self.radius + 1.0] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append([item[2] for item in nbr[: self.max_num_nbr]])
                nbr_fea.append([item[1] for item in nbr[: self.max_num_nbr]])

        nbr_fea_np = self.gdf.expand(np.asarray(nbr_fea, dtype=float))
        result = (
            torch.tensor(atom_fea, dtype=torch.float32),
            torch.tensor(nbr_fea_np, dtype=torch.float32),
            torch.tensor(np.asarray(nbr_fea_idx), dtype=torch.long),
        )
        if self.cache_data:
            self._cif_cache[cif_path] = result
        if disk_cache is not None:
            disk_cache.parent.mkdir(parents=True, exist_ok=True)
            temp_path = disk_cache.with_name(f".{disk_cache.name}.{os.getpid()}.tmp")
            torch.save(result, temp_path)
            os.replace(temp_path, disk_cache)
        return result

    def _cif_cache_path(self, cif_path: Path) -> Path | None:
        if self.cif_cache_dir is None:
            return None
        cache_key = "|".join(
            (
                str(cif_path.resolve()),
                str(self.max_num_nbr),
                str(self.radius),
                str(self.gdf.filter[0]),
                str(self.gdf.var),
                str(len(self.gdf.filter)),
            )
        )
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return self.cif_cache_dir / digest[:2] / f"{digest}.pt"

    def _load_point_cloud(
        self, point_path: Path, sample_id: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.point_pack_path is not None:
            if sample_id is None:
                raise ValueError("sample_id is required when using a point pack")
            if self._point_pack is None:
                self._point_pack = np.load(self.point_pack_path, mmap_mode="r")
            row = self._point_pack_rows[sample_id]
            packed = self._point_pack[row]
            return packed[:, :3], packed[:, 3:4], packed[:, 4:7]
        cached = self._point_cache.get(point_path) if self.cache_data else None
        if cached is not None:
            return cached

        vertex = read_binary_little_endian_ply(point_path)
        xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        scalar_name = "density" if "density" in vertex.dtype.names else "esp"
        scalar = np.asarray(vertex[scalar_name], dtype=np.float32)[:, None]
        frac_names = ("frac_x", "frac_y", "frac_z")
        missing = [name for name in frac_names if name not in vertex.dtype.names]
        if missing:
            raise ValueError(f"Missing fractional-coordinate PLY properties {missing} in {point_path}")
        frac = np.stack([vertex[name] for name in frac_names], axis=1).astype(np.float32)
        if self.cache_data:
            self._point_cache[point_path] = (xyz, scalar, frac)
        return xyz, scalar, frac


class PointCloudPairDataset(Dataset):
    """Dataset for paired ED and ESP point clouds from the same structure."""

    def __init__(
        self,
        records: list[PointPairRecord],
        feature_blocks: Iterable[str],
        scalar_name: str | None = None,
        frac_pe_frequencies: int = 1,
        normalize_xyz: bool = True,
        augment: bool = False,
        cache_data: bool = False,
        ed_point_pack_path: str | Path | None = None,
        esp_point_pack_path: str | Path | None = None,
    ) -> None:
        self.records = records
        self.feature_blocks = list(feature_blocks)
        self.scalar_name = scalar_name
        self.frac_pe_frequencies = int(frac_pe_frequencies)
        self.normalize_xyz = normalize_xyz
        self.augment = augment
        self.cache_data = cache_data
        self._cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._pack_paths = {
            "ed": Path(ed_point_pack_path) if ed_point_pack_path else None,
            "esp": Path(esp_point_pack_path) if esp_point_pack_path else None,
        }
        self._packs: dict[str, np.ndarray] = {}
        self._pack_rows: dict[str, dict[str, int]] = {}
        for modality, pack_path in self._pack_paths.items():
            if pack_path is not None:
                with pack_path.with_suffix(".ids.txt").open("r", encoding="utf-8") as handle:
                    self._pack_rows[modality] = {
                        sample_id.rstrip("\n"): row for row, sample_id in enumerate(handle)
                    }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        ed_xyz, ed_features = self._load_features(record.ed_path, record.sample_id, "ed")
        esp_xyz, esp_features = self._load_features(record.esp_path, record.sample_id, "esp")
        return {
            "sample_id": record.sample_id,
            "ed_xyz": torch.from_numpy(ed_xyz),
            "ed_features": torch.from_numpy(ed_features),
            "esp_xyz": torch.from_numpy(esp_xyz),
            "esp_features": torch.from_numpy(esp_features),
        }

    def _load_features(self, path: Path, sample_id: str, modality: str) -> tuple[np.ndarray, np.ndarray]:
        pack_path = self._pack_paths[modality]
        if pack_path is not None:
            if modality not in self._packs:
                self._packs[modality] = np.load(pack_path, mmap_mode="r")
            packed = self._packs[modality][self._pack_rows[modality][sample_id]]
            xyz, scalar, frac = packed[:, :3], packed[:, 3:4], packed[:, 4:7]
            return self._prepare_features(xyz, scalar, frac)
        cached = self._cache.get(path) if self.cache_data else None
        if cached is None:
            vertex = read_binary_little_endian_ply(path)
            xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
            point_scalar_name = self.scalar_name or ("density" if "density" in vertex.dtype.names else "esp")
            scalar = np.asarray(vertex[point_scalar_name], dtype=np.float32)[:, None]
            frac = np.stack([vertex[name] for name in ("frac_x", "frac_y", "frac_z")], axis=1).astype(np.float32)
            cached = (xyz, scalar, frac)
            if self.cache_data:
                self._cache[path] = cached
        return self._prepare_features(*cached)

    def _prepare_features(self, xyz, scalar, frac) -> tuple[np.ndarray, np.ndarray]:
        xyz = xyz.copy()
        if self.normalize_xyz:
            xyz = normalize_xyz(xyz)
        if self.augment:
            xyz = augment_xyz(xyz)
        blocks: list[np.ndarray] = []
        for block in self.feature_blocks:
            if block in {"esp", "density", "scalar"}:
                blocks.append(scalar)
            elif block == "frac":
                blocks.append(frac)
            elif block == "frac_pe":
                blocks.append(fourier_encode_frac(frac, self.frac_pe_frequencies))
            elif block == "xyz":
                blocks.append(xyz)
            else:
                raise ValueError(f"Unsupported feature block: {block}")
        if not blocks:
            raise ValueError("feature_blocks must contain at least one feature block")
        return xyz.astype(np.float32), np.concatenate(blocks, axis=1).astype(np.float32)


def collate_cif_point_cloud(batch: list[dict[str, object]]) -> dict[str, object]:
    batch_atom_fea: list[torch.Tensor] = []
    batch_nbr_fea: list[torch.Tensor] = []
    batch_nbr_fea_idx: list[torch.Tensor] = []
    crystal_atom_idx: list[torch.Tensor] = []
    xyz_list: list[torch.Tensor] = []
    point_feature_list: list[torch.Tensor] = []
    sample_ids: list[str] = []
    base_idx = 0

    for item in batch:
        atom_fea = item["atom_fea"]
        nbr_fea = item["nbr_fea"]
        nbr_fea_idx = item["nbr_fea_idx"]
        xyz = item["xyz"]
        point_features = item["point_features"]
        assert isinstance(atom_fea, torch.Tensor)
        assert isinstance(nbr_fea, torch.Tensor)
        assert isinstance(nbr_fea_idx, torch.Tensor)
        assert isinstance(xyz, torch.Tensor)
        assert isinstance(point_features, torch.Tensor)
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.arange(atom_fea.shape[0], dtype=torch.long) + base_idx)
        xyz_list.append(xyz)
        point_feature_list.append(point_features)
        sample_ids.append(str(item["sample_id"]))
        base_idx += atom_fea.shape[0]

    return {
        "sample_ids": sample_ids,
        "cif": (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ),
        "point_xyz": torch.stack(xyz_list, dim=0),
        "point_features": torch.stack(point_feature_list, dim=0),
    }


def collate_point_cloud_pair(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "ed_xyz": torch.stack([item["ed_xyz"] for item in batch]),
        "ed_features": torch.stack([item["ed_features"] for item in batch]),
        "esp_xyz": torch.stack([item["esp_xyz"] for item in batch]),
        "esp_features": torch.stack([item["esp_features"] for item in batch]),
    }


CIFEDDataset = CIFPointCloudDataset
collate_cif_ed = collate_cif_point_cloud


def split_records(records: list[PairRecord]) -> dict[str, list[PairRecord]]:
    result = {"train": [], "val": [], "test": []}
    for record in records:
        result[record.split].append(record)
    return result


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius > 0:
        centered = centered / radius
    return centered


def fourier_encode_frac(frac: np.ndarray, num_frequencies: int) -> np.ndarray:
    """Match the fractional-coordinate encoding supported by test_point."""
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


def _variant_suffix(variant: str) -> str:
    if "_" not in variant:
        return variant
    return variant.split("_", 1)[1]


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
        unsupported = [dtype for dtype in property_types if dtype not in PLY_PROPERTY_DTYPES]
        if unsupported:
            raise ValueError(f"Unsupported PLY property types: {unsupported}")

        dtype = np.dtype(
            [
                (name, np.dtype(PLY_PROPERTY_DTYPES[kind]).newbyteorder("<"))
                for name, kind in zip(property_names, property_types)
            ]
        )
        rows = np.fromfile(handle, dtype=dtype, count=num_vertices)
    return rows
