#!/usr/bin/env python3
"""Predict CP2K-aligned electron-density grids from CIFs with ChargE3Net.

The CP3D reference file is used only for its grid geometry and metadata.  Its
scalar field is never decoded.  CIF atoms are used as the model input.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from . import ROOT
from .manifest import load_manifest
HERE = ROOT
REPO = ROOT / "models/charge3net"
DEFAULT_CUBE_ROOT = ROOT / "data/raw"
DEFAULT_CIF_ROOT = ROOT / "data/raw/cif"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/predictions"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/ed/step_157000.pt"
BOHR_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class MaterialTask:
    material: str
    cif_path: Path
    reference_cp3d: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict electron density from CIF structures using the packaged "
            "MOF ChargE3Net checkpoint."
        )
    )
    parser.add_argument("--cube-root", type=Path, default=DEFAULT_CUBE_ROOT)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_CIF_ROOT)
    parser.add_argument(
        "--paired-manifest", type=Path,
        help="JSON manifest with material, cif, ed, and esp CP3D paths.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--target", choices=("ed", "esp"), default="ed",
        help="Predict electron density (ed) or CP2K Hartree potential (esp).",
    )
    parser.add_argument(
        "--materials",
        nargs="*",
        help="Material directory names to process; default discovers all cp2k_cube entries.",
    )
    parser.add_argument("--limit", type=int, help="Limit discovered materials after sorting.")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--cutoff", type=float, default=4.0, help="Graph cutoff in Angstrom.")
    parser.add_argument(
        "--probe-chunk",
        type=int,
        default=2500,
        help="Probe points predicted per GPU/CPU forward pass.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="CP3D absolute compression tolerance in electron/Bohr^3.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--clamp-nonnegative", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate CIF/reference mappings and units without loading the model.",
    )
    parser.add_argument(
        "--preview-probes",
        type=int,
        help=(
            "Predict only the first N grid points and write a preview NPZ. "
            "This never writes a final CP3D or DONE.json."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def material_to_cif_name(material: str) -> str:
    stem = material[:-7] if material.endswith("_repeat") else material
    return f"{stem}.cif"


def find_reference_cp3d(material_dir: Path, target: str = "ed") -> Path:
    pattern = "*ELECTRON_DENSITY*.cp3d" if target == "ed" else "*v_hartree*.cp3d"
    matches = sorted(material_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{material_dir}: expected exactly one electron-density CP3D, got {len(matches)}"
        )
    return matches[0]


def discover_tasks(
    cube_root: Path,
    cif_root: Path,
    output_root: Path,
    materials: list[str] | None,
    limit: int | None,
    skip_missing: bool = False,
    target: str = "ed",
) -> list[MaterialTask]:
    if materials:
        material_dirs = [cube_root / name for name in materials]
    else:
        material_dirs = sorted(path for path in cube_root.iterdir() if path.is_dir())
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        material_dirs = material_dirs[:limit]

    tasks: list[MaterialTask] = []
    for material_dir in material_dirs:
        if not material_dir.is_dir():
            raise FileNotFoundError(f"Material directory not found: {material_dir}")
        cif_path = cif_root / material_to_cif_name(material_dir.name)
        if not cif_path.is_file():
            if skip_missing:
                continue
            raise FileNotFoundError(
                f"No CIF for {material_dir.name}; expected {cif_path}"
            )
        try:
            reference_cp3d = find_reference_cp3d(material_dir, target)
        except (ValueError, FileNotFoundError):
            if skip_missing:
                continue
            raise
        tasks.append(
            MaterialTask(
                material=material_dir.name,
                cif_path=cif_path,
                reference_cp3d=reference_cp3d,
                output_dir=output_root / material_dir.name,
            )
        )
    return tasks


def discover_manifest_tasks(manifest_path, output_root, materials, limit, target):
    payload = load_manifest(manifest_path)
    by_name = {record["material"]: record for record in payload["records"]}
    names = list(materials) if materials else sorted(by_name)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        names = names[:limit]
    tasks = []
    for name in names:
        if name not in by_name:
            raise KeyError(f"Material not present in paired manifest: {name}")
        record = by_name[name]
        cif_path = Path(record["cif"])
        reference_cp3d = Path(record[target])
        if not cif_path.is_file() or not reference_cp3d.is_file():
            raise FileNotFoundError(
                f"Manifest paths unavailable for {name}: cif={cif_path}, target={reference_cp3d}"
            )
        tasks.append(MaterialTask(
            material=name,
            cif_path=cif_path,
            reference_cp3d=reference_cp3d,
            output_dir=Path(output_root) / name,
        ))
    return tasks


def read_cp3d_metadata(path: Path) -> dict[str, object]:
    """Read only metadata_json.npy; the multi-million-point field stays compressed."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"{path} is not a supported CP3D v1 archive")
        return json.loads(str(archive["metadata_json"]))


def cp3d_cell_bohr(metadata: dict[str, object]) -> np.ndarray:
    shape = np.asarray(metadata["shape"], dtype=np.float64)
    axes = np.asarray(metadata["axes"], dtype=np.float64)
    if shape.shape != (3,) or axes.shape != (3, 3):
        raise ValueError("CP3D metadata must contain shape[3] and axes[3,3]")
    return axes * shape[:, None]


def validate_structure_against_cp3d(atoms, metadata: dict[str, object]) -> dict[str, float]:
    cif_cell = np.asarray(atoms.cell.array, dtype=np.float64)
    reference_cell = cp3d_cell_bohr(metadata) * BOHR_ANGSTROM
    max_cell_error = float(np.max(np.abs(cif_cell - reference_cell)))
    reference_atoms = metadata.get("atoms", [])
    if reference_atoms:
        reference_numbers = sorted(int(row[0]) for row in reference_atoms)
        cif_numbers = sorted(int(value) for value in atoms.numbers)
        if cif_numbers != reference_numbers:
            raise ValueError(
                "CIF and CP3D atom species/counts differ: "
                f"CIF={len(cif_numbers)}, CP3D={len(reference_numbers)}"
            )
    if max_cell_error > 2e-3:
        raise ValueError(
            "CIF and CP3D cells do not align after Bohr->Angstrom conversion; "
            f"max component error={max_cell_error:.6g} Angstrom"
        )
    return {"max_cell_component_error_angstrom": max_cell_error}


def flat_grid_positions(
    start: int,
    stop: int,
    shape: tuple[int, int, int],
    cell_angstrom: np.ndarray,
    origin_angstrom: np.ndarray,
) -> np.ndarray:
    """Return CP2K C-order grid points (third axis fastest) in Angstrom."""
    flat = np.arange(start, stop, dtype=np.int64)
    ny, nz = shape[1], shape[2]
    i = flat // (ny * nz)
    rem = flat % (ny * nz)
    j = rem // nz
    k = rem % nz
    frac = np.column_stack((i / shape[0], j / shape[1], k / shape[2]))
    return origin_angstrom[None, :] + frac @ cell_angstrom


class CachedPeriodicGraphBuilder:
    """ChargE3Net-compatible graph construction with a cached periodic atom KDTree."""

    def __init__(self, atoms, cutoff: float):
        from scipy.spatial import KDTree
        from src.charge3net.data.graph_construction import KdTreeGraphConstructor, _cell_heights

        self.atoms = atoms
        self.cutoff = float(cutoff)
        constructor = KdTreeGraphConstructor(cutoff=self.cutoff, sorted_edges=False)
        atom_edges, atom_disp, _, _ = constructor.atoms_to_graph(atoms)
        self.atom_edges = np.concatenate(atom_edges, axis=0).astype(np.int64)
        self.atom_displacements = np.concatenate(atom_disp, axis=0).astype(np.float32)

        heights = _cell_heights(atoms.get_cell())
        n_rep = np.ceil(self.cutoff / (heights + 1e-12)).astype(int)
        ranges = [np.arange(-value, value + 1) if atoms.pbc[d] else np.array([0])
                  for d, value in enumerate(n_rep)]
        shifts = np.array(
            [(i, j, k) for i in ranges[0] for j in ranges[1] for k in ranges[2]],
            dtype=np.int64,
        )
        positions = np.asarray(atoms.positions, dtype=np.float64)
        repeated = positions[:, None, :] + shifts[None, :, :] @ atoms.cell.array
        self.super_positions = repeated.reshape(-1, 3)
        self.super_atom_indices = np.repeat(
            np.arange(len(atoms), dtype=np.int64), len(shifts)
        )
        self.super_shifts = np.tile(shifts, (len(atoms), 1)).astype(np.float32)
        self.tree = KDTree(self.super_positions)

    def probe_graph(self, probe_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neighborhoods = self.tree.query_ball_point(probe_positions, r=self.cutoff)
        counts = np.fromiter((len(x) for x in neighborhoods), dtype=np.int64)
        total = int(counts.sum())
        if total == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float32)
        super_indices = np.concatenate(neighborhoods).astype(np.int64, copy=False)
        destination = np.repeat(np.arange(len(probe_positions), dtype=np.int64), counts)
        edges = np.column_stack((self.super_atom_indices[super_indices], destination))
        return edges, self.super_shifts[super_indices]


def batched_tensor(array, *, dtype, device):
    import torch

    return torch.as_tensor(array, dtype=dtype, device=device).unsqueeze(0)


def make_atom_batch(builder: CachedPeriodicGraphBuilder, atoms, device) -> dict[str, object]:
    import torch

    return {
        "nodes": batched_tensor(atoms.numbers, dtype=torch.long, device=device),
        "atom_edges": batched_tensor(
            builder.atom_edges, dtype=torch.long, device=device
        ),
        "atom_edges_displacement": batched_tensor(
            builder.atom_displacements, dtype=torch.float32, device=device
        ),
        "num_nodes": torch.tensor([len(atoms)], dtype=torch.long, device=device),
        "num_atom_edges": torch.tensor(
            [len(builder.atom_edges)], dtype=torch.long, device=device
        ),
        "atom_xyz": batched_tensor(
            atoms.positions, dtype=torch.float32, device=device
        ),
        "cell": batched_tensor(
            atoms.cell.array, dtype=torch.float32, device=device
        ),
    }


def add_probe_batch(base: dict[str, object], builder, positions, device):
    import torch

    edges, displacements = builder.probe_graph(positions)
    batch = dict(base)
    batch.update(
        probe_edges=batched_tensor(edges, dtype=torch.long, device=device),
        probe_edges_displacement=batched_tensor(
            displacements, dtype=torch.float32, device=device
        ),
        probe_xyz=batched_tensor(positions, dtype=torch.float32, device=device),
        num_probes=torch.tensor([len(positions)], dtype=torch.long, device=device),
        num_probe_edges=torch.tensor([len(edges)], dtype=torch.long, device=device),
    )
    return batch


def load_model(checkpoint: Path, device, cutoff: float):
    import torch
    from src.charge3net.models.e3 import E3DensityModel

    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=cutoff,
        basis="gaussian",
        num_basis=20,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def resolve_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def predict_task(task: MaterialTask, args, model, device) -> dict[str, object]:
    import ase.io
    import ase.units
    import cp3d
    import torch

    started = time.time()
    metadata = read_cp3d_metadata(task.reference_cp3d)
    atoms = ase.io.read(task.cif_path)
    atoms.set_pbc(True)
    validation = validate_structure_against_cp3d(atoms, metadata)
    shape = tuple(int(value) for value in metadata["shape"])
    total = int(np.prod(shape, dtype=np.int64))
    origin_angstrom = np.asarray(metadata.get("origin", [0, 0, 0]), dtype=np.float64) * BOHR_ANGSTROM
    cell_angstrom = np.asarray(atoms.cell.array, dtype=np.float64)

    task.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"{task.material}-CHARGE3NET_ELECTRON_DENSITY" if args.target == "ed"
            else f"{task.material}-CHARGE3NET_V_HARTREE")
    final_cp3d = task.output_dir / f"{stem}.cube.cp3d"
    done_path = task.output_dir / "DONE.json"
    if done_path.exists() and final_cp3d.exists() and not args.overwrite:
        return {"status": "skipped", "material": task.material, "output": str(final_cp3d)}
    if final_cp3d.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {final_cp3d}")

    builder = CachedPeriodicGraphBuilder(atoms, args.cutoff)
    atom_batch = make_atom_batch(builder, atoms, device)
    with torch.inference_mode():
        atom_representation = model.atom_model(atom_batch)

    predict_total = min(total, args.preview_probes) if args.preview_probes else total
    if predict_total <= 0:
        raise ValueError("--preview-probes must be positive")

    if args.preview_probes:
        output_values = np.empty(predict_total, dtype=np.float32)
        partial_npy = None
    else:
        partial_npy = task.output_dir / f".{stem}.partial.npy"
        if partial_npy.exists() and not args.overwrite:
            raise FileExistsError(
                f"Partial output exists: {partial_npy}; inspect it or rerun with --overwrite"
            )
        output_values = np.lib.format.open_memmap(
            partial_npy, mode="w+", dtype=np.float32, shape=shape
        ).reshape(-1)

    chunks = (predict_total + args.probe_chunk - 1) // args.probe_chunk
    prediction_min = float("inf")
    prediction_max = float("-inf")
    density_sum = 0.0
    for chunk_id, start in enumerate(range(0, predict_total, args.probe_chunk), 1):
        stop = min(start + args.probe_chunk, predict_total)
        positions = flat_grid_positions(
            start, stop, shape, cell_angstrom, origin_angstrom
        )
        batch = add_probe_batch(atom_batch, builder, positions, device)
        with torch.inference_mode():
            prediction = model.probe_model(batch, atom_representation).reshape(-1)
        model_values = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
        if args.target == "ed":
            if args.clamp_nonnegative:
                np.maximum(model_values, 0.0, out=model_values)
            # ChargE3Net uses electron/Angstrom^3; CP2K CP3D uses electron/Bohr^3.
            stored_values = model_values * np.float32(ase.units.Bohr ** 3)
        else:
            # The ESP model was directly trained on CP2K v_hartree values.
            stored_values = model_values
        output_values[start:stop] = stored_values
        prediction_min = min(prediction_min, float(stored_values.min()))
        prediction_max = max(prediction_max, float(stored_values.max()))
        density_sum += float(stored_values.astype(np.float64).sum())
        if chunk_id == 1 or chunk_id == chunks or chunk_id % args.progress_every == 0:
            elapsed = time.time() - started
            rate = stop / max(elapsed, 1e-9)
            eta = (predict_total - stop) / max(rate, 1e-9)
            print(
                f"  {task.material}: {stop:,}/{predict_total:,} probes "
                f"({100*stop/predict_total:.2f}%), ETA {eta/60:.1f} min",
                flush=True,
            )

    if args.preview_probes:
        preview_path = task.output_dir / f"{stem}.preview_{predict_total}.npz"
        np.savez_compressed(
            preview_path,
            flat_indices=np.arange(predict_total, dtype=np.int64),
            values=output_values,
            shape=np.asarray(shape, dtype=np.int64),
        )
        return {
            "status": "preview",
            "material": task.material,
            "probes": predict_total,
            "output": str(preview_path),
        }

    output_values.flush()
    full_values = np.load(partial_npy, mmap_mode="r")
    output_metadata = {
        **metadata,
        "source_file": str(task.cif_path),
        "prediction": {
            "method": "ChargE3Net",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "reference_grid": str(task.reference_cp3d),
            "model_input_structure": str(task.cif_path),
            "model_output_unit": ("electron/Angstrom^3" if args.target == "ed" else "Hartree/e"),
            "stored_unit": ("electron/Bohr^3" if args.target == "ed" else "Hartree/e"),
            "target": args.target,
            "cutoff_angstrom": args.cutoff,
            "probe_chunk": args.probe_chunk,
            "clamp_nonnegative": args.clamp_nonnegative,
        },
    }
    save_info = cp3d.save(
        full_values,
        final_cp3d,
        metadata=output_metadata,
        tolerance=args.tolerance,
        quantity=("electron_density" if args.target == "ed" else "hartree_potential"),
        unit=("electron/Bohr^3" if args.target == "ed" else "Hartree/e"),
        verify=True,
    )
    voxel_volume_bohr3 = abs(float(np.linalg.det(cp3d_cell_bohr(metadata)))) / total
    report = {
        "status": "complete",
        "material": task.material,
        "cif": str(task.cif_path),
        "reference_cp3d": str(task.reference_cp3d),
        "output": str(final_cp3d),
        "shape": list(shape),
        "grid_points": total,
        "density_min_e_per_bohr3": prediction_min,
        "density_max_e_per_bohr3": prediction_max,
        "integrated_electrons": (density_sum * voxel_volume_bohr3 if args.target == "ed" else None),
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "structure_validation": validation,
        "cp3d": save_info,
    }
    atomic_json_write(done_path, report)
    Path(partial_npy).unlink()
    return report


def dry_run_task(task: MaterialTask) -> dict[str, object]:
    import ase.io

    metadata = read_cp3d_metadata(task.reference_cp3d)
    atoms = ase.io.read(task.cif_path)
    atoms.set_pbc(True)
    validation = validate_structure_against_cp3d(atoms, metadata)
    return {
        "material": task.material,
        "cif": str(task.cif_path),
        "reference_cp3d": str(task.reference_cp3d),
        "shape": metadata["shape"],
        "grid_points": int(np.prod(metadata["shape"], dtype=np.int64)),
        "atoms": len(atoms),
        **validation,
    }


def main() -> int:
    args = parse_args()
    if args.probe_chunk <= 0 or args.progress_every <= 0:
        raise ValueError("--probe-chunk and --progress-every must be positive")
    if args.preview_probes is not None and args.preview_probes <= 0:
        raise ValueError("--preview-probes must be positive")

    # Official code uses imports rooted at charge3net/.
    sys.path.insert(0, str(REPO))
    if args.paired_manifest:
        tasks = discover_manifest_tasks(
            args.paired_manifest.resolve(), args.output_root.resolve(), args.materials,
            args.limit, args.target,
        )
    else:
        tasks = discover_tasks(
            args.cube_root.resolve(),
            args.cif_root.resolve(),
            args.output_root.resolve(),
            args.materials,
            args.limit,
            False,
            args.target,
        )
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require --num-shards > 0 and 0 <= --shard-index < --num-shards")
    tasks = tasks[args.shard_index :: args.num_shards]
    print(f"Discovered {len(tasks)} material(s)", flush=True)

    if args.dry_run:
        reports = []
        for i, task in enumerate(tasks, 1):
            report = dry_run_task(task)
            reports.append(report)
            print(
                f"[{i}/{len(tasks)}] {task.material}: shape={report['shape']} "
                f"points={report['grid_points']:,} atoms={report['atoms']}",
                flush=True,
            )
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json_write(args.output_root / "dry_run_report.json", {"materials": reports})
        return 0

    import torch

    device = resolve_device(args.device)
    print(f"Loading checkpoint on {device}: {args.checkpoint}", flush=True)
    model = load_model(args.checkpoint.resolve(), device, args.cutoff)
    failures = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.material}", flush=True)
        try:
            report = predict_task(task, args, model, device)
            print(f"  status={report['status']} output={report.get('output')}", flush=True)
        except Exception as exc:
            failures.append({"material": task.material, "error": repr(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            if len(tasks) == 1:
                raise
    if failures:
        atomic_json_write(args.output_root / "failed_jobs.json", {"failures": failures})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
