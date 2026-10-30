#!/usr/bin/env python3
"""Evaluate test-set ED/ESP metrics in value and nearest-atom regions.

Each region uses a deterministic, stratified sample of reference grid points
per test material. Aggregate errors are pooled across all sampled points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


from .manifest import load_manifest

from .predict import (  # noqa: E402
    BOHR_ANGSTROM,
    CachedPeriodicGraphBuilder,
    validate_structure_against_cp3d,
)
from .train import collate_graphs, move_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("ed", "esp"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probes", type=int, default=1000,
                        help="Points sampled per material and region (default: 1000).")
    parser.add_argument("--distance-candidates", type=int, default=50000,
                        help="Uniform candidates/material used to find distance zones.")
    parser.add_argument("--inference-chunk", type=int, default=512,
                        help="Probe points per forward pass; lower values reduce GPU memory.")
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--materials", nargs="*")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    return parser.parse_args()


def positions_from_indices(indices, shape, origin_angstrom, cell):
    ny, nz = shape[1], shape[2]
    i = indices // (ny * nz)
    rem = indices % (ny * nz)
    j = rem // nz
    k = rem % nz
    frac = np.column_stack((i / shape[0], j / shape[1], k / shape[2]))
    return origin_angstrom[None, :] + frac @ cell


def metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not len(target):
        return {"points": 0, "mae": None, "normalized_mae_percent": None,
                "r2": None, "pearson_r": None, "spearman_rho": None}
    from scipy.stats import pearsonr, spearmanr

    error = prediction - target
    sst = np.square(target - target.mean()).sum()
    pearson = pearsonr(target, prediction).statistic if len(target) > 1 else np.nan
    spearman = spearmanr(target, prediction).statistic if len(target) > 1 else np.nan
    return {
        "points": int(len(target)),
        "mae": float(np.abs(error).mean()),
        "normalized_mae_percent": float(100 * np.abs(error).sum() /
                                         max(np.abs(target).sum(), 1e-30)),
        "r2": float(1 - np.square(error).sum() / sst) if sst > 1e-30 else None,
        "pearson_r": float(pearson) if np.isfinite(pearson) else None,
        "spearman_rho": float(spearman) if np.isfinite(spearman) else None,
    }


def sample(pool, count, rng):
    pool = np.asarray(pool, dtype=np.int64)
    if not len(pool):
        return pool
    return rng.choice(pool, size=count, replace=len(pool) < count)


def make_zones(target: str, values: np.ndarray, rng: np.random.Generator,
               probes: int, distance_indices: np.ndarray,
               distance_bins: np.ndarray):
    zones = {"uniform": sample(np.arange(values.size), probes, rng)}
    if target == "ed":
        for threshold in (0.1, 0.08, 0.05, 0.03, 0.02, 0.01):
            pool = np.flatnonzero(values > threshold)
            zones[f"gt_{threshold:g}_e_per_angstrom3"] = sample(pool, probes, rng)
    else:
        for pct in (3, 5, 7):
            lo = np.quantile(values, pct / 100.0)
            hi = np.quantile(values, 1 - pct / 100.0)
            zones[f"top_{pct}pct"] = sample(np.flatnonzero(values >= hi), probes, rng)
            zones[f"bottom_{pct}pct"] = sample(np.flatnonzero(values <= lo), probes, rng)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4)):
        pool = distance_indices[(distance_bins >= left) & (distance_bins < right)]
        zones[f"distance_{left}_{right}_angstrom"] = sample(pool, probes, rng)
    return zones


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, metadata, rows, failures, retained, write_samples=False):
    zone_names = metadata["zone_names"]
    aggregate = {}
    for zone in zone_names:
        pairs = retained.get(zone, [])
        y = np.concatenate([pair[0] for pair in pairs]) if pairs else np.asarray([])
        p = np.concatenate([pair[1] for pair in pairs]) if pairs else np.asarray([])
        base = metrics(y, p)
        aggregate[zone] = base
    payload = {**metadata, "aggregate": aggregate,
               "test_materials_completed": len(rows), "failures": failures,
               "per_material": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    fields = ["region", "points", "mae", "r2", "normalized_mae_percent",
              "pearson_r", "spearman_rho"]
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region, result in aggregate.items():
            writer.writerow({"region": region, **result})
    fields = ["material", "region", *fields[1:]]
    with path.with_name(path.stem + "_per_material.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for region, result in row["regions"].items():
                writer.writerow({"material": row["material"], "region": region, **result})
    if write_samples:
        arrays = {}
        for zone, pairs in retained.items():
            arrays[f"{zone}__target"] = (np.concatenate([pair[0] for pair in pairs])
                                               if pairs else np.asarray([], dtype=np.float32))
            arrays[f"{zone}__prediction"] = (np.concatenate([pair[1] for pair in pairs])
                                                   if pairs else np.asarray([], dtype=np.float32))
        np.savez_compressed(path.with_name(path.stem + "_samples.npz"), **arrays)


def main() -> int:
    args = parse_args()
    if args.probes <= 0 or args.distance_candidates <= 0 or args.inference_chunk <= 0:
        raise ValueError("--probes, --distance-candidates, and --inference-chunk must be positive")

    import ase.io
    import ase.units
    import cp3d
    import torch
    from scipy.spatial import cKDTree
    from src.charge3net.models.e3 import E3DensityModel
    from src.utils.data import supercell_atoms_positions

    manifest = load_manifest(args.manifest)
    by_name = {record["material"]: record for record in manifest["records"]}
    test_names = json.loads(args.split_file.read_text())["test"]
    original_indices = {name: index for index, name in enumerate(test_names)}
    if args.materials:
        missing = sorted(set(args.materials) - set(test_names))
        if missing:
            raise ValueError(f"Materials not in test split: {missing[:10]}")
        test_names = args.materials
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and a valid shard index")
    test_names = test_names[args.shard_index::args.num_shards]
    if args.limit is not None:
        test_names = test_names[:args.limit]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = E3DensityModel(num_interactions=3, num_neighbors=20, mul=500, lmax=4,
                           cutoff=args.cutoff, basis="gaussian", num_basis=20)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    step = int(checkpoint.get("step", -1))
    epoch = int(checkpoint.get("epoch", -1))
    zone_names = (["uniform"] + [f"gt_{v:g}_e_per_angstrom3"
                  for v in (0.1, 0.08, 0.05, 0.03, 0.02, 0.01)] +
                  [f"distance_{a}_{a+1}_angstrom" for a in range(4)]
                  if args.target == "ed" else
                  ["uniform"] + [f"{side}_{pct}pct" for pct in (3, 5, 7)
                                 for side in ("top", "bottom")] +
                  [f"distance_{a}_{a+1}_angstrom" for a in range(4)])
    metadata = {
        "target": args.target,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": calculate_sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "split_file": str(args.split_file.resolve()),
        "test_materials_requested": len(test_names),
        "probes_per_material_per_region": args.probes,
        "distance_candidates_per_material": args.distance_candidates,
        "inference_chunk": args.inference_chunk,
        "cutoff_angstrom": args.cutoff,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "ed_threshold_units": "electron/angstrom^3 (reference ED converted from electron/Bohr^3)",
        "esp_percentile_definition": "per-MOF reference-field percentile; top >= (100-p)th, bottom <= pth",
        "distance_definition": "nearest periodic atom from 3x3x3 atom images; ranges [left,right) angstrom",
        "distance_sampling": "uniform random grid candidates per MOF, conditioned on distance bin",
        "aggregate_definition": "pooled sampled grid points across completed test MOFs",
        "zone_names": zone_names,
    }
    rows, failures = [], []
    retained = {name: [] for name in zone_names}
    started = time.time()
    for mi, material in enumerate(test_names):
        try:
            record = by_name[material]
            field, cp_metadata = cp3d.load(record[args.target], return_metadata=True)
            atoms = ase.io.read(record["cif"])
            atoms.set_pbc(True)
            validate_structure_against_cp3d(atoms, cp_metadata)
            shape = tuple(int(x) for x in field.shape)
            values = np.asarray(field, dtype=np.float32).reshape(-1)
            if args.target == "ed":
                values = values / np.float32(ase.units.Bohr ** 3)
            origin = np.asarray(cp_metadata.get("origin", [0, 0, 0]), dtype=np.float64)
            origin *= BOHR_ANGSTROM

            material_index = original_indices[material]
            rng = np.random.default_rng(args.seed + material_index * 20)
            candidate_indices = rng.choice(values.size,
                                           size=min(args.distance_candidates, values.size),
                                           replace=False)
            candidate_xyz = positions_from_indices(candidate_indices, shape, origin,
                                                   atoms.cell.array)
            atom_xyz, _ = supercell_atoms_positions(atoms)
            tree = cKDTree(atom_xyz)
            nearest = tree.query(candidate_xyz, k=1, workers=-1)[0]
            distance_bins = np.floor(nearest).astype(np.int8)
            zones = make_zones(args.target, values, rng, args.probes,
                               candidate_indices, distance_bins)
            ordered = list(zones)
            all_indices = np.concatenate([zones[z] for z in ordered])
            counts = [len(zones[z]) for z in ordered]
            targets = values[all_indices]

            builder = CachedPeriodicGraphBuilder(atoms, args.cutoff)
            atom_graph = {
                "nodes": torch.as_tensor(atoms.numbers, dtype=torch.long),
                "atom_edges": torch.as_tensor(builder.atom_edges, dtype=torch.long),
                "atom_edges_displacement": torch.as_tensor(builder.atom_displacements,
                                                            dtype=torch.float32),
                "num_nodes": torch.tensor(len(atoms), dtype=torch.long),
                "num_atom_edges": torch.tensor(len(builder.atom_edges), dtype=torch.long),
                "atom_xyz": torch.as_tensor(atoms.positions, dtype=torch.float32),
                "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float32),
                "filename": material,
            }
            prediction_parts = []
            for start in range(0, len(all_indices), args.inference_chunk):
                stop = min(start + args.inference_chunk, len(all_indices))
                positions = positions_from_indices(all_indices[start:stop], shape, origin,
                                                   atoms.cell.array)
                probe_edges, probe_displacements = builder.probe_graph(positions)
                graph = {
                    **atom_graph,
                    "probe_edges": torch.as_tensor(probe_edges, dtype=torch.long),
                    "probe_edges_displacement": torch.as_tensor(probe_displacements,
                                                                dtype=torch.float32),
                    "probe_target": torch.as_tensor(targets[start:stop], dtype=torch.float32),
                    "num_probes": torch.tensor(stop - start, dtype=torch.long),
                    "num_probe_edges": torch.tensor(len(probe_edges), dtype=torch.long),
                    "probe_xyz": torch.as_tensor(positions, dtype=torch.float32),
                }
                batch = move_batch(collate_graphs([graph]), device)
                with torch.no_grad():
                    prediction_parts.append(
                        model(batch).reshape(-1).detach().cpu().numpy()
                    )
                del batch, graph
            prediction = np.concatenate(prediction_parts)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            result = {}
            offset = 0
            for zone, count in zip(ordered, counts):
                target = targets[offset:offset + count].astype(np.float64, copy=False)
                pred = prediction[offset:offset + count].astype(np.float64, copy=False)
                result[zone] = metrics(target, pred)
                retained[zone].append((target.astype(np.float32), pred.astype(np.float32)))
                offset += count
            rows.append({"material": material, "grid_points": int(values.size),
                         "atom_count": len(atoms), "regions": result})
            print(f"[{args.target}] {len(rows)}/{len(test_names)} {material} "
                  f"elapsed={time.time()-started:.1f}s", flush=True)
        except Exception as exc:
            failures.append({"material": material,
                             "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{args.target}] FAILED {material}: {type(exc).__name__}: {exc}",
                  flush=True)
        if (mi + 1) % args.save_every == 0:
            save(args.output, metadata, rows, failures, retained)
    save(args.output, metadata, rows, failures, retained, write_samples=True)
    print(json.dumps({"completed": len(rows), "failures": len(failures),
                      "output": str(args.output), "elapsed_seconds": time.time()-started},
                     indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
