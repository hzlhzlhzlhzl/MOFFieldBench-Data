#!/usr/bin/env python3
"""Train ChargE3Net on CIF structures supervised by CP2K CP3D densities."""

from __future__ import annotations

import argparse
from functools import lru_cache
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np


from . import ROOT
from .manifest import load_manifest
HERE = ROOT

from .predict import (  # noqa: E402
    BOHR_ANGSTROM,
    DEFAULT_CIF_ROOT,
    DEFAULT_CUBE_ROOT,
    CachedPeriodicGraphBuilder,
    MaterialTask,
    discover_tasks,
    flat_grid_positions,
    material_to_cif_name,
    read_cp3d_metadata,
    validate_structure_against_cp3d,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-root", type=Path, default=DEFAULT_CUBE_ROOT)
    parser.add_argument("--cif-root", type=Path, default=DEFAULT_CIF_ROOT)
    parser.add_argument(
        "--paired-manifest", type=Path,
        help="JSON manifest with records containing material, cif, ed, and esp CP3D paths.",
    )
    parser.add_argument(
        "--trust-paired-manifest", action="store_true",
        help=(
            "Skip eager existence checks for every paired-manifest entry. Use only for a "
            "previously validated manifest; individual files are still opened by the dataset."
        ),
    )
    parser.add_argument(
        "--target", choices=("ed", "esp"), default="ed",
        help="Train electron density (ed) or CP2K Hartree electrostatic potential (esp).",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/training")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--split-strategy", choices=("random", "family"), default="random",
        help="family keeps all .sym.N variants of a base material in one split.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--train-probes", type=int, default=500)
    parser.add_argument("--val-probes", type=int, default=1000)
    parser.add_argument("--test-probes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1_000_000)
    parser.add_argument("--val-batches", type=int, default=0,
                        help="Validation batches per epoch; 0 evaluates the complete set.")
    parser.add_argument("--test-batches", type=int, default=0,
                        help="Test batches per epoch; 0 evaluates the complete set.")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--scheduler-alpha", type=float, default=0.96)
    parser.add_argument("--scheduler-beta", type=float, default=3000.0)
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights only (e.g. charge3net_mp.pt); optimizer starts fresh.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, scheduler, and step from a training checkpoint.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Force a tiny 2-step run.")
    parser.add_argument("--prepare-split-only", action="store_true",
                        help="Create/validate the split file and exit before model construction.")
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@lru_cache(maxsize=1)
def public_material_families():
    return json.loads((ROOT / "data/manifests/material_families.json").read_text())


def material_family(name: str) -> str:
    return public_material_families().get(name, name)


def make_or_load_split(tasks, path: Path, seed: int, train_fraction: float,
                       val_fraction: float, strategy: str = "random"):
    names = [task.material for task in tasks]
    if path.is_file():
        split = json.loads(path.read_text())
        listed = set(split["train"] + split["validation"] + split["test"])
        missing = listed - set(names)
        if missing:
            raise ValueError(f"Split references {len(missing)} unavailable materials")
        return split
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Training and validation fractions must be in (0,1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Training + validation fractions must be < 1")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        shuffled = np.asarray(names, dtype=object)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_fraction))
        n_val = max(1, int(len(shuffled) * val_fraction))
        if len(shuffled) >= 3 and n_train + n_val >= len(shuffled):
            n_train = len(shuffled) - n_val - 1
        parts = [shuffled[:n_train].tolist(),
                 shuffled[n_train:n_train + n_val].tolist(),
                 shuffled[n_train + n_val:].tolist()]
    else:
        groups = {}
        for name in names:
            groups.setdefault(material_family(name), []).append(name)
        grouped = list(groups.items())
        rng.shuffle(grouped)
        targets = np.asarray(
            [train_fraction, val_fraction, 1.0 - train_fraction - val_fraction]
        ) * len(names)
        parts = [[], [], []]
        counts = np.zeros(3, dtype=np.int64)
        # Assign whole families to the split furthest below its target fraction.
        for _, members in grouped:
            destination = int(np.argmin(counts / np.maximum(targets, 1.0)))
            parts[destination].extend(members)
            counts[destination] += len(members)
        for part in parts:
            part.sort()
    split = {
        "train": parts[0], "validation": parts[1], "test": parts[2],
        "seed": seed, "strategy": strategy,
        "family_definition": "public material_families.json" if strategy == "family" else None,
    }
    atomic_json(path, split)
    return split


def discover_training_tasks(args):
    if args.paired_manifest:
        payload = load_manifest(args.paired_manifest)
        records = payload.get("records", [])
        if args.limit is not None:
            records = records[:args.limit]
        tasks = []
        for record in records:
            cif_path = Path(record["cif"])
            target_path = Path(record[args.target])
            if (not args.trust_paired_manifest
                    and (not cif_path.is_file() or not target_path.is_file())):
                raise FileNotFoundError(
                    f"Manifest entry {record.get('material')} is unavailable: "
                    f"cif={cif_path} target={target_path}"
                )
            tasks.append(MaterialTask(
                material=record["material"],
                cif_path=cif_path,
                reference_cp3d=target_path,
                output_dir=args.run_dir / record["material"],
            ))
        return tasks
    if args.target == "ed":
        return discover_tasks(
            args.cube_root.resolve(), args.cif_root.resolve(), args.run_dir, None, args.limit,
            True,
        )

    material_dirs = sorted(path for path in args.cube_root.resolve().iterdir() if path.is_dir())
    if args.limit is not None:
        material_dirs = material_dirs[:args.limit]
    tasks = []
    for material_dir in material_dirs:
        cif_path = args.cif_root.resolve() / material_to_cif_name(material_dir.name)
        matches = sorted(material_dir.glob("*v_hartree*.cp3d"))
        if not cif_path.is_file() or len(matches) != 1:
            continue
        tasks.append(MaterialTask(
            material=material_dir.name,
            cif_path=cif_path,
            reference_cp3d=matches[0],
            output_dir=args.run_dir / material_dir.name,
        ))
    return tasks


class CP3DProbeDataset:
    def __init__(self, tasks, probes: int, cutoff: float, deterministic: bool, seed: int,
                 target: str = "ed"):
        import torch

        self.tasks = list(tasks)
        self.probes = int(probes)
        self.cutoff = float(cutoff)
        self.deterministic = deterministic
        self.seed = int(seed)
        self.target = target
        if self.probes <= 0:
            raise ValueError("Probe count must be positive")
        self._dataset_base = torch.utils.data.Dataset

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, index):
        import ase.io
        import ase.units
        import cp3d
        import torch

        task = self.tasks[index]
        density_bohr, metadata = cp3d.load(task.reference_cp3d, return_metadata=True)
        atoms = ase.io.read(task.cif_path)
        atoms.set_pbc(True)
        validate_structure_against_cp3d(atoms, metadata)
        shape = tuple(int(v) for v in density_bohr.shape)
        total = int(np.prod(shape, dtype=np.int64))
        rng = np.random.default_rng(self.seed + index) if self.deterministic else np.random.default_rng()
        flat_indices = rng.integers(0, total, size=self.probes, endpoint=False)
        target_values = density_bohr.reshape(-1)[flat_indices].astype(np.float32, copy=False)
        if self.target == "ed":
            # CP3D/CP2K: electron/Bohr^3. ChargE3Net convention: electron/Angstrom^3.
            targets = target_values / np.float32(ase.units.Bohr ** 3)
        else:
            # CP2K v_hartree Cube values are the Hartree potential in atomic units.
            targets = target_values
        origin_angstrom = np.asarray(metadata.get("origin", [0, 0, 0])) * BOHR_ANGSTROM

        # flat_grid_positions accepts a contiguous interval; directly decode arbitrary indices here.
        ny, nz = shape[1], shape[2]
        i = flat_indices // (ny * nz)
        rem = flat_indices % (ny * nz)
        j = rem // nz
        k = rem % nz
        frac = np.column_stack((i / shape[0], j / shape[1], k / shape[2]))
        probe_positions = origin_angstrom[None, :] + frac @ atoms.cell.array

        builder = CachedPeriodicGraphBuilder(atoms, self.cutoff)
        probe_edges, probe_displacements = builder.probe_graph(probe_positions)
        result = {
            "nodes": torch.as_tensor(atoms.numbers, dtype=torch.long),
            "atom_edges": torch.as_tensor(builder.atom_edges, dtype=torch.long),
            "atom_edges_displacement": torch.as_tensor(builder.atom_displacements, dtype=torch.float32),
            "probe_edges": torch.as_tensor(probe_edges, dtype=torch.long),
            "probe_edges_displacement": torch.as_tensor(probe_displacements, dtype=torch.float32),
            "probe_target": torch.as_tensor(targets, dtype=torch.float32),
            "num_nodes": torch.tensor(len(atoms), dtype=torch.long),
            "num_atom_edges": torch.tensor(len(builder.atom_edges), dtype=torch.long),
            "num_probes": torch.tensor(self.probes, dtype=torch.long),
            "num_probe_edges": torch.tensor(len(probe_edges), dtype=torch.long),
            "probe_xyz": torch.as_tensor(probe_positions, dtype=torch.float32),
            "atom_xyz": torch.as_tensor(atoms.positions, dtype=torch.float32),
            "cell": torch.as_tensor(atoms.cell.array, dtype=torch.float32),
            "filename": task.material,
        }
        return result


def collate_graphs(items):
    from src.charge3net.data.collate import collate_list_of_dicts

    return collate_list_of_dicts(items, pin_memory=False)


def move_batch(batch, device):
    import torch

    return {k: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for k, value in batch.items()}


def distributed_context(args):
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return world_size, rank, local_rank, device


def save_checkpoint(path, model, optimizer, scheduler, step, epoch, args, best_nmape):
    import torch

    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch": epoch,
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_nmape": best_nmape,
        "training_args": vars(args),
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class DistributedEvalSampler:
    """Shard evaluation data without DistributedSampler's padded duplicates."""

    def __init__(self, dataset, world_size, rank):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def evaluate(model, loader, device, max_batches, world_size, rank, description):
    import torch
    import torch.distributed as dist
    from tqdm.auto import tqdm

    model.eval()
    # [count, absolute error, squared error, sum(y), sum(y^2), sum(abs(y))]
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    batches = 0
    with torch.no_grad():
        progress = tqdm(
            loader, desc=description, total=min(len(loader), max_batches) if max_batches else len(loader),
            disable=rank != 0, dynamic_ncols=True, leave=True,
        )
        for batch in progress:
            batch = move_batch(batch, device)
            prediction = model(batch).reshape(-1).double()
            target = batch["probe_target"].reshape(-1).double()
            residual = prediction - target
            totals[0] += target.numel()
            totals[1] += residual.abs().sum()
            totals[2] += residual.square().sum()
            totals[3] += target.sum()
            totals[4] += target.square().sum()
            totals[5] += target.abs().sum()
            batches += 1
            if max_batches and batches >= max_batches:
                break
    if world_size > 1:
        dist.all_reduce(totals)
    count, sae, sse, sum_y, sum_y2, sum_abs_y = totals.tolist()
    sst = sum_y2 - sum_y * sum_y / max(count, 1.0)
    return {
        "mae": sae / max(count, 1.0),
        "nmape": 100.0 * sae / max(sum_abs_y, 1e-12),
        "r2": 1.0 - sse / sst if sst > 1e-20 else float("nan"),
        "points": int(count),
    }


def append_metrics(path: Path, row: dict) -> None:
    fields = list(row)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.smoke_test:
        args.limit = max(args.limit or 0, 8)
        args.max_steps = 2
        args.val_batches = 1
        args.test_batches = 1
        args.save_every = 1
        args.train_probes = min(args.train_probes, 32)
        args.val_probes = min(args.val_probes, 32)
        args.test_probes = min(args.test_probes, 32)
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from tqdm.auto import tqdm
    from src.charge3net.models.e3 import E3DensityModel
    from src.charge3net.models.scheduler import PowerDecayScheduler

    world_size, rank, local_rank, device = distributed_context(args)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    tasks = discover_training_tasks(args)
    split_path = args.split_file or (args.run_dir / "split.json")
    if world_size > 1:
        if rank == 0:
            split = make_or_load_split(
                tasks, split_path, args.seed, args.train_fraction, args.val_fraction,
                args.split_strategy,
            )
        torch.distributed.barrier()
        if rank != 0:
            split = make_or_load_split(
                tasks, split_path, args.seed, args.train_fraction, args.val_fraction,
                args.split_strategy,
            )
    else:
        split = make_or_load_split(
            tasks, split_path, args.seed, args.train_fraction, args.val_fraction,
            args.split_strategy,
        )
    by_name = {task.material: task for task in tasks}
    train_tasks = [by_name[name] for name in split["train"]]
    val_tasks = [by_name[name] for name in split["validation"]]
    test_tasks = [by_name[name] for name in split["test"]]
    if rank == 0:
        print(
            f"materials={len(tasks)} train={len(train_tasks)} val={len(val_tasks)} "
            f"test={len(test_tasks)} "
            f"world_size={world_size} device={device}", flush=True
        )
    if args.prepare_split_only:
        if world_size > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return 0

    train_set = CP3DProbeDataset(train_tasks, args.train_probes, args.cutoff, False, args.seed,
                                 args.target)
    val_set = CP3DProbeDataset(val_tasks, args.val_probes, args.cutoff, True,
                               args.seed + 10_000_000, args.target)
    test_set = CP3DProbeDataset(test_tasks, args.test_probes, args.cutoff, True,
                                args.seed + 20_000_000, args.target)
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedEvalSampler(val_set, world_size, rank) if world_size > 1 else None
    test_sampler = DistributedEvalSampler(test_set, world_size, rank) if world_size > 1 else None
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.workers, collate_fn=collate_graphs,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.workers, collate_fn=collate_graphs,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, sampler=test_sampler,
        num_workers=args.workers, collate_fn=collate_graphs,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    model = E3DensityModel(
        num_interactions=3, num_neighbors=20, mul=500, lmax=4,
        cutoff=args.cutoff, basis="gaussian", num_basis=20,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = PowerDecayScheduler(
        optimizer, alpha=args.scheduler_alpha, beta=args.scheduler_beta
    )
    step = epoch = 0
    best_nmape = float("inf")
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
        if rank == 0:
            print(f"Initialized model weights from {args.init_checkpoint}", flush=True)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        step, epoch = int(payload["step"]), int(payload["epoch"])
        best_nmape = float(payload.get("best_nmape", best_nmape))
        if rank == 0:
            print(f"Resumed {args.resume} at step {step}", flush=True)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    latest = args.run_dir / "checkpoint_latest.pt"
    best = args.run_dir / "checkpoint_best.pt"
    model.train()
    while step < args.max_steps:
        epoch_started = time.time()
        train_totals = torch.zeros(2, dtype=torch.float64, device=device)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        progress = tqdm(
            train_loader, desc=f"Epoch {epoch + 1} train", disable=rank != 0,
            dynamic_ncols=True, leave=True,
        )
        for batch in progress:
            if step >= args.max_steps:
                break
            batch = move_batch(batch, device)
            prediction = model(batch)
            loss = torch.nn.functional.l1_loss(prediction, batch["probe_target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            with torch.no_grad():
                train_totals[0] += torch.abs(prediction - batch["probe_target"]).sum().double()
                train_totals[1] += batch["probe_target"].numel()
            if rank == 0:
                progress.set_postfix(step=step, mae=f"{loss.item():.4g}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.3g}")
            if rank == 0 and (step % args.save_every == 0 or step == args.max_steps):
                save_checkpoint(latest, model, optimizer, scheduler, step, epoch, args, best_nmape)
        if world_size > 1:
            torch.distributed.all_reduce(train_totals)
        train_mae = float((train_totals[0] / train_totals[1].clamp_min(1)).item())

        # Bypass the DDP wrapper during evaluation: ranks can own unequal shard sizes.
        eval_model = model.module if hasattr(model, "module") else model
        val_metrics = evaluate(eval_model, val_loader, device, args.val_batches, world_size,
                               rank, f"Epoch {epoch + 1} validation")
        test_metrics = evaluate(eval_model, test_loader, device, args.test_batches, world_size,
                                rank, f"Epoch {epoch + 1} test")
        model.train()
        epoch += 1
        if rank == 0:
            target_unit = "e/Ang^3" if args.target == "ed" else "Hartree/e"
            print(
                f"EPOCH {epoch} | step={step} | target={args.target} | "
                f"train_MAE={train_mae:.8g} {target_unit} | "
                f"val_MAE={val_metrics['mae']:.8g} {target_unit} | "
                f"val_R2={val_metrics['r2']:.6f} | val_normalized_MAE={val_metrics['nmape']:.6f}% | "
                f"test_MAE={test_metrics['mae']:.8g} {target_unit} | "
                f"test_R2={test_metrics['r2']:.6f} | test_normalized_MAE={test_metrics['nmape']:.6f}% | "
                f"seconds={time.time() - epoch_started:.1f}", flush=True,
            )
            append_metrics(args.run_dir / "metrics.csv", {
                "epoch": epoch, "step": step, "train_mae": train_mae,
                "val_mae": val_metrics["mae"], "val_r2": val_metrics["r2"],
                "val_normalized_mae_percent": val_metrics["nmape"],
                "test_mae": test_metrics["mae"], "test_r2": test_metrics["r2"],
                "test_normalized_mae_percent": test_metrics["nmape"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.time() - epoch_started,
            })
            if val_metrics["nmape"] < best_nmape:
                best_nmape = val_metrics["nmape"]
                save_checkpoint(best, model, optimizer, scheduler, step, epoch, args, best_nmape)
            save_checkpoint(latest, model, optimizer, scheduler, step, epoch, args, best_nmape)

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
