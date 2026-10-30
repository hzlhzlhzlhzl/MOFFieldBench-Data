from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import apply_overrides, load_config
from .data import make_datasets, paired_collate
from .model import build_multimodal_model
from .registry import apply_registry_selection, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIF + EP/ESP multimodal regressor.")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--task", help="task ID from --tasks-registry")
    parser.add_argument("--modalities", default="cif_ed_esp", help="cif, cif_ed, cif_esp, or cif_ed_esp")
    parser.add_argument("--tasks-registry", default="configs/tasks.json", help="13-task registry JSON")
    parser.add_argument("--data-manifest", default="data/dataset_manifest.example.json", help="local dataset/point-cloud manifest JSON")
    parser.add_argument("--list-tasks", action="store_true", help="print registered tasks and exit")
    parser.add_argument("--set", nargs="*", default=[], help="key=value configuration overrides")
    parser.add_argument("--dry-run", action="store_true", help="validate data/model and run one batch")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate existing best checkpoint on test without training")
    parser.add_argument("--export-all-splits", action="store_true", help="with --evaluate-only, export train/val/test predictions")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is False")
    return device


def safe_target_name(target: str) -> str:
    return target.replace("/", "_per_").replace("%", "pct")


def make_loaders(config: dict, datasets: dict) -> dict[str, DataLoader]:
    train_cfg = config["training"]
    workers = int(train_cfg.get("num_workers", 0))
    loaders = {}
    for split, dataset in datasets.items():
        kwargs = dict(
            dataset=dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            collate_fn=paired_collate,
            drop_last=False,
        )
        if workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
        loaders[split] = DataLoader(**kwargs)
    return loaders


def move_batch(batch, device: torch.device):
    if len(batch) == 3:
        point, secondary, cif = batch
    else:
        point, cif = batch
        secondary = None
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = cif
    inputs = (
        point["features"].to(device, non_blocking=True),
        point["xyz"].to(device, non_blocking=True),
        atom_fea.to(device, non_blocking=True),
        nbr_fea.to(device, non_blocking=True),
        nbr_fea_idx.to(device, non_blocking=True),
        [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
    )
    targets = point["target"].to(device, non_blocking=True)
    if secondary is not None:
        inputs = inputs + (
            secondary["features"].to(device, non_blocking=True),
            secondary["xyz"].to(device, non_blocking=True),
        )
    return inputs, targets, point["sample_id"]


def run_train_epoch(model, loader, optimizer, criterion, device, mean, std, grad_clip, epoch):
    model.train()
    total_loss = 0.0
    total_count = 0
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", dynamic_ncols=True, leave=False)
    for batch in progress:
        inputs, targets, _ = move_batch(batch, device)
        normalized = (targets - mean) / std
        optimizer.zero_grad(set_to_none=True)
        output = model(*inputs)
        loss = criterion(output, normalized)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = targets.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        progress.set_postfix(loss=f"{total_loss / total_count:.5f}")
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, mean, std, split, epoch):
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions, targets_all, sample_ids = [], [], []
    progress = tqdm(loader, desc=f"Epoch {epoch} [{split}]", dynamic_ncols=True, leave=False)
    for batch in progress:
        inputs, targets, ids = move_batch(batch, device)
        normalized = (targets - mean) / std
        output = model(*inputs)
        loss = criterion(output, normalized)
        predictions.extend((output * std + mean).cpu().tolist())
        targets_all.extend(targets.cpu().tolist())
        sample_ids.extend(ids)
        total_loss += float(loss.item()) * targets.shape[0]
        total_count += targets.shape[0]
    mae = mean_absolute_error(targets_all, predictions)
    rmse = math.sqrt(mean_squared_error(targets_all, predictions))
    r2 = r2_score(targets_all, predictions)
    return {
        "loss": total_loss / total_count,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "ids": sample_ids,
        "targets": targets_all,
        "predictions": predictions,
    }


def public_metrics(metrics: dict) -> dict:
    return {key: metrics[key] for key in ("loss", "mae", "rmse", "r2")}


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        for key, value in load_json(args.tasks_registry)["tasks"].items():
            print(f"{key}\t{value['display_name']}\t{value['target_column']}")
        return
    config = apply_overrides(load_config(args.config), args.set)
    if args.task:
        config = apply_registry_selection(config, args.tasks_registry, args.data_manifest, args.task, args.modalities)
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(str(config["training"].get("device", "auto")))
    print(f"Device: {device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

    datasets, alignment_report, cif_root = make_datasets(config)
    loaders = make_loaders(config, datasets)
    sample = datasets["train"][0]
    if len(sample) == 3:
        sample_point, sample_secondary, sample_cif = sample
        config["data"]["secondary_input_dim"] = int(sample_secondary["features"].shape[-1])
    else:
        sample_point, sample_cif = sample
    input_dim = int(sample_point["features"].shape[-1])
    model = build_multimodal_model(config, input_dim, sample_cif).to(device)
    print(json.dumps(alignment_report, indent=2, ensure_ascii=False))
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.dry_run:
        model.eval()
        with torch.no_grad():
            inputs, targets, ids = move_batch(next(iter(loaders["train"])), device)
            predictions = model(*inputs)
        print(json.dumps({"dry_run": "ok", "batch_size": len(ids), "prediction_shape": list(predictions.shape), "target_shape": list(targets.shape)}, indent=2))
        return

    train_cfg = config["training"]
    output_root = Path(train_cfg["output_dir"])
    target_name = safe_target_name(config["data"]["target_column"])
    output_dir = output_root if output_root.name == target_name else output_root / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False, default=str)
    with (output_dir / "alignment_report.json").open("w", encoding="utf-8") as handle:
        json.dump(alignment_report, handle, indent=2, ensure_ascii=False)

    train_targets = np.asarray([r.target for r in datasets["train"].records], dtype=np.float32)
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std() + 1e-8)
    criterion = nn.MSELoss()
    selection_metric = str(train_cfg.get("selection_metric", "mae"))
    if selection_metric not in {"mae", "rmse"}:
        raise ValueError(f"Unsupported regression selection metric: {selection_metric}")
    checkpoint_path = output_dir / "best_model.pt"
    # Optional two-stage fusion: initialise the strong CIF branch from a
    # completed CIF-only checkpoint and freeze it.  The point branch then
    # learns a residual correction, so validation selection can never prefer
    # an untrained random CIF predictor over the known baseline.
    init_checkpoint = train_cfg.get("init_cif_checkpoint")
    freeze_cif = bool(train_cfg.get("freeze_cif", False))
    freeze_cif_encoder = bool(train_cfg.get("freeze_cif_encoder", freeze_cif))
    freeze_cif_head = bool(train_cfg.get("freeze_cif_head", freeze_cif))
    if init_checkpoint and not checkpoint_path.is_file():
        init_state = torch.load(str(init_checkpoint), map_location=device, weights_only=False)["model_state"]
        own_state = model.state_dict()
        compatible = {key: value for key, value in init_state.items()
                      if key in own_state and own_state[key].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        print(json.dumps({"initialized_from_cif": str(init_checkpoint), "loaded_tensors": len(compatible)}))
        if freeze_cif_encoder or freeze_cif_head:
            frozen_prefixes = ()
            if freeze_cif_encoder:
                frozen_prefixes += ("cif_encoder.", "cif_projection.")
            if freeze_cif_head:
                frozen_prefixes += ("cif_head.",)
            for name, parameter in model.named_parameters():
                if name.startswith(frozen_prefixes):
                    parameter.requires_grad_(False)
            print(json.dumps({"frozen_cif_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(train_cfg["epochs"]))
    )
    # Resume an interrupted run from its best checkpoint.  The optimizer and
    # scheduler are intentionally re-created with the current config, while
    # the best validation score and epoch are retained for correct selection.
    resume_checkpoint = None
    if checkpoint_path.is_file():
        resume_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(resume_checkpoint["model_state"])
    if resume_checkpoint:
        saved_score = resume_checkpoint.get("best_val_score")
        if saved_score is None:
            saved_score = resume_checkpoint.get("best_val_rmse") if selection_metric == "rmse" else resume_checkpoint.get("best_val_mae")
        best_val_score = float(saved_score) if saved_score is not None else float("inf")
    else:
        best_val_score = float("inf")
    if args.evaluate_only:
        if not resume_checkpoint:
            raise FileNotFoundError(f"--evaluate-only requires {checkpoint_path}")
        checkpoint = resume_checkpoint
        if args.export_all_splits:
            for split in ("train", "val"):
                result = evaluate(model, loaders[split], criterion, device, target_mean, target_std, f"{split}-best", checkpoint["epoch"])
                pd.DataFrame({"id": result["ids"], "target": result["targets"], "prediction": result["predictions"]}).to_csv(output_dir / f"{split}_predictions.csv", index=False)
        test = evaluate(model, loaders["test"], criterion, device, target_mean, target_std, "test-best", checkpoint["epoch"])
        pd.DataFrame({"id": test["ids"], "target": test["targets"], "prediction": test["predictions"]}).to_csv(output_dir / "test_predictions.csv", index=False)
        metrics = {"target_column": config["data"]["target_column"], "device": str(device), "cif_data_root": str(cif_root),
                   "dataset_sizes": {key: len(value) for key, value in datasets.items()}, "best_epoch": checkpoint["epoch"],
                   "selection_metric": checkpoint.get("selection_metric", "mae"), "best_val_score": checkpoint.get("best_val_score", checkpoint["best_val_mae"]),
                   "best_val_mae": checkpoint["best_val_mae"], "best_val_rmse": checkpoint.get("best_val_rmse"), "test_metrics": public_metrics(test)}
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return
    # Old runs stored flattened CSV rows (val_mae/val_rmse), whereas new
    # in-memory rows use a nested `val` object.  Do not replay old rows into
    # the writer; the checkpoint epoch/score is sufficient for resuming and
    # avoids a format-dependent KeyError.
    history = []
    # Score the loaded CIF predictor before residual training.  Saving it as
    # epoch zero makes the comparison to CIF-only exact under this split.
    if init_checkpoint and not resume_checkpoint:
        initial_val = evaluate(model, loaders["val"], criterion, device, target_mean, target_std, "val-init", 0)
        best_val_score = initial_val[selection_metric]
        torch.save({"model_state": model.state_dict(), "config": config, "epoch": 0,
                    "target_mean": target_mean, "target_std": target_std,
                    "selection_metric": selection_metric, "best_val_score": best_val_score,
                    "best_val_mae": initial_val["mae"], "best_val_rmse": initial_val["rmse"]}, checkpoint_path)
        print(json.dumps({"initial_cif_val": public_metrics(initial_val)}, ensure_ascii=False))
    start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1 if resume_checkpoint else 1
    if start_epoch > int(train_cfg["epochs"]):
        start_epoch = int(train_cfg["epochs"]) + 1
    for epoch in range(start_epoch, int(train_cfg["epochs"]) + 1):
        start = time.time()
        train_loss = run_train_epoch(
            model, loaders["train"], optimizer, criterion, device, target_mean, target_std,
            float(train_cfg.get("grad_clip", 0.0)), epoch,
        )
        scheduler.step()
        val = evaluate(model, loaders["val"], criterion, device, target_mean, target_std, "val", epoch)
        row = {"epoch": epoch, "seconds": time.time() - start, "train_loss": train_loss, "val": public_metrics(val)}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        pd.DataFrame([{"epoch": h["epoch"], "seconds": h["seconds"], "train_loss": h["train_loss"], **{f"val_{k}": v for k, v in h["val"].items()}} for h in history]).to_csv(output_dir / "history.csv", index=False)
        if val[selection_metric] < best_val_score:
            best_val_score = val[selection_metric]
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch,
                        "target_mean": target_mean, "target_std": target_std,
                        "selection_metric": selection_metric, "best_val_score": best_val_score,
                        "best_val_mae": val["mae"], "best_val_rmse": val["rmse"]}, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, loaders["test"], criterion, device, target_mean, target_std, "test-best", checkpoint["epoch"])
    pd.DataFrame({"id": test["ids"], "target": test["targets"], "prediction": test["predictions"]}).to_csv(output_dir / "test_predictions.csv", index=False)
    metrics = {
        "target_column": config["data"]["target_column"],
        "device": str(device),
        "cif_data_root": str(cif_root),
        "dataset_sizes": {key: len(value) for key, value in datasets.items()},
        "best_epoch": checkpoint["epoch"],
        "selection_metric": checkpoint.get("selection_metric", "mae"),
        "best_val_score": checkpoint.get("best_val_score", checkpoint["best_val_mae"]),
        "best_val_mae": checkpoint["best_val_mae"],
        "best_val_rmse": checkpoint.get("best_val_rmse"),
        "test_metrics": public_metrics(test),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
