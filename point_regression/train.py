from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mofreg.config import apply_overrides, load_config
from mofreg.data import MOFPointCloudDataset, load_records, split_records
from mofreg.models import build_model
from mofreg.train_utils import compute_metrics, ensure_dir, resolve_device, save_json, set_seed

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MOF ESP point-cloud regressor.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config values with key=value pairs, e.g. model.name=transformer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    run_training(config)


def run_training(config: dict) -> None:
    set_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    output_dir = ensure_dir(config["training"]["output_dir"])
    device = resolve_device(config["training"].get("device", "auto"))

    records = load_records(config)
    splits = split_records(records, config)
    feature_blocks = config["data"]["feature_blocks"]
    datasets = {
        split: MOFPointCloudDataset(
            split_records_,
            feature_blocks=feature_blocks,
            augment=(split == "train" and bool(config["data"].get("augment", False))),
            normalize_xyz=bool(config["data"].get("normalize_xyz", True)),
            frac_pe_frequencies=int(config["data"].get("frac_pe_frequencies", 1)),
            point_pack_path=config["data"].get("point_pack_path"),
        )
        for split, split_records_ in splits.items()
    }

    target_transform = str(config["training"].get("target_transform", "none")).lower()
    if target_transform not in {"none", "log1p"}:
        raise ValueError(f"Unsupported target transform: {target_transform}")
    train_targets = np.asarray([record.target for record in splits["train"]], dtype=np.float32)
    if target_transform == "log1p":
        if np.any(train_targets < 0):
            raise ValueError("log1p target transform requires non-negative targets")
        train_targets = np.log1p(train_targets)
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std() + 1e-8)
    normalize_target = bool(config["training"].get("normalize_target", True))
    grad_clip = float(config["training"].get("grad_clip", 0.0))

    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", True))
    prefetch_factor = int(config["training"].get("prefetch_factor", 2))
    loaders = {}
    for split, dataset in datasets.items():
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(config["training"]["batch_size"]),
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = prefetch_factor
        loaders[split] = DataLoader(**loader_kwargs)

    input_dim = datasets["train"][0]["features"].shape[-1]
    model = build_model(config, input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_checkpoint = output_dir / "best_model.pt"
    history: list[dict] = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        epoch_start = time.time()
        train_loss = run_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
            grad_clip=grad_clip,
            target_transform=target_transform,
            epoch=epoch,
            split_name="train",
        )
        scheduler.step()

        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            device=device,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
            epoch=epoch,
            split_name="val",
            target_transform=target_transform,
        )
        test_metrics = evaluate(
            model=model,
            loader=loaders["test"],
            device=device,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
            epoch=epoch,
            split_name="test",
        target_transform=target_transform,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val": val_metrics,
                "test": test_metrics,
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "epoch_seconds": round(time.time() - epoch_start, 2),
                    "train_loss": round(train_loss, 6),
                    "val_mae": round(val_metrics["mae"], 6),
                    "val_rmse": round(val_metrics["rmse"], 6),
                    "val_r2": round(val_metrics["r2"], 6) if np.isfinite(val_metrics["r2"]) else None,
                    "test_mae": round(test_metrics["mae"], 6),
                    "test_rmse": round(test_metrics["rmse"], 6),
                    "test_r2": round(test_metrics["r2"], 6) if np.isfinite(test_metrics["r2"]) else None,
                },
                ensure_ascii=True,
            )
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "feature_blocks": feature_blocks,
                    "target_transform": target_transform,
                },
                best_checkpoint,
            )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
            device=device,
            normalize_target=normalize_target,
            target_mean=target_mean,
            target_std=target_std,
            epoch="best",
            split_name="test",
            target_transform=target_transform,
        )

    summary = {
        "target_column": config["data"]["target_column"],
        "point_cloud_variant": config["data"]["point_cloud_variant"],
        "num_points": config["data"]["num_points"],
        "model_name": config["model"]["name"],
        "feature_blocks": feature_blocks,
        "frac_pe_frequencies": int(config["data"].get("frac_pe_frequencies", 1)),
        "target_transform": target_transform,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_val_rmse": best_val_rmse,
        "test_metrics": test_metrics,
    }
    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "history.json", {"history": history})
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
    grad_clip: float,
    epoch: int,
    split_name: str,
    target_transform: str,
) -> float:
    model.train(mode=train)
    total_loss = 0.0
    total_count = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch} [{split_name}]",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for batch in progress:
        features = batch["features"].to(device, non_blocking=True)
        xyz = batch["xyz"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if target_transform == "log1p":
            targets = torch.log1p(targets)
        if normalize_target:
            targets_for_loss = (targets - target_mean) / target_std
        else:
            targets_for_loss = targets

        with torch.set_grad_enabled(train):
            preds = model(features, xyz)
            loss = criterion(preds, targets_for_loss)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, total_count)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalize_target: bool,
    target_mean: float,
    target_std: float,
    epoch: int | str,
    split_name: str,
    target_transform: str,
) -> dict[str, float]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} [{split_name}]",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for batch in progress:
            features = batch["features"].to(device, non_blocking=True)
            xyz = batch["xyz"].to(device, non_blocking=True)
            targets = batch["target"].cpu().numpy()
            preds = model(features, xyz).cpu().numpy()
            if normalize_target:
                preds = preds * target_std + target_mean
            if target_transform == "log1p":
                preds = np.expm1(preds)
            all_preds.append(preds)
            all_targets.append(targets)
            if all_preds:
                current_metrics = compute_metrics(
                    np.concatenate(all_preds, axis=0),
                    np.concatenate(all_targets, axis=0),
                )
                progress.set_postfix(rmse=f"{current_metrics['rmse']:.4f}")

    pred_array = np.concatenate(all_preds, axis=0)
    target_array = np.concatenate(all_targets, axis=0)
    return compute_metrics(pred_array, target_array)
if __name__ == "__main__":
    main()
