from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from cif_ed_retrieval.config import apply_overrides, load_config
from cif_ed_retrieval.data import (
    PointCloudPairDataset,
    build_point_pair_index,
    collate_point_cloud_pair,
    split_records,
)
from cif_ed_retrieval.models import EDESPRetrievalModel, contrastive_loss

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Iterable, **kwargs):
        del kwargs
        return iterable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ED-to-ESP point-cloud retrieval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--checkpoint", help="Checkpoint to evaluate.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate --checkpoint without training.")
    args = parser.parse_args()
    run_training(apply_overrides(load_config(args.config), args.set), args.checkpoint, args.evaluate_only)


def run_training(config: dict, checkpoint_path: str | None = None, evaluate_only: bool = False) -> None:
    if evaluate_only and not checkpoint_path:
        raise ValueError("--evaluate-only requires --checkpoint")
    print(f"Initializing run in {config['training']['output_dir']}", flush=True)
    set_seed(int(config["seed"]))
    device = resolve_device(config["training"].get("device", "auto"))
    if config["training"].get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError("ED/ESP X3D retrieval requires CUDA.")
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading dataset index...", flush=True)
    records = build_point_pair_index(config)
    print(f"Dataset index loaded: {len(records)} pairs", flush=True)
    save_pairs(records, output_dir / "pairs.csv")
    splits = split_records(records)
    common = dict(
        feature_blocks=config["data"].get("feature_blocks", ["frac_pe"]),
        frac_pe_frequencies=int(config["data"].get("frac_pe_frequencies", 2)),
        normalize_xyz=bool(config["data"].get("normalize_xyz", True)),
        cache_data=bool(config["data"].get("cache_data", False)),
        ed_point_pack_path=config["data"].get("ed_point_pack_path"),
        esp_point_pack_path=config["data"].get("esp_point_pack_path"),
    )
    datasets = {
        name: PointCloudPairDataset(items, augment=(name == "train" and bool(config["data"].get("augment", True))), **common)
        for name, items in splits.items()
    }
    loader_kwargs = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        collate_fn=collate_point_cloud_pair,
        drop_last=False,
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))
    loaders = {name: DataLoader(ds, shuffle=(name == "train"), **loader_kwargs) for name, ds in datasets.items()}
    print("Loading first sample and initializing model...", flush=True)
    sample = datasets["train"][0]
    model = EDESPRetrievalModel(
        config,
        ed_input_dim=int(sample["ed_features"].shape[-1]),
        esp_input_dim=int(sample["esp_features"].shape[-1]),
    ).to(device)
    if evaluate_only:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        summary = {
            "checkpoint": str(checkpoint_path),
            "dataset_sizes": {k: len(v) for k, v in datasets.items()},
            "test_metrics": evaluate(model, loaders["test"], device, "test"),
        }
        (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return
    print(f"Model initialized on {device}; starting training", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["training"]["epochs"])))
    best_r1 = -1.0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history = []
    start_epoch = 1
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint.get("history", [])
        best_r1 = float(checkpoint.get("best_r1", -1.0))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from {last_path} at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_stats = train_epoch(model, loaders["train"], optimizer, device, float(config["training"]["temperature"]), epoch, int(config["training"]["epochs"]))
        scheduler.step()
        val = evaluate(model, loaders["val"], device, "val")
        test = evaluate(model, loaders["test"], device, "test")
        row = {"epoch": epoch, "train_loss": train_stats["loss"], "val_ed_to_esp_r1": val["ed_to_esp"]["Recall@1"], "val_esp_to_ed_r1": val["esp_to_ed"]["Recall@1"], "test_ed_to_esp_r1": test["ed_to_esp"]["Recall@1"], "test_esp_to_ed_r1": test["esp_to_ed"]["Recall@1"]}
        history.append(row)
        save_history(output_dir / "history.csv", history)
        print(json.dumps({"epoch": epoch, "train": train_stats, "val": val, "test": test}), flush=True)
        score = 0.5 * (val["ed_to_esp"]["Recall@1"] + val["esp_to_ed"]["Recall@1"])
        if score > best_r1:
            best_r1 = score
            torch.save({"model_state": model.state_dict(), "config": config, "val_metrics": val, "test_metrics": test}, best_path)
        save_checkpoint_atomic(last_path, {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "history": history, "best_r1": best_r1, "config": config})
    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    summary = {"dataset_sizes": {k: len(v) for k, v in datasets.items()}, "best_val_metrics": evaluate(model, loaders["val"], device, "val-best"), "best_test_metrics": evaluate(model, loaders["test"], device, "test-best"), "output_dir": str(output_dir)}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def train_epoch(model, loader, optimizer, device, temperature, epoch, total_epochs):
    model.train(); total_loss = 0.0; total = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs:03d} [train]", dynamic_ncols=True, leave=False):
        batch = move_batch(batch, device); optimizer.zero_grad(set_to_none=True)
        ed_z, esp_z = model(batch); loss, _ = contrastive_loss(ed_z, esp_z, temperature)
        loss.backward(); optimizer.step()
        n = ed_z.shape[0]; total_loss += float(loss.item()) * n; total += n
    return {"loss": total_loss / max(1, total)}


@torch.no_grad()
def evaluate(model, loader, device, name):
    model.eval(); ed_all = []; esp_all = []
    for batch in tqdm(loader, desc=f"Eval [{name}]", dynamic_ncols=True, leave=False):
        ed_z, esp_z = model(move_batch(batch, device)); ed_all.append(ed_z.cpu()); esp_all.append(esp_z.cpu())
    sim = torch.cat(ed_all) @ torch.cat(esp_all).T; targets = list(range(sim.shape[0]))
    return {"ed_to_esp": ranking(sim, targets), "esp_to_ed": ranking(sim.T, targets)}


def ranking(similarity, targets):
    ranks = []
    for i, target in enumerate(targets):
        order = torch.argsort(similarity[i], descending=True)
        ranks.append(int((order == target).nonzero().item()) + 1)
    ranks = np.asarray(ranks)
    return {"Recall@1": float(np.mean(ranks <= 1)), "Recall@5": float(np.mean(ranks <= 5)), "Recall@10": float(np.mean(ranks <= 10)), "MRR": float(np.mean(1.0 / ranks)), "MedianRank": float(np.median(ranks))}


def move_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def save_pairs(records, path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["sample_id", "ed_path", "esp_path", "split"])
        writer.writerows((r.sample_id, r.ed_path, r.esp_path, r.split) for r in records)


def save_history(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_checkpoint_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resolve_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
