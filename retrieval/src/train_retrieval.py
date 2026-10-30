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
from cif_ed_retrieval.data import CIFPointCloudDataset, build_index, collate_cif_point_cloud, save_index, split_records
from cif_ed_retrieval.models import CIFEDRetrievalModel, contrastive_loss

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):
        del kwargs
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIF-ED retrieval baseline.")
    parser.add_argument("--config", type=str, required=True, help="YAML config path.")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        help="Override config values with key=value pairs.",
    )
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to evaluate.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate --checkpoint without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    run_training(config, checkpoint_path=args.checkpoint, evaluate_only=args.evaluate_only)


def run_training(config: dict, checkpoint_path: str | None = None, evaluate_only: bool = False) -> None:
    if evaluate_only and not checkpoint_path:
        raise ValueError("--evaluate-only requires --checkpoint")
    print(f"Initializing run in {config['training']['output_dir']}", flush=True)
    set_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    output_dir = ensure_dir(config["training"]["output_dir"])
    device = resolve_device(config["training"].get("device", "auto"))
    if config["training"].get("require_cuda", False) and device.type != "cuda":
        raise RuntimeError(
            "This retrieval setup requires CUDA because the X3D encoder depends on GPU point ops. "
            f"Resolved device was {device} and torch.cuda.is_available()={torch.cuda.is_available()}."
        )

    print("Loading dataset index...", flush=True)
    records = build_index(config)
    print(f"Dataset index loaded: {len(records)} pairs", flush=True)
    save_index(records, output_dir / "pairs.csv")
    splits = split_records(records)
    dataset_map = {
        split: CIFPointCloudDataset(
            split_records_,
            max_num_nbr=int(config["data"]["cif_graph"]["max_num_nbr"]),
            radius=float(config["data"]["cif_graph"]["radius"]),
            dmin=float(config["data"]["cif_graph"]["dmin"]),
            step=float(config["data"]["cif_graph"]["step"]),
            feature_blocks=config["data"].get("feature_blocks", ["xyz", "scalar", "frac"]),
            frac_pe_frequencies=int(config["data"].get("frac_pe_frequencies", 1)),
            normalize_xyz=bool(config["data"]["normalize_xyz"]),
            augment=(split == "train" and bool(config["data"]["augment"])),
            cache_data=bool(config["data"].get("cache_data", True)),
            cif_cache_dir=config["data"].get("cif_cache_dir"),
            point_pack_path=config["data"].get("point_pack_path"),
        )
        for split, split_records_ in splits.items()
    }

    num_workers = int(config["training"].get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(config["training"].get("pin_memory", True)),
        "collate_fn": collate_cif_point_cloud,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))

    loaders = {
        split: DataLoader(dataset, shuffle=(split == "train"), **loader_kwargs)
        for split, dataset in dataset_map.items()
    }

    print("Loading first sample and initializing model...", flush=True)
    sample = dataset_map["train"][0]
    model = CIFEDRetrievalModel(
        config=config,
        orig_atom_fea_len=int(sample["atom_fea"].shape[-1]),
        nbr_fea_len=int(sample["nbr_fea"].shape[-1]),
        ed_input_dim=int(sample["point_features"].shape[-1]),
    ).to(device)

    if evaluate_only:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        summary = {
            "checkpoint": str(checkpoint_path),
            "dataset_sizes": {split: len(dataset) for split, dataset in dataset_map.items()},
            "test_metrics": evaluate_retrieval(model, loaders["test"], device, split_name="test"),
        }
        save_json(output_dir / "metrics.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    print(f"Model initialized on {device}; starting training", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config["training"]["epochs"])),
    )

    history: list[dict[str, object]] = []
    best_val_r1 = -1.0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    start_epoch = 1
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint.get("history", [])
        best_val_r1 = float(checkpoint.get("best_val_r1", -1.0))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming from {last_path} at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        train_stats = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            temperature=float(config["training"]["temperature"]),
            epoch=epoch,
            total_epochs=int(config["training"]["epochs"]),
        )
        scheduler.step()

        val_metrics = evaluate_retrieval(model, loaders["val"], device, split_name="val")
        test_metrics = evaluate_retrieval(model, loaders["test"], device, split_name="test")
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_diag_sim": train_stats["diag_sim"],
            "val_r1_cif_to_point": val_metrics["cif_to_point"]["Recall@1"],
            "val_r5_cif_to_point": val_metrics["cif_to_point"]["Recall@5"],
            "val_mrr_cif_to_point": val_metrics["cif_to_point"]["MRR"],
            "val_r1_point_to_cif": val_metrics["point_to_cif"]["Recall@1"],
            "val_r5_point_to_cif": val_metrics["point_to_cif"]["Recall@5"],
            "val_mrr_point_to_cif": val_metrics["point_to_cif"]["MRR"],
            "test_r1_cif_to_point": test_metrics["cif_to_point"]["Recall@1"],
            "test_r5_cif_to_point": test_metrics["cif_to_point"]["Recall@5"],
            "test_mrr_cif_to_point": test_metrics["cif_to_point"]["MRR"],
            "test_r1_point_to_cif": test_metrics["point_to_cif"]["Recall@1"],
            "test_r5_point_to_cif": test_metrics["point_to_cif"]["Recall@5"],
            "test_mrr_point_to_cif": test_metrics["point_to_cif"]["MRR"],
        }
        history.append(row)
        append_history(output_dir / "history.csv", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "train_diag_sim": row["train_diag_sim"],
                    "val": val_metrics,
                    "test": test_metrics,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

        val_r1 = 0.5 * (
            val_metrics["cif_to_point"]["Recall@1"] + val_metrics["point_to_cif"]["Recall@1"]
        )
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                best_path,
            )
        save_checkpoint_atomic(
            last_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_val_r1": best_val_r1,
                "config": config,
            },
        )

    best = torch.load(best_path, map_location=device)
    model.load_state_dict(best["model_state"])
    final_val = evaluate_retrieval(model, loaders["val"], device, split_name="val-best")
    final_test = evaluate_retrieval(model, loaders["test"], device, split_name="test-best")
    summary = {
        "dataset_sizes": {split: len(dataset) for split, dataset in dataset_map.items()},
        "best_val_metrics": final_val,
        "best_test_metrics": final_test,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def train_epoch(
    model: CIFEDRetrievalModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_diag = 0.0
    total_count = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d}/{total_epochs:03d} [train]",
        dynamic_ncols=True,
        leave=False,
    )
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        cif_z, point_z = model(batch)
        loss, logits = contrastive_loss(cif_z, point_z, temperature=temperature)
        loss.backward()
        optimizer.step()

        batch_size = cif_z.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_diag += float(logits.diag().mean().item()) * batch_size
        total_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {"loss": total_loss / max(1, total_count), "diag_sim": total_diag / max(1, total_count)}


@torch.no_grad()
def evaluate_retrieval(
    model: CIFEDRetrievalModel,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> dict[str, dict[str, float]]:
    model.eval()
    cif_embeddings: list[torch.Tensor] = []
    point_embeddings: list[torch.Tensor] = []
    sample_ids: list[str] = []

    for batch in tqdm(loader, desc=f"Eval [{split_name}]", dynamic_ncols=True, leave=False):
        batch = move_batch_to_device(batch, device)
        cif_z, point_z = model(batch)
        cif_embeddings.append(cif_z.cpu())
        point_embeddings.append(point_z.cpu())
        sample_ids.extend(batch["sample_ids"])

    cif_matrix = torch.cat(cif_embeddings, dim=0)
    point_matrix = torch.cat(point_embeddings, dim=0)
    similarity = cif_matrix @ point_matrix.T
    targets = list(range(len(sample_ids)))
    return {
        "cif_to_point": compute_ranking_metrics(similarity, targets),
        "point_to_cif": compute_ranking_metrics(similarity.T, targets),
    }


def compute_ranking_metrics(similarity: torch.Tensor, targets: list[int]) -> dict[str, float]:
    ranks: list[int] = []
    for row_idx, target_idx in enumerate(targets):
        scores = similarity[row_idx]
        order = torch.argsort(scores, descending=True)
        rank = int((order == target_idx).nonzero(as_tuple=False).item()) + 1
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=np.int64)
    return {
        "Recall@1": float(np.mean(ranks_np <= 1)),
        "Recall@5": float(np.mean(ranks_np <= 5)),
        "Recall@10": float(np.mean(ranks_np <= 10)),
        "MRR": float(np.mean(1.0 / ranks_np)),
        "MedianRank": float(np.median(ranks_np)),
    }


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = batch["cif"]
    return {
        "sample_ids": batch["sample_ids"],
        "cif": (
            atom_fea.to(device, non_blocking=True),
            nbr_fea.to(device, non_blocking=True),
            nbr_fea_idx.to(device, non_blocking=True),
            [idx.to(device, non_blocking=True) for idx in crystal_atom_idx],
        ),
        "point_xyz": batch["point_xyz"].to(device, non_blocking=True),
        "point_features": batch["point_features"].to(device, non_blocking=True),
    }


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_checkpoint_atomic(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def append_history(path: str | Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
