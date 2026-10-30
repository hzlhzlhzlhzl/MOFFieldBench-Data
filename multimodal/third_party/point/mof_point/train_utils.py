from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    target_mean = float(np.mean(targets))
    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(targets - target_mean)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

