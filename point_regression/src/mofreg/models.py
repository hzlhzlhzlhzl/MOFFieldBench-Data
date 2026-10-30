from __future__ import annotations

import torch
from torch import nn

from mofreg.repo_models import PointMambaRegressor, X3DRegressor


class PointNetRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        del xyz
        encoded = self.point_mlp(features)
        pooled_max = encoded.max(dim=1).values
        pooled_mean = encoded.mean(dim=1)
        fused = torch.cat([pooled_max, pooled_mean], dim=-1)
        return self.head(fused).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_layers: int,
        num_heads: int,
        token_count: int,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        del xyz
        tokens = self._downsample(features)
        hidden = self.input_proj(tokens)
        hidden = self.encoder(hidden)
        weights = torch.softmax(self.attn_pool(hidden), dim=1)
        pooled = (hidden * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)

    def _downsample(self, features: torch.Tensor) -> torch.Tensor:
        num_points = features.shape[1]
        if num_points <= self.token_count:
            return features
        idx = torch.linspace(
            0,
            num_points - 1,
            steps=self.token_count,
            device=features.device,
        ).round().long()
        return features.index_select(1, idx)


def build_model(config: dict, input_dim: int) -> nn.Module:
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    if name == "pointnet":
        return PointNetRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
    if name == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            token_count=int(model_cfg["token_count"]),
        )
    if name == "pointmamba":
        return PointMambaRegressor(
            input_dim=input_dim,
            num_group=int(model_cfg.get("num_group", 64)),
            group_size=int(model_cfg.get("group_size", 32)),
            trans_dim=int(model_cfg.get("trans_dim", 384)),
            depth=int(model_cfg.get("depth", 12)),
            drop_path=float(model_cfg.get("drop_path", 0.1)),
            drop_out=float(model_cfg.get("drop_out", 0.0)),
            rms_norm=bool(model_cfg.get("rms_norm", False)),
            avg_head=bool(model_cfg.get("avg_head", True)),
            max_head=bool(model_cfg.get("max_head", False)),
        )
    if name == "x3d":
        return X3DRegressor(
            input_dim=input_dim,
            width=int(model_cfg.get("width", 32)),
            blocks=list(model_cfg.get("blocks", [1, 2, 2, 2, 1])),
            strides=list(model_cfg.get("strides", [1, 4, 4, 4, 4])),
            nsample=int(model_cfg.get("nsample", 16)),
            radius=float(model_cfg.get("radius", 0.2)),
            aggr_feature_type=str(model_cfg.get("aggr_feature_type", "dp_fj")),
            expansion=int(model_cfg.get("expansion", 4)),
            sa_layers=int(model_cfg.get("sa_layers", 1)),
            sa_use_res=bool(model_cfg.get("sa_use_res", False)),
            x3d_layers=list(model_cfg.get("x3d_layers", [False, False, False, True, True])),
            sampler=str(model_cfg.get("sampler", "fps")),
            fps_backend=str(model_cfg.get("fps_backend", "linspace")),
        )
    raise ValueError(f"Unsupported model name: {model_cfg['name']}")
