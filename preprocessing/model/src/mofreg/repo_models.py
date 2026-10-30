from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


LOCAL_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = LOCAL_ROOT / "vendor"
X3D_ROOT = VENDOR_ROOT


def _ensure_x3d_repo_paths() -> None:
    paths = [LOCAL_ROOT, VENDOR_ROOT]
    for extension in ("pointnet2_batch", "pointops"):
        root = VENDOR_ROOT / "openpoints" / "cpp" / extension
        paths.append(root)
        paths.extend(sorted((root / "build").glob("lib.*")))
    for path in reversed(paths):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _configure_x3d_sampler(sampler: str, fps_backend: str) -> None:
    import openpoints.models.backbone.pointmetabase_X3D as pointmetabase_x3d
    import openpoints.models.layers.subsample as subsample

    sampler = sampler.lower()
    fps_backend = fps_backend.lower()
    if sampler not in {"fps", "random"}:
        raise ValueError(f"Unsupported X3D sampler: {sampler}")
    if fps_backend not in {"native", "linspace"}:
        raise ValueError(f"Unsupported X3D FPS backend: {fps_backend}")

    fps_fn = (
        subsample.FurthestPointSampling.apply
        if fps_backend == "native"
        else _safe_x3d_sample
    )
    # The backbone imports this symbol directly, so both module references must agree.
    subsample.furthest_point_sample = fps_fn
    pointmetabase_x3d.furthest_point_sample = fps_fn


class X3DRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        blocks: list[int],
        strides: list[int],
        nsample: int,
        radius: float,
        aggr_feature_type: str,
        expansion: int,
        sa_layers: int,
        sa_use_res: bool,
        x3d_layers: list[bool],
        sampler: str,
        fps_backend: str,
    ) -> None:
        super().__init__()
        _ensure_x3d_repo_paths()
        from easydict import EasyDict
        from openpoints.models.backbone.pointmetabase_X3D import PointMetaBaseEncoder_X3D

        _configure_x3d_sampler(sampler, fps_backend)

        structure_kernel_args = EasyDict(
            hidden_dim=max(width, 32),
            modu="true",
            denoise="false",
        )
        group_args = EasyDict(NAME="ballquery", radius=radius, nsample=nsample)
        aggr_args = {
            "feature_type": aggr_feature_type,
            "reduction": "max",
            "structure_kernel_args": structure_kernel_args,
            "es_type": "pca",
        }
        self.backbone = PointMetaBaseEncoder_X3D(
            in_channels=input_dim,
            width=width,
            blocks=blocks,
            strides=strides,
            nsample=nsample,
            radius=radius,
            aggr_args=aggr_args,
            group_args=group_args,
            expansion=expansion,
            sa_layers=sa_layers,
            sa_use_res=sa_use_res,
            x3d_layers=x3d_layers,
            norm_args={"norm": "bn"},
            act_args={"act": "relu"},
            conv_args={},
            sampler=sampler,
            use_res=True,
            NCP="false",
        )
        out_dim = int(self.backbone.out_channels)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, 1),
        )

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        _, feature_pyramid = self.backbone.forward_seg_feat(xyz, features.transpose(1, 2).contiguous())
        encoded = feature_pyramid[-1]
        if encoded.ndim == 3:
            pooled_max = encoded.max(dim=-1).values
            pooled_mean = encoded.mean(dim=-1)
            encoded = 0.5 * (pooled_max + pooled_mean)
        return self.head(encoded).squeeze(-1)
