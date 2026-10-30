from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


LOCAL_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = LOCAL_ROOT / "vendor"
POINTMAMBA_ROOT = VENDOR_ROOT / "PointMamba-main"
POINTMAMBA_MAMBA_BUILD = POINTMAMBA_ROOT / "mamba" / "build" / "lib.linux-x86_64-cpython-39"
X3D_ROOT = VENDOR_ROOT
X3D_POINTOPS_BUILD = X3D_ROOT / "openpoints" / "cpp" / "pointops" / "build" / "lib.linux-x86_64-cpython-39"
X3D_POINTNET2_BUILD = X3D_ROOT / "openpoints" / "cpp" / "pointnet2_batch" / "build" / "lib.linux-x86_64-cpython-39"


def _ensure_repo_paths() -> None:
    local_text = str(LOCAL_ROOT)
    if local_text in sys.path:
        sys.path.remove(local_text)
    sys.path.insert(0, local_text)

    repo_text = str(POINTMAMBA_ROOT)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(1, repo_text)

    build_text = str(POINTMAMBA_MAMBA_BUILD)
    if build_text in sys.path:
        sys.path.remove(build_text)
    sys.path.append(build_text)


def _ensure_x3d_repo_paths() -> None:
    local_text = str(LOCAL_ROOT)
    if local_text in sys.path:
        sys.path.remove(local_text)
    sys.path.insert(0, local_text)

    repo_text = str(X3D_ROOT)
    if repo_text in sys.path:
        sys.path.remove(repo_text)
    sys.path.insert(1, repo_text)

    pointops_build_text = str(X3D_POINTOPS_BUILD)
    if pointops_build_text in sys.path:
        sys.path.remove(pointops_build_text)
    sys.path.insert(2, pointops_build_text)

    pointnet2_build_text = str(X3D_POINTNET2_BUILD)
    if pointnet2_build_text in sys.path:
        sys.path.remove(pointnet2_build_text)
    sys.path.insert(3, pointnet2_build_text)


def _safe_x3d_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    if npoint >= num_points:
        return torch.arange(num_points, device=xyz.device).unsqueeze(0).expand(batch_size, -1)
    idx = torch.linspace(
        0,
        num_points - 1,
        steps=npoint,
        device=xyz.device,
    ).round().long()
    return idx.unsqueeze(0).expand(batch_size, -1).contiguous()


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


class PointMambaRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_group: int,
        group_size: int,
        trans_dim: int,
        depth: int,
        drop_path: float,
        drop_out: float,
        rms_norm: bool,
        avg_head: bool,
        max_head: bool,
    ) -> None:
        super().__init__()
        del input_dim
        _ensure_repo_paths()
        from models.point_mamba_scan import PointMambaScan

        config = SimpleNamespace(
            trans_dim=trans_dim,
            depth=depth,
            cls_dim=1,
            num_heads=6,
            group_size=group_size,
            num_group=num_group,
            encoder_dims=trans_dim,
            rms_norm=rms_norm,
            drop_path=drop_path,
            drop_out=drop_out,
            use_cls_token=False,
            max_head=max_head,
            avg_head=avg_head,
        )
        self.backbone = PointMambaScan(config)

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        del features
        return self.backbone(xyz).squeeze(-1)


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
