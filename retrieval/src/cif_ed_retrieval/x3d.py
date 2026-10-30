from __future__ import annotations

from easydict import EasyDict
from .vendor.openpoints.models.backbone.pointmetabase_X3D import PointMetaBaseEncoder_X3D


class X3DBackbone(PointMetaBaseEncoder_X3D):
    """Project-local construction wrapper for the vendored X3D backbone."""

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
    ) -> None:
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
        super().__init__(
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
            sampler="fps",
            use_res=True,
            NCP="false",
        )
