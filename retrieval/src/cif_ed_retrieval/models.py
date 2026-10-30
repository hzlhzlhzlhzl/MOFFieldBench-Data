from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor.cgcnn.model import CrystalGraphConvNet
from .x3d import X3DBackbone


class CGCNNEncoder(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int,
        n_conv: int,
        h_fea_len: int,
        n_h: int,
    ) -> None:
        super().__init__()
        self.backbone = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            classification=False,
        )
        self.output_dim = h_fea_len

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: list[torch.Tensor],
    ) -> torch.Tensor:
        hidden = self.backbone.embedding(atom_fea)
        for conv in self.backbone.convs:
            hidden = conv(hidden, nbr_fea, nbr_fea_idx)
        crystal = self.backbone.pooling(hidden, crystal_atom_idx)
        crystal = self.backbone.conv_to_fc(self.backbone.conv_to_fc_softplus(crystal))
        crystal = self.backbone.conv_to_fc_softplus(crystal)
        if hasattr(self.backbone, "fcs") and hasattr(self.backbone, "softpluses"):
            for fc, softplus in zip(self.backbone.fcs, self.backbone.softpluses):
                crystal = softplus(fc(crystal))
        return crystal


class X3DEncoder(nn.Module):
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
        super().__init__()
        self.backbone = X3DBackbone(
            input_dim=input_dim,
            width=width,
            blocks=blocks,
            strides=strides,
            nsample=nsample,
            radius=radius,
            aggr_feature_type=aggr_feature_type,
            expansion=expansion,
            sa_layers=sa_layers,
            sa_use_res=sa_use_res,
            x3d_layers=x3d_layers,
        )
        self.output_dim = int(self.backbone.out_channels)

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        _, feature_pyramid = self.backbone.forward_seg_feat(
            xyz, features.transpose(1, 2).contiguous()
        )
        encoded = feature_pyramid[-1]
        if encoded.ndim == 3:
            pooled_max = encoded.max(dim=-1).values
            pooled_mean = encoded.mean(dim=-1)
            encoded = 0.5 * (pooled_max + pooled_mean)
        return encoded


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(input_dim, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CIFEDRetrievalModel(nn.Module):
    def __init__(self, config: dict, orig_atom_fea_len: int, nbr_fea_len: int, ed_input_dim: int) -> None:
        super().__init__()
        cif_cfg = config["model"]["cif_encoder"]
        ed_cfg = config["model"].get("point_encoder")
        if ed_cfg is None:
            ed_cfg = config["model"]["ed_encoder"]
        proj_cfg = config["model"]["projection"]

        self.cif_encoder = CGCNNEncoder(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=int(cif_cfg["atom_fea_len"]),
            n_conv=int(cif_cfg["n_conv"]),
            h_fea_len=int(cif_cfg["h_fea_len"]),
            n_h=int(cif_cfg["n_h"]),
        )
        self.ed_encoder = X3DEncoder(
            input_dim=ed_input_dim,
            width=int(ed_cfg["width"]),
            blocks=list(ed_cfg["blocks"]),
            strides=list(ed_cfg["strides"]),
            nsample=int(ed_cfg["nsample"]),
            radius=float(ed_cfg["radius"]),
            aggr_feature_type=str(ed_cfg["aggr_feature_type"]),
            expansion=int(ed_cfg["expansion"]),
            sa_layers=int(ed_cfg["sa_layers"]),
            sa_use_res=bool(ed_cfg["sa_use_res"]),
            x3d_layers=list(ed_cfg["x3d_layers"]),
        )
        self.cif_proj = ProjectionHead(
            input_dim=self.cif_encoder.output_dim,
            embedding_dim=int(proj_cfg["embedding_dim"]),
            dropout=float(proj_cfg["dropout"]),
        )
        self.ed_proj = ProjectionHead(
            input_dim=self.ed_encoder.output_dim,
            embedding_dim=int(proj_cfg["embedding_dim"]),
            dropout=float(proj_cfg["dropout"]),
        )

    def forward(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = batch["cif"]
        crystal_atom_idx = [idx.to(atom_fea.device) for idx in crystal_atom_idx]
        cif_hidden = self.cif_encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        ed_hidden = self.ed_encoder(batch["point_features"], batch["point_xyz"])
        return self.cif_proj(cif_hidden), self.ed_proj(ed_hidden)


class EDESPRetrievalModel(nn.Module):
    """Two X3D encoders for cross-modal ED-to-ESP retrieval."""

    def __init__(self, config: dict, ed_input_dim: int, esp_input_dim: int) -> None:
        super().__init__()
        encoder_cfg = config["model"].get("point_encoder", config["model"].get("ed_encoder"))
        proj_cfg = config["model"]["projection"]

        def make_encoder(input_dim: int) -> X3DEncoder:
            return X3DEncoder(
                input_dim=input_dim,
                width=int(encoder_cfg["width"]),
                blocks=list(encoder_cfg["blocks"]),
                strides=list(encoder_cfg["strides"]),
                nsample=int(encoder_cfg["nsample"]),
                radius=float(encoder_cfg["radius"]),
                aggr_feature_type=str(encoder_cfg["aggr_feature_type"]),
                expansion=int(encoder_cfg["expansion"]),
                sa_layers=int(encoder_cfg["sa_layers"]),
                sa_use_res=bool(encoder_cfg["sa_use_res"]),
                x3d_layers=list(encoder_cfg["x3d_layers"]),
            )

        self.ed_encoder = make_encoder(ed_input_dim)
        self.esp_encoder = make_encoder(esp_input_dim)
        embedding_dim = int(proj_cfg["embedding_dim"])
        dropout = float(proj_cfg["dropout"])
        self.ed_proj = ProjectionHead(self.ed_encoder.output_dim, embedding_dim, dropout)
        self.esp_proj = ProjectionHead(self.esp_encoder.output_dim, embedding_dim, dropout)

    def forward(self, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
        ed_hidden = self.ed_encoder(batch["ed_features"], batch["ed_xyz"])
        esp_hidden = self.esp_encoder(batch["esp_features"], batch["esp_xyz"])
        return self.ed_proj(ed_hidden), self.esp_proj(esp_hidden)


def contrastive_loss(cif_z: torch.Tensor, ed_z: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    logits = cif_z @ ed_z.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_c2e = F.cross_entropy(logits, targets)
    loss_e2c = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_c2e + loss_e2c), logits
