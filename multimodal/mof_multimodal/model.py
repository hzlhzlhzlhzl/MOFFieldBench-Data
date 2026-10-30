from __future__ import annotations

import sys
import types
import uuid
from pathlib import Path

import torch
from torch import nn


def _add_model_paths(config: dict) -> None:
    point_root = str(Path(config["paths"]["point_project"]).resolve())
    cif_vendor = str(Path(config["paths"]["cif_project"]).resolve() / "vendor")
    for path in (cif_vendor, point_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    # The mofesp environment has all numerical X3D dependencies but omits two
    # config-only packages imported by OpenPoints during package initialization.
    # X3D construction below does not use either API, so lightweight import
    # shims keep the original point model usable without modifying the env.
    try:
        import multimethod  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("multimethod")
        module.multimethod = lambda function: function
        sys.modules["multimethod"] = module
    try:
        import yaml  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("yaml")
        module.safe_load = lambda stream: {}
        sys.modules["yaml"] = module
    try:
        import shortuuid  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("shortuuid")
        module.uuid = lambda: uuid.uuid4().hex
        sys.modules["shortuuid"] = module


class CIFEncoder(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.output_dim = int(model.fc_out.in_features)
        self.model.fc_out = nn.Identity()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        model = self.model
        atom_fea = model.embedding(atom_fea)
        for conv in model.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        encoded = model.pooling(atom_fea, crystal_atom_idx)
        encoded = model.conv_to_fc(model.conv_to_fc_softplus(encoded))
        encoded = model.conv_to_fc_softplus(encoded)
        if hasattr(model, "fcs") and hasattr(model, "softpluses"):
            for fc, softplus in zip(model.fcs, model.softpluses):
                encoded = softplus(fc(encoded))
        return encoded


class PointEncoder(nn.Module):
    def __init__(self, model: nn.Module, name: str) -> None:
        super().__init__()
        self.model = model
        self.name = name.lower()
        if self.name == "pointnet":
            self.output_dim = int(model.head[0].in_features)
            model.head = nn.Identity()
        elif self.name == "transformer":
            self.output_dim = int(model.head[1].in_features)
            model.head = nn.Identity()
        elif self.name == "x3d":
            self.output_dim = int(model.backbone.out_channels)
            model.head = nn.Identity()
        else:
            raise ValueError("Fusion currently supports pointnet, transformer, or x3d")

    def forward(self, features: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        if self.name == "pointnet":
            hidden = self.model.point_mlp(features)
            return torch.cat([hidden.max(dim=1).values, hidden.mean(dim=1)], dim=-1)
        if self.name == "transformer":
            tokens = self.model._downsample(features)
            hidden = self.model.encoder(self.model.input_proj(tokens))
            weights = torch.softmax(self.model.attn_pool(hidden), dim=1)
            return (hidden * weights).sum(dim=1)
        _, pyramid = self.model.backbone.forward_seg_feat(
            xyz, features.transpose(1, 2).contiguous()
        )
        encoded = pyramid[-1]
        if encoded.ndim == 3:
            encoded = 0.5 * (encoded.max(dim=-1).values + encoded.mean(dim=-1))
        return encoded


class MultimodalRegressor(nn.Module):
    def __init__(
        self,
        cif_encoder: CIFEncoder,
        point_encoder: PointEncoder,
        secondary_point_encoder: PointEncoder | None,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
        fusion_type: str,
    ) -> None:
        super().__init__()
        self.cif_encoder = cif_encoder
        self.point_encoder = point_encoder
        self.secondary_point_encoder = secondary_point_encoder
        self.fusion_type = fusion_type.lower()
        self.cif_projection = nn.Sequential(
            nn.LayerNorm(cif_encoder.output_dim), nn.Linear(cif_encoder.output_dim, projection_dim), nn.GELU()
        )
        self.point_projection = nn.Sequential(
            nn.LayerNorm(point_encoder.output_dim), nn.Linear(point_encoder.output_dim, projection_dim), nn.GELU()
        )
        self.secondary_point_projection = None
        if secondary_point_encoder is not None:
            self.secondary_point_projection = nn.Sequential(
                nn.LayerNorm(secondary_point_encoder.output_dim), nn.Linear(secondary_point_encoder.output_dim, projection_dim), nn.GELU()
            )
        # Modality-neutral fusion alternatives.  CIF, ESP, and ED are tokens
        # in one set; neither CIF nor a field branch is a residual anchor.
        if self.fusion_type in {"symmetric_gated", "attention_gated"}:
            self.num_modalities = 3 if secondary_point_encoder is not None else 2
            self.cif_head = None
            self.gate = None
            if self.fusion_type == "symmetric_gated":
                # Apply a shared scoring function to each token, then normalize
                # across modalities; changing token order does not favor CIF.
                self.symmetric_gate = nn.Sequential(
                    nn.LayerNorm(projection_dim), nn.Linear(projection_dim, 1)
                )
            else:
                # Pool modality tokens with a learned query, independent of
                # every input modality (four heads for the 128-D embedding).
                self.attention_query = nn.Parameter(torch.empty(1, 1, projection_dim))
                nn.init.normal_(self.attention_query, std=0.02)
                self.modal_attention = nn.MultiheadAttention(
                    projection_dim, num_heads=4, dropout=dropout, batch_first=True
                )
                self.attention_norm = nn.LayerNorm(projection_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            return
        if secondary_point_encoder is not None:
            self.tri_gate = nn.Sequential(nn.Linear(projection_dim * 3, projection_dim), nn.Sigmoid())
            self.tri_head = nn.Sequential(
                nn.LayerNorm(projection_dim * 4), nn.Linear(projection_dim * 4, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.tri_head[-1].weight)
            nn.init.zeros_(self.tri_head[-1].bias)
            return
        if self.fusion_type == "cif_only":
            self.gate = None
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim),
                nn.Linear(projection_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.head = None
            return
        self.cif_head = None
        if self.fusion_type == "concat":
            fused_dim = projection_dim * 2
            self.gate = None
        elif self.fusion_type in {"gated", "residual_gated"}:
            self.gate = nn.Sequential(nn.Linear(projection_dim * 2, projection_dim), nn.Sigmoid())
            fused_dim = projection_dim * 3
        elif self.fusion_type in {"residual_add", "residual_film", "residual_point", "residual_cross"}:
            # `head` is unused for these late-fusion variants but is kept as
            # a harmless module so the constructor remains uniform.
            self.gate = None
            fused_dim = projection_dim * 2
        else:
            raise ValueError("Unsupported fusion type")
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # A residual fusion starts as a CIF predictor.  This prevents a weak
        # point modality from degrading the strong CIF representation during
        # the first optimisation steps, while still allowing it to learn a
        # sample-dependent correction.
        if self.fusion_type == "residual_gated":
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim),
                nn.Linear(projection_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        elif self.fusion_type == "residual_add":
            # Late additive correction: CIF produces the main prediction and
            # an independent point-conditioned MLP learns only its residual.
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.residual_head = nn.Sequential(
                nn.LayerNorm(projection_dim * 2), nn.Linear(projection_dim * 2, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        elif self.fusion_type == "residual_film":
            # ESP generates a bounded FiLM modulation of CIF features before
            # the residual correction; this is less destructive than direct
            # feature concatenation.
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.film = nn.Sequential(nn.Linear(projection_dim, projection_dim * 2), nn.Tanh())
            self.residual_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        elif self.fusion_type == "residual_point":
            # Strictly late fusion: ESP can only add a correction to the CIF
            # prediction, so it cannot overwrite structural information.
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.residual_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        elif self.fusion_type == "residual_cross":
            # Product and difference carry agreement/disagreement between the
            # two modalities while retaining CIF as an explicit base term.
            self.cif_head = nn.Sequential(
                nn.LayerNorm(projection_dim), nn.Linear(projection_dim, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            self.residual_head = nn.Sequential(
                nn.LayerNorm(projection_dim * 3), nn.Linear(projection_dim * 3, hidden_dim),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, features, xyz, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, secondary_features=None, secondary_xyz=None):
        cif = self.cif_projection(
            self.cif_encoder(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        )
        if self.fusion_type == "cif_only":
            return self.cif_head(cif).squeeze(-1)
        point = self.point_projection(self.point_encoder(features, xyz))
        if self.fusion_type in {"symmetric_gated", "attention_gated"}:
            tokens = [cif, point]
            if self.secondary_point_encoder is not None:
                tokens.append(self.secondary_point_projection(
                    self.secondary_point_encoder(secondary_features, secondary_xyz)
                ))
            tokens = torch.stack(tokens, dim=1)
            if self.fusion_type == "symmetric_gated":
                weights = torch.softmax(self.symmetric_gate(tokens), dim=1)
                fused = (weights * tokens).sum(dim=1)
            else:
                query = self.attention_query.expand(tokens.shape[0], -1, -1)
                fused, _ = self.modal_attention(query, tokens, tokens, need_weights=False)
                fused = self.attention_norm(fused.squeeze(1))
            return self.head(fused).squeeze(-1)
        if self.secondary_point_encoder is not None:
            secondary = self.secondary_point_projection(self.secondary_point_encoder(secondary_features, secondary_xyz))
            gate = self.tri_gate(torch.cat([cif, point, secondary], dim=-1))
            point_mix = gate * point + (1.0 - gate) * secondary
            correction = self.tri_head(torch.cat([point_mix, cif * point, cif * secondary, point - secondary], dim=-1))
            return (self.cif_head(cif).squeeze(-1) + correction.squeeze(-1))
        if self.fusion_type == "residual_add":
            return (self.cif_head(cif) + self.residual_head(torch.cat([cif, point], dim=-1))).squeeze(-1)
        if self.fusion_type == "residual_film":
            gamma, beta = self.film(point).chunk(2, dim=-1)
            conditioned = cif * (1.0 + 0.1 * gamma) + 0.1 * beta
            return (self.cif_head(cif) + self.residual_head(conditioned)).squeeze(-1)
        if self.fusion_type == "residual_point":
            return (self.cif_head(cif) + self.residual_head(point)).squeeze(-1)
        if self.fusion_type == "residual_cross":
            cross = torch.cat([point, cif * point, cif - point], dim=-1)
            return (self.cif_head(cif) + self.residual_head(cross)).squeeze(-1)
        if self.fusion_type == "concat":
            fused = torch.cat([cif, point], dim=-1)
        else:
            gate = self.gate(torch.cat([cif, point], dim=-1))
            cif_gated, point_gated = gate * cif, (1.0 - gate) * point
            fused = torch.cat([cif_gated, point_gated, cif * point], dim=-1)
        prediction = self.head(fused).squeeze(-1)
        if self.fusion_type == "residual_gated":
            prediction = prediction + self.cif_head(cif).squeeze(-1)
        return prediction

def build_multimodal_model(config: dict, input_dim: int, cif_sample) -> MultimodalRegressor:
    _add_model_paths(config)
    from cgcnn.model import CrystalGraphConvNet
    from mof_point.models import build_model

    structures, _, _ = cif_sample
    cif_cfg = config["cif_model"]
    cif_model = CrystalGraphConvNet(
        structures[0].shape[-1],
        structures[1].shape[-1],
        atom_fea_len=int(cif_cfg["atom_fea_len"]),
        n_conv=int(cif_cfg["n_conv"]),
        h_fea_len=int(cif_cfg["h_fea_len"]),
        n_h=int(cif_cfg["n_h"]),
        classification=False,
    )
    point_cfg = {"model": dict(config["point_model"])}
    point_model = build_model(point_cfg, input_dim=input_dim)
    secondary_encoder = None
    if config["data"].get("secondary_point_pack_path"):
        secondary_cfg = {"model": dict(config["point_model"])}
        secondary_input_dim = int(config["data"].get("secondary_input_dim", input_dim))
        secondary_encoder = PointEncoder(build_model(secondary_cfg, input_dim=secondary_input_dim), config["point_model"]["name"])

    fusion_cfg = config["fusion"]
    return MultimodalRegressor(
        CIFEncoder(cif_model),
        PointEncoder(point_model, config["point_model"]["name"]),
        secondary_encoder,
        projection_dim=int(fusion_cfg["projection_dim"]),
        hidden_dim=int(fusion_cfg["hidden_dim"]),
        dropout=float(fusion_cfg["dropout"]),
        fusion_type=str(fusion_cfg["type"]),
    )
