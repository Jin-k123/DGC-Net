"""
DGC-Net: Dual-Granularity Cross-Stage Calibration Network.

The model contains three paper-aligned components:

1. Dual-Granularity Representation Encoding (DGRE)
2. Progressive Stage-Coupled Fusion (PSCF)
3. Bilateral Gated Semantic Calibration (BGSC)

The structural-granularity branch uses LWGANet, while the
detail-granularity branch uses a four-stage convolutional encoder.
PSCF and BGSC are imported from independent module files.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BGSC import BilateralGatedSemanticCalibration
from .common import DropPath, LayerNorm2d
from .PSCF import ProgressiveStageCoupledFusion


def build_lwganet_backbone() -> nn.Module:
    """
    Build the structural-granularity LWGANet backbone.

    The repository root must be available on PYTHONPATH and the official
    LWGANet implementation must be placed under:

        model.LWGA.py
    """

    try:
        from model.LWGA import (
            LWGANet_L1_1242_e64_k11_GELU,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Unable to import the LWGANet backbone. Place the official "
            "LWGANet implementation under "
            "'model.LWGA.py', or pass "
            "a custom structural_encoder to DGCNet."
        ) from error

    return LWGANet_L1_1242_e64_k11_GELU(fork_feat=True)


class LocalDetailBlock(nn.Module):
    """Residual local-detail block used by the detail branch."""

    def __init__(
        self,
        dim: int,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=True,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pointwise(x)
        x = self.activation(x)
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class DetailGranularityBranch(nn.Module):
    """Four-stage encoder for local cytological details."""

    def __init__(
        self,
        in_channels: int = 3,
        stage_dims: Sequence[int] = (96, 192, 384, 768),
        stage_depths: Sequence[int] = (3, 4, 6, 3),
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if len(stage_dims) != 4 or len(stage_depths) != 4:
            raise ValueError(
                "The detail-granularity branch requires four stages."
            )

        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        stage_dims[0],
                        kernel_size=4,
                        stride=4,
                        bias=True,
                    ),
                    LayerNorm2d(stage_dims[0]),
                )
            ]
        )

        for index in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(stage_dims[index]),
                    nn.Conv2d(
                        stage_dims[index],
                        stage_dims[index + 1],
                        kernel_size=2,
                        stride=2,
                        bias=True,
                    ),
                )
            )

        rates = torch.linspace(
            0.0,
            drop_path_rate,
            sum(stage_depths),
        ).tolist()

        self.stages = nn.ModuleList()
        offset = 0

        for dim, depth in zip(stage_dims, stage_depths):
            stage = nn.Sequential(
                *[
                    LocalDetailBlock(
                        dim=dim,
                        drop_path_rate=rates[offset + block_index],
                    )
                    for block_index in range(depth)
                ]
            )
            self.stages.append(stage)
            offset += depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = []

        for downsample, stage in zip(
            self.downsample_layers,
            self.stages,
        ):
            x = downsample(x)
            x = stage(x)
            features.append(x)

        return tuple(features)


class DGCNet(nn.Module):
    """
    Dual-Granularity Cross-Stage Calibration Network.

    Args:
        num_classes:
            Number of output classes.
        in_channels:
            Number of input image channels.
        structural_encoder:
            Optional custom structural encoder. It must return four NCHW
            feature maps. When omitted, the LWGANet backbone is constructed.
        structural_dims:
            Channel dimensions returned by the structural encoder.
        stage_dims:
            Channel dimensions shared by the aligned structural features,
            detail features, PSCF outputs, and BGSC.
        detail_depths:
            Number of local-detail blocks in the four detail stages.
        detail_drop_path_rate:
            Maximum stochastic-depth rate in the detail branch.
        pscf_reduction:
            Channel reduction ratio used by PSCF calibration.
        pscf_drop_path_rate:
            Maximum stochastic-depth rate used by PSCF.
        pscf_refinement_dropout:
            Dropout rate used by PSCF residual calibration.
        bgsc_num_heads:
            Number of heads used by bilateral cross-attention.
        bgsc_attention_dropout:
            Dropout rate used by bilateral cross-attention.
        bgsc_gate_reduction:
            Channel reduction ratio used by BGSC gates.
        classifier_dropout:
            Dropout rate in the BGSC classification head.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        structural_encoder: nn.Module | None = None,
        structural_dims: Sequence[int] = (64, 128, 256, 512),
        stage_dims: Sequence[int] = (96, 192, 384, 768),
        detail_depths: Sequence[int] = (3, 4, 6, 3),
        detail_drop_path_rate: float = 0.1,
        pscf_reduction: int = 16,
        pscf_drop_path_rate: float = 0.0,
        pscf_refinement_dropout: float = 0.0,
        bgsc_num_heads: int = 8,
        bgsc_attention_dropout: float = 0.1,
        bgsc_gate_reduction: int = 4,
        bgsc_pooling_hidden_dim: int = 256,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if len(structural_dims) != 4:
            raise ValueError("structural_dims must contain four values.")
        if len(stage_dims) != 4:
            raise ValueError("stage_dims must contain four values.")
        if stage_dims[-1] % bgsc_num_heads != 0:
            raise ValueError(
                "The final stage dimension must be divisible by "
                "bgsc_num_heads."
            )

        self.num_classes = num_classes
        self.structural_dims = tuple(structural_dims)
        self.stage_dims = tuple(stage_dims)

        self.structural_encoder = (
            structural_encoder
            if structural_encoder is not None
            else build_lwganet_backbone()
        )

        self.detail_encoder = DetailGranularityBranch(
            in_channels=in_channels,
            stage_dims=stage_dims,
            stage_depths=detail_depths,
            drop_path_rate=detail_drop_path_rate,
        )

        self.structural_projections = nn.ModuleList(
            [
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=1,
                    bias=True,
                )
                for input_dim, output_dim in zip(
                    structural_dims,
                    stage_dims,
                )
            ]
        )

        self.pscf = ProgressiveStageCoupledFusion(
            stage_dims=stage_dims,
            reduction=pscf_reduction,
            drop_path_rate=pscf_drop_path_rate,
            refinement_dropout=pscf_refinement_dropout,
        )

        self.bgsc = BilateralGatedSemanticCalibration(
            stage_dims=stage_dims,
            dim=stage_dims[-1],
            num_classes=num_classes,
            num_heads=bgsc_num_heads,
            attention_dropout=bgsc_attention_dropout,
            gate_reduction=bgsc_gate_reduction,
            pooling_hidden_dim=bgsc_pooling_hidden_dim,
            classifier_dropout=classifier_dropout,
        )

    @staticmethod
    def _validate_feature_hierarchy(
        features: Sequence[torch.Tensor],
        branch_name: str,
    ) -> Tuple[torch.Tensor, ...]:
        features = tuple(features)

        if len(features) != 4:
            raise ValueError(
                f"{branch_name} must return four feature maps, "
                f"but returned {len(features)}."
            )

        for stage_index, feature in enumerate(features):
            if not isinstance(feature, torch.Tensor):
                raise TypeError(
                    f"{branch_name} stage {stage_index + 1} is not a tensor."
                )
            if feature.ndim != 4:
                raise ValueError(
                    f"{branch_name} stage {stage_index + 1} must be NCHW."
                )

        return features

    def encode_dual_granularity(
        self,
        x: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        Tuple[torch.Tensor, ...],
    ]:
        """Extract and align structural and detail feature hierarchies."""

        structural_features = self._validate_feature_hierarchy(
            self.structural_encoder(x),
            "Structural encoder",
        )
        detail_features = self._validate_feature_hierarchy(
            self.detail_encoder(x),
            "Detail encoder",
        )

        aligned_structural_features = []

        for stage_index, (
            structural_feature,
            detail_feature,
            projection,
        ) in enumerate(
            zip(
                structural_features,
                detail_features,
                self.structural_projections,
            )
        ):
            expected_channels = self.structural_dims[stage_index]

            if structural_feature.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Structural stage {stage_index + 1} returned "
                    f"{structural_feature.shape[1]} channels, but "
                    f"{expected_channels} were configured."
                )

            structural_feature = projection(structural_feature)

            if structural_feature.shape[-2:] != detail_feature.shape[-2:]:
                structural_feature = F.interpolate(
                    structural_feature,
                    size=detail_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            if structural_feature.shape[1] != detail_feature.shape[1]:
                raise RuntimeError(
                    f"Stage {stage_index + 1} structural and detail "
                    "features have inconsistent channel dimensions."
                )

            aligned_structural_features.append(structural_feature)

        return tuple(aligned_structural_features), detail_features

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        """Return all feature hierarchies before final classification."""

        structural_features, detail_features = (
            self.encode_dual_granularity(x)
        )
        fused_features = self.pscf(
            structural_features=structural_features,
            detail_features=detail_features,
        )

        return {
            "structural_features": structural_features,
            "detail_features": detail_features,
            "fused_features": fused_features,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ):
        features = self.forward_features(x)

        output = self.bgsc(
            structural_feature=features["structural_features"][-1],
            detail_feature=features["detail_features"][-1],
            fused_features=features["fused_features"],
            return_aux=return_aux,
        )

        if not return_aux:
            return output

        output.update(features)
        return output


# Backward-compatible class name for existing training scripts.
DGC_Net = DGCNet


def build_dgc_net(
    num_classes: int = 6,
    **kwargs,
) -> DGCNet:
    """Build a DGC-Net model."""

    return DGCNet(
        num_classes=num_classes,
        **kwargs,
    )


__all__ = [
    "DGCNet",
    "DGC_Net",
    "DetailGranularityBranch",
    "LocalDetailBlock",
    "build_dgc_net",
]
