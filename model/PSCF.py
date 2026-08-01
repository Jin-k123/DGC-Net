"""
Progressive Stage-Coupled Fusion (PSCF).

Each stage receives the aligned structural feature, the detail feature,
and, except for the first stage, the fused representation from the
preceding stage. A residual calibration unit is applied after every
fusion stage, matching the DGC-Net paper description.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DropPath, LayerNorm2d, make_group_norm


class SpatialCalibration(nn.Module):
    """Generate a spatial gate for a detail-granularity feature."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()

        self.projection = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maximum = torch.amax(x, dim=1, keepdim=True)
        average = torch.mean(x, dim=1, keepdim=True)
        gate = torch.sigmoid(
            self.projection(torch.cat([maximum, average], dim=1))
        )
        return gate * x


class ChannelCalibration(nn.Module):
    """Generate a channel gate for a structural-granularity feature."""

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        hidden_dim = max(dim // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                dim,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(x, output_size=1)
        maximum = F.adaptive_max_pool2d(x, output_size=1)
        gate = torch.sigmoid(
            self.mlp(average) + self.mlp(maximum)
        )
        return gate * x


class InvertedResidualMLP(nn.Module):
    """Inverted residual projection used by a fusion stage."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_ratio: int = 4,
    ) -> None:
        super().__init__()

        hidden_dim = input_dim * expansion_ratio

        self.depthwise = nn.Conv2d(
            input_dim,
            input_dim,
            kernel_size=3,
            padding=1,
            groups=input_dim,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(input_dim)
        self.expand = nn.Conv2d(
            input_dim,
            hidden_dim,
            kernel_size=1,
            bias=False,
        )
        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                output_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.depthwise(x)
        x = self.activation(x)
        x = x + residual
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)

        return x


class ResidualCalibrationUnit(nn.Module):
    """
    Lightweight residual calibration unit.

    R(U) = Drop(GELU(GN(Conv1x1(DWConv3x3(U)))))
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
                bias=False,
            ),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(dim),
            nn.GELU(),
            nn.Dropout2d(dropout)
            if dropout > 0.0
            else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StageCoupledFusionBlock(nn.Module):
    """One stage of progressive structural-detail fusion."""

    def __init__(
        self,
        dim: int,
        previous_dim: int | None = None,
        reduction: int = 16,
        drop_path_rate: float = 0.0,
        refinement_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.previous_projection = (
            nn.Sequential(
                nn.Conv2d(
                    previous_dim,
                    dim,
                    kernel_size=1,
                    bias=False,
                ),
                make_group_norm(dim),
                nn.GELU(),
            )
            if previous_dim is not None
            else None
        )

        self.detail_projection = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False,
        )
        self.structural_projection = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=False,
        )

        branch_count = 3 if previous_dim is not None else 2
        coupled_dim = branch_count * dim

        self.coupled_projection = nn.Sequential(
            LayerNorm2d(coupled_dim),
            nn.Conv2d(
                coupled_dim,
                dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GELU(),
        )

        self.detail_calibration = SpatialCalibration(kernel_size=7)
        self.structural_calibration = ChannelCalibration(
            dim=dim,
            reduction=reduction,
        )

        self.output_projection = InvertedResidualMLP(
            input_dim=dim * 3,
            output_dim=dim,
        )
        self.drop_path = (
            DropPath(drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity()
        )
        self.refinement = ResidualCalibrationUnit(
            dim=dim,
            dropout=refinement_dropout,
        )

    def forward(
        self,
        detail_feature: torch.Tensor,
        structural_feature: torch.Tensor,
        previous_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if detail_feature.shape != structural_feature.shape:
            raise ValueError(
                "Detail and aligned structural features must have "
                "identical shapes."
            )

        detail_projected = self.detail_projection(detail_feature)
        structural_projected = self.structural_projection(
            structural_feature
        )

        coupled_inputs = [
            detail_projected,
            structural_projected,
        ]
        propagated_feature = None

        if self.previous_projection is not None:
            if previous_feature is None:
                raise ValueError(
                    "A previous fused feature is required at this stage."
                )

            previous_feature = F.interpolate(
                previous_feature,
                size=detail_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            propagated_feature = self.previous_projection(
                previous_feature
            )
            coupled_inputs.insert(0, propagated_feature)
        elif previous_feature is not None:
            raise ValueError(
                "The first PSCF stage must not receive a previous feature."
            )

        coupled_feature = self.coupled_projection(
            torch.cat(coupled_inputs, dim=1)
        )

        calibrated_detail = self.detail_calibration(detail_feature)
        calibrated_structural = self.structural_calibration(
            structural_feature
        )

        fused_feature = self.output_projection(
            torch.cat(
                [
                    calibrated_structural,
                    calibrated_detail,
                    coupled_feature,
                ],
                dim=1,
            )
        )

        if propagated_feature is not None:
            fused_feature = (
                propagated_feature
                + self.drop_path(fused_feature)
            )
        else:
            fused_feature = self.drop_path(fused_feature)

        return self.refinement(fused_feature)


class ProgressiveStageCoupledFusion(nn.Module):
    """Four-stage PSCF module."""

    def __init__(
        self,
        stage_dims: Sequence[int] = (96, 192, 384, 768),
        reduction: int = 16,
        drop_path_rate: float = 0.0,
        refinement_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(stage_dims) != 4:
            raise ValueError("PSCF requires exactly four stages.")

        rates = torch.linspace(
            0.0,
            drop_path_rate,
            len(stage_dims),
        ).tolist()

        self.blocks = nn.ModuleList(
            [
                StageCoupledFusionBlock(
                    dim=dim,
                    previous_dim=(
                        None
                        if index == 0
                        else stage_dims[index - 1]
                    ),
                    reduction=reduction,
                    drop_path_rate=rates[index],
                    refinement_dropout=refinement_dropout,
                )
                for index, dim in enumerate(stage_dims)
            ]
        )

    def forward(
        self,
        structural_features: Sequence[torch.Tensor],
        detail_features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if len(structural_features) != 4:
            raise ValueError(
                "PSCF expects four structural feature maps."
            )
        if len(detail_features) != 4:
            raise ValueError(
                "PSCF expects four detail feature maps."
            )

        fused_features = []
        previous_feature = None

        for index, block in enumerate(self.blocks):
            previous_feature = block(
                detail_feature=detail_features[index],
                structural_feature=structural_features[index],
                previous_feature=previous_feature,
            )
            fused_features.append(previous_feature)

        return tuple(fused_features)


__all__ = [
    "ChannelCalibration",
    "ProgressiveStageCoupledFusion",
    "ResidualCalibrationUnit",
    "SpatialCalibration",
    "StageCoupledFusionBlock",
]
