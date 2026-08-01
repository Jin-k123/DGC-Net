"""
Bilateral Gated Semantic Calibration (BGSC).

This implementation follows the module described in the DGC-Net paper:

1. Bidirectional cross-attention between the highest-stage structural
   and detail representations.
2. Concatenation-based multi-stage aggregation of F1-F4.
3. Two channel gates and one spatial gate.
4. Residual gated calibration:
       F_out = F_ms + W_g * G_ctx + W_l * S * L_ctx
5. Attention-guided pooling and classification.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import make_group_norm


class BidirectionalCrossAttention(nn.Module):
    """Exchange semantic information in both structural-detail directions."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "The feature dimension must be divisible by num_heads."
            )

        self.structural_queries_detail = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.detail_queries_structural = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.structural_scale = nn.Parameter(torch.tensor(0.1))
        self.detail_scale = nn.Parameter(torch.tensor(0.1))

        self.structural_norm = nn.LayerNorm(dim)
        self.detail_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _to_tokens(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, _, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, (height, width)

    @staticmethod
    def _to_feature_map(
        tokens: torch.Tensor,
        spatial_size: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = spatial_size
        batch_size, _, channels = tokens.shape
        return (
            tokens.transpose(1, 2)
            .reshape(batch_size, channels, height, width)
            .contiguous()
        )

    def forward(
        self,
        structural_feature: torch.Tensor,
        detail_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if detail_feature.shape[-2:] != structural_feature.shape[-2:]:
            detail_feature = F.interpolate(
                detail_feature,
                size=structural_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if structural_feature.shape[1] != detail_feature.shape[1]:
            raise ValueError(
                "Structural and detail features must have equal channels."
            )

        structural_tokens, spatial_size = self._to_tokens(
            structural_feature
        )
        detail_tokens, _ = self._to_tokens(detail_feature)

        structural_context, _ = self.structural_queries_detail(
            query=structural_tokens,
            key=detail_tokens,
            value=detail_tokens,
            need_weights=False,
        )
        detail_context, _ = self.detail_queries_structural(
            query=detail_tokens,
            key=structural_tokens,
            value=structural_tokens,
            need_weights=False,
        )

        structural_tokens = self.structural_norm(
            structural_tokens
            + self.dropout(
                self.structural_scale * structural_context
            )
        )
        detail_tokens = self.detail_norm(
            detail_tokens
            + self.dropout(
                self.detail_scale * detail_context
            )
        )

        structural_context_map = self._to_feature_map(
            structural_tokens,
            spatial_size,
        )
        detail_context_map = self._to_feature_map(
            detail_tokens,
            spatial_size,
        )

        return structural_context_map, detail_context_map


class MultiStageAggregation(nn.Module):
    """Resize and concatenate F1-F4 before a lightweight projection."""

    def __init__(
        self,
        input_dims: Sequence[int] = (96, 192, 384, 768),
        output_dim: int = 768,
    ) -> None:
        super().__init__()

        if len(input_dims) != 4:
            raise ValueError(
                "Multi-stage aggregation requires four feature levels."
            )

        self.projection = nn.Sequential(
            nn.Conv2d(
                sum(input_dims),
                output_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        fused_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(fused_features) != 4:
            raise ValueError(
                "Multi-stage aggregation expects F1, F2, F3, and F4."
            )

        target_size = fused_features[-1].shape[-2:]
        resized_features = []

        for feature in fused_features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            resized_features.append(feature)

        return self.projection(
            torch.cat(resized_features, dim=1)
        )


class BilateralChannelGate(nn.Module):
    """Generate separate channel gates for both contextual features."""

    def __init__(
        self,
        dim: int = 768,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        hidden_dim = max(dim // reduction, 64)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                dim * 2,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                dim * 2,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        structural_context: torch.Tensor,
        detail_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_feature = torch.cat(
            [structural_context, detail_context],
            dim=1,
        )
        descriptor = F.adaptive_avg_pool2d(
            joint_feature,
            output_size=1,
        )
        gates = torch.sigmoid(self.mlp(descriptor))
        return gates.chunk(2, dim=1)


class SpatialGate(nn.Module):
    """Generate the spatial calibration map used by the detail context."""

    def __init__(
        self,
        dim: int = 768,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        hidden_dim = max(dim // reduction, 64)

        self.projection = nn.Sequential(
            nn.Conv2d(
                dim * 2,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=7,
                padding=3,
                bias=True,
            ),
        )

    def forward(
        self,
        structural_context: torch.Tensor,
        detail_context: torch.Tensor,
    ) -> torch.Tensor:
        joint_feature = torch.cat(
            [structural_context, detail_context],
            dim=1,
        )
        return torch.sigmoid(self.projection(joint_feature))


class AttentionPooling2D(nn.Module):
    """Learn spatial attention weights and compute a weighted feature vector."""

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.score = nn.Sequential(
            nn.Conv2d(
                dim,
                hidden_dim,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = x.shape

        attention = self.score(x).flatten(2)
        attention = F.softmax(attention, dim=-1)

        features = x.flatten(2)
        pooled = torch.sum(features * attention, dim=-1)

        attention_map = attention.view(
            batch_size,
            1,
            height,
            width,
        )
        return pooled, attention_map


class ClassificationHead(nn.Module):
    """MLP classifier used after attention pooling."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or input_dim // 2

        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
            if dropout > 0.0
            else nn.Identity(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BilateralGatedSemanticCalibration(nn.Module):
    """Complete BGSC module with attention pooling and classification."""

    def __init__(
        self,
        stage_dims: Sequence[int] = (96, 192, 384, 768),
        dim: int = 768,
        num_classes: int = 6,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        gate_reduction: int = 4,
        pooling_hidden_dim: int = 256,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cross_attention = BidirectionalCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.multi_stage_aggregation = MultiStageAggregation(
            input_dims=stage_dims,
            output_dim=dim,
        )
        self.channel_gate = BilateralChannelGate(
            dim=dim,
            reduction=gate_reduction,
        )
        self.spatial_gate = SpatialGate(
            dim=dim,
            reduction=gate_reduction,
        )
        self.output_norm = make_group_norm(dim)
        self.attention_pool = AttentionPooling2D(
            dim=dim,
            hidden_dim=pooling_hidden_dim,
        )
        self.classifier = ClassificationHead(
            input_dim=dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
            hidden_dim=dim // 2,
        )

    def forward(
        self,
        structural_feature: torch.Tensor,
        detail_feature: torch.Tensor,
        fused_features: Sequence[torch.Tensor],
        return_aux: bool = False,
    ):
        structural_context, detail_context = self.cross_attention(
            structural_feature,
            detail_feature,
        )
        multi_stage_feature = self.multi_stage_aggregation(
            fused_features
        )

        structural_gate, detail_gate = self.channel_gate(
            structural_context,
            detail_context,
        )
        spatial_gate = self.spatial_gate(
            structural_context,
            detail_context,
        )

        output_feature = (
            multi_stage_feature
            + structural_gate * structural_context
            + detail_gate * spatial_gate * detail_context
        )
        output_feature = self.output_norm(output_feature)

        pooled_feature, attention_map = self.attention_pool(
            output_feature
        )
        logits = self.classifier(pooled_feature)

        if not return_aux:
            return logits

        return {
            "logits": logits,
            "feature": output_feature,
            "pooled_feature": pooled_feature,
            "attention_map": attention_map,
            "structural_context": structural_context,
            "detail_context": detail_context,
            "structural_gate": structural_gate,
            "detail_gate": detail_gate,
            "spatial_gate": spatial_gate,
            "multi_stage_feature": multi_stage_feature,
        }


__all__ = [
    "AttentionPooling2D",
    "BilateralGatedSemanticCalibration",
    "BidirectionalCrossAttention",
    "BilateralChannelGate",
    "MultiStageAggregation",
    "SpatialGate",
]
