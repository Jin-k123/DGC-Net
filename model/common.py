"""Shared layers used by the DGC-Net modules."""

from __future__ import annotations

import torch
import torch.nn as nn


def make_group_norm(
    num_channels: int,
    max_groups: int = 32,
) -> nn.GroupNorm:
    """Create GroupNorm with a valid number of groups."""

    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Apply stochastic depth independently to each sample."""

    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device,
    )
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Stochastic depth layer."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Module):
    """Layer normalization for NCHW tensors."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


__all__ = [
    "DropPath",
    "LayerNorm2d",
    "drop_path",
    "make_group_norm",
]
