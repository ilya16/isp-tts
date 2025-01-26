""" Transformer Normalization layers. """

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch import Tensor

from tts.modules.constructor import ModuleConfig


@dataclass
class LayerNormConfig(ModuleConfig):
    dim: int = 384
    eps: float = 1e-5


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__(normalized_shape=dim, eps=eps)
        if not bias:
            self.bias = None

    def forward(self, x: Tensor, condition: Optional[Tensor] = None):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


@dataclass
class AdaptiveLayerNormConfig(ModuleConfig):
    dim: int = 384
    condition_dim: int = MISSING
    eps: float = 1e-5


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, condition_dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = nn.Linear(condition_dim, dim)
        nn.init.zeros_(self.weight.weight)
        nn.init.ones_(self.weight.bias)

        self.bias = None
        if bias:
            self.bias = nn.Linear(condition_dim, dim)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, x: Tensor, condition: Optional[Tensor] = None):
        if condition is None:
            weight, bias = x.new_ones(1), x.new_zeros(1)
        else:
            condition = condition.unsqueeze(1) if condition.ndim == 2 else condition
            weight = self.weight(condition)
            bias = self.bias(condition) if self.bias is not None else x.new_zeros(1)

        return weight * F.layer_norm(x, x.shape[-1:], None, None, eps=self.eps) + bias

    def extra_repr(self) -> str:
        return f'bias={self.bias is not None}'
