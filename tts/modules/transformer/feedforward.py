from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from tts.modules.constructor import ModuleConfig, Constructor
from tts.modules.layers import choose_activation


@dataclass
class FeedForwardConfig(ModuleConfig):
    dim: int = 384
    inner_dim: int = 1536
    dropout: float = 0.0
    activation: str = "relu"
    bias: bool = False
    glu: bool = False


class FeedForward(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 384,
            inner_dim: int = 1536,
            dropout: float = 0.0,
            activation: str = "relu",
            bias: bool = False,
            glu: bool = False
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * (1 + int(glu)), bias=bias),
            GLU(activation, dim=1) if glu else choose_activation(activation)(inplace=True),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Linear(inner_dim, dim, bias=bias)
        )

    def forward(self, x: Tensor):
        return self.net(x)


class GLU(nn.Module):
    def __init__(self, activation: str = "sigmoid", dim: int = -1):
        super().__init__()
        self.dim = dim
        self.act = choose_activation(activation)(inplace=False)

    def forward(self, x):
        x, gate = x.chunk(2, dim=self.dim)
        return x * self.act(gate)
