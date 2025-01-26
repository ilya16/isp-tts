""" Positional Embeddings for Transformer layers and models. """

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor):
        pos = torch.arange(x.shape[1], dtype=self.inv_freq.dtype, device=x.device)
        sinusoid_inp = pos.unsqueeze(-1) * self.inv_freq
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class ALiBiPositionalBias(nn.Module):
    def __init__(self, heads: int, total_heads: int, symmetric: bool = True):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads
        self.symmetric = symmetric

        slopes = torch.Tensor(self._compute_slopes(heads)).view(-1, 1, 1)
        if not symmetric:
            slopes = torch.stack([slopes, torch.roll(slopes, -1)])
        self.register_buffer('slopes', slopes, persistent=False)

    @staticmethod
    def _compute_slopes(heads):
        def slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return slopes_power_of_2(closest_power_of_2) \
            + slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]

    def get_bias(self, i: int, j: int, k: int = 0):
        i_arange = torch.arange(k, i + k, dtype=torch.int, device=self.slopes.device)
        j_arange = torch.arange(j, dtype=torch.int, device=self.slopes.device)
        return -torch.abs(j_arange[None, None, :] - i_arange[None, :, None])

    def get_slopes(self):
        return self.slopes

    def forward(self, i: int, j: int, k: int = 0, bias: Optional[Tensor] = None):
        if bias is not None and bias.shape[-2] >= i and bias.shape[-1] >= j:
            bias = bias[..., :i, :j] if bias.shape[-2] > i or bias.shape[-1] > j else bias
        else:
            bias = self.get_bias(i, j, k)

        slopes = self.get_slopes()
        if self.total_heads - slopes.shape[-3] > 0:
            slopes = F.pad(slopes, (0, 0, 0, 0, 0, self.total_heads - slopes.shape[-3]))

        if self.symmetric:
            return slopes * bias
        else:
            return slopes[0] * torch.tril(bias) + slopes[1] * torch.triu(bias)


class LearnedALiBiPositionalBias(ALiBiPositionalBias):
    def __init__(self, heads: int, total_heads: int, symmetric: bool = True):
        super().__init__(heads, total_heads, symmetric)
        log_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(log_slopes)

    def get_slopes(self):
        return self.learned_logslopes.exp()


class SinusoidalEmbedding(nn.Module):
    def __init__(
            self,
            dim: int,
            theta: float = 10000,
            freq_scale: float = 1.,
            with_positions: bool = False
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

        self.theta = theta
        self.register_buffer("freq_scale", torch.ones(1) * freq_scale, persistent=True)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.with_positions = with_positions

    def forward(self, x: Tensor, is_pos: bool = True, seq_dim: int = 1, offset: float = 0):
        pos = x if is_pos else torch.arange(x.shape[seq_dim], device=x.device)

        inv_freq = self.get_inv_freq()
        pos = pos.type_as(inv_freq) + offset
        emb = pos.unsqueeze(-1) * self.freq_scale * inv_freq
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.with_positions:
            return torch.cat((pos[:, None], emb), dim=-1)
        return emb

    def get_inv_freq(self):
        return self.inv_freq

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"theta={float(self.theta):.3f}, "
            f"freq_scale={float(self.freq_scale):.3f}, "
            f"with_positions={self.with_positions}"
        )


class TimePositionalEmbedding(nn.Module):
    def __init__(
            self,
            freq_dim: int = 256,
            emb_dim: int = 512,
            theta: int = 1000.,
            freq_scale: int = 1000.,
            with_steps: bool = False
    ):
        super().__init__()

        self.freq_emb = SinusoidalEmbedding(
            freq_dim,
            theta=theta,
            freq_scale=freq_scale,
            with_positions=with_steps
        )

        self.mlp = nn.Sequential(
            nn.Linear(freq_dim + int(with_steps), emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim, bias=True),
        )

    def forward(self, x: Tensor):
        freq_emb = self.freq_emb(x)
        return self.mlp(freq_emb)
