from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from omegaconf import MISSING
from torch import Tensor
from torch.nn import functional as F

from tts.modules.constructor import Constructor, ModuleConfig
from tts.modules.layers import choose_activation, choose_normalization
from tts.utils import min_dtype_value, max_dtype_value, get_mask_from_lengths


def batch_diagonal_prior(text_lengths, mel_lengths, gamma: float = 0.1, threshold: float = 1e-4):
    dtype, device = torch.float32, text_lengths.device

    grid_text = torch.arange(text_lengths.max(), dtype=dtype, device=device)
    grid_text = grid_text.view(1, -1) / text_lengths.view(-1, 1)  # (B, T)

    grid_mel = torch.arange(mel_lengths.max(), dtype=dtype, device=device)
    grid_mel = grid_mel.view(1, -1) / mel_lengths.view(-1, 1)  # (B, M)

    grid = grid_text.unsqueeze(1) - grid_mel.unsqueeze(2)  # (B, M, T)

    prior = torch.exp(-grid ** 2 / (2 * gamma ** 2))

    prior.transpose(2, 1)[~get_mask_from_lengths(text_lengths)] = 0.
    prior[~get_mask_from_lengths(mel_lengths)] = 0.

    prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-5)
    prior = prior.masked_fill(prior < threshold, 0.0)

    return prior


class ConvBlock1D(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int | None = None,
            dilation: int = 1,
            bias: bool = True,
            activation: str = 'relu',
            normalization: str = 'batch',
            dropout_p: int | None = None
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int(dilation * (kernel_size - 1) / 2) if padding is None else padding,
            dilation=dilation,
            bias=bias and normalization is None
        )
        self.act = choose_activation(activation)()
        self.norm = choose_normalization(1, normalization)(
            num_features=out_channels) if normalization is not None else None
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
            self,
            x: Tensor,
            input_mask: Tensor | None = None,
            output_mask: Tensor | None = None
    ):
        if input_mask is not None:
            x = x * input_mask  # zero out padded values before applying convolution

        x = self.act(self.conv(x))
        if self.norm is not None:
            x = self.norm(x, mask=output_mask)
        x = self.dropout(x)

        return x


@dataclass
class ConvAttentionConfig(ModuleConfig):
    mel_dim: int = MISSING
    text_dim: int = 512
    attention_dim: int = 80
    key_kernel_size: int = 3
    query_kernel_size: int | Sequence[int] = (3, 3)
    dropout: float = 0.0
    normalization: str | None = "instance"
    activation: str = "relu"


class ConvAttention(nn.Module, Constructor):
    def __init__(
            self,
            mel_dim: int,
            text_dim: int = 512,
            attention_dim: int = 80,
            key_kernel_size: int = 3,
            query_kernel_size: int | Sequence[int] = (3, 3),
            dropout: float = 0.0,
            normalization: str | None = "instance",
            activation: str = "relu",
            attention_prior: bool = True
    ):
        super().__init__()

        self.mel_dim = mel_dim
        self.text_dim = text_dim

        self.scale = attention_dim ** -0.5

        key_params = [
            (text_dim, text_dim * 2, key_kernel_size, activation),
            (text_dim * 2, attention_dim, 1, "linear")
        ]
        self.key_proj = nn.ModuleList([
            ConvBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                activation=act,
                normalization=normalization if i < len(key_params) - 1 else None,
                dropout_p=dropout if dropout > 0. else None
            )
            for i, (in_channels, out_channels, kernel_size, act) in enumerate(key_params)
        ])

        if isinstance(query_kernel_size, int):
            query_kernel_size = [query_kernel_size] * 2

        query_params = [
            (mel_dim, mel_dim * 2, query_kernel_size[0], activation),
            (mel_dim * 2, mel_dim, query_kernel_size[1], activation),
            (mel_dim, attention_dim, 1, "linear")
        ]
        self.query_proj = nn.ModuleList([
            ConvBlock1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=False,
                activation=act,
                normalization=normalization if i < len(query_params) - 1 else None,
                dropout_p=dropout if dropout > 0. else None
            )
            for i, (in_channels, out_channels, kernel_size, act) in enumerate(query_params)
        ])

        self.attention_prior = attention_prior

    @torch.jit.unused
    def forward(
            self,
            queries: Tensor,
            keys: Tensor,
            query_len: Tensor,
            key_len: Tensor
    ):
        """Parallel attention.

        Args:
            queries (Tensor): B x mel_dim x T1 tensor (probably going to be mel data)
            keys (Tensor): B x text_dim x T2 tensor (text data)
            query_len (Tensor): B x mel_dim (mel lengths)
            key_len (Tensor): B x text_len (text lengths)
        Output:
            attn (Tensor): B x T1 x T2 attention mask. Final dim T2 should sum to 1.
        """
        key_mask = get_mask_from_lengths(key_len).unsqueeze(1)
        query_mask = get_mask_from_lengths(query_len).unsqueeze(1)
        mask = query_mask.transpose(1, 2) & key_mask

        keys_enc = keys.transpose(1, 2) if keys.shape[1] != self.text_dim else keys
        for conv in self.key_proj:
            keys_enc = conv(keys_enc, input_mask=key_mask, output_mask=key_mask)  # (B, attn_dim, T2)

        # Beware can only do this since query_dim = attn_dim = n_mel_channels
        queries_enc = queries.transpose(1, 2) if queries.shape[1] != self.mel_dim else queries
        for conv in self.query_proj:
            queries_enc = conv(queries_enc, input_mask=query_mask, output_mask=query_mask)  # (B, attn_dim, T1)

        attn = torch.matmul(queries_enc.transpose(1, 2), keys_enc)  # (B, T1, T2)
        attn = self.scale * attn  # (B, T1, T2)

        attn = torch.clamp(attn, max=max_dtype_value(attn))  # clamp potential +inf that produce nan in softmax

        if self.attention_prior:
            attn_prior = batch_diagonal_prior(text_lengths=key_len, mel_lengths=query_len)
            attn = F.log_softmax(attn, dim=2, dtype=torch.float32) + torch.log(attn_prior + 1e-6)

        attn_logits = attn.clone()

        if mask is not None:
            attn.masked_fill_(~mask[:, :1], min_dtype_value(attn))

        attn = F.softmax(attn, dim=2, dtype=torch.float32)  # softmax along T2

        if mask is not None:
            attn = attn * mask

        return attn, attn_logits


@dataclass
class AlignerConfig(ConvAttentionConfig):
    ...


class AlignerOutput(NamedTuple):
    attn_soft: Tensor
    attn_logits: Tensor
    attn_hard: Tensor
    attn_hard_duration: Tensor


class Aligner(nn.Module, Constructor):
    def __init__(
            self,
            mel_dim: int,
            text_dim: int = 512,
            attention_dim: int = 80,
            key_kernel_size: int = 3,
            query_kernel_size: int | Sequence[int] = (3, 3),
            dropout: float = 0.0,
            normalization: str | None = "instance",
            activation: str = "relu",
            attention_prior: bool = True
    ):
        super().__init__()

        self.attention = ConvAttention(
            mel_dim=mel_dim,
            text_dim=text_dim,
            attention_dim=attention_dim,
            key_kernel_size=key_kernel_size,
            query_kernel_size=query_kernel_size,
            dropout=dropout,
            normalization=normalization,
            activation=activation,
            attention_prior=attention_prior
        )

        self._numba_cuda = False
        if torch.cuda.is_available():
            try:
                from tts.modules.aligner.cuda_mas import cuda_b_mas
                self._numba_cuda = True
            except Exception:
                pass

    @torch.jit.unused
    def forward(
            self,
            mel: Tensor,
            enc_text: Tensor,
            mel_len: Tensor,
            text_len: Tensor
    ):
        # make sure to do the alignments before folding
        attn_soft, attn_logits = self.attention(
            queries=mel, keys=enc_text,
            query_len=mel_len, key_len=text_len
        )

        attn_hard = self.binarize_attention_parallel(attn_logits, text_len, mel_len)

        # Viterbi --> durations
        attn_hard_duration = attn_hard.sum(dim=1)

        # check duration sums and mel lengths
        if not torch.all(torch.eq(attn_hard_duration.sum(dim=1), mel_len)):
            print("Failed `duration`/`mel_len` assertion, some text lengths are longer than mel lengths: ",
                  attn_hard_duration.sum(dim=1), mel_len)
            print("Trying to fix that for you, but better check your data!")
            attn_hard_duration[:, 0] += mel_len - attn_hard_duration.sum(dim=1)

        return AlignerOutput(
            attn_soft=attn_soft,
            attn_logits=attn_logits,
            attn_hard=attn_hard,
            attn_hard_duration=attn_hard_duration
        )

    @torch.no_grad()
    def binarize_attention_parallel(self, attn_logits: Tensor, text_len: Tensor, mel_len: Tensor):
        """For training purposes only. Binarizes attention with MAS.
           These will no longer receive a gradient.
        Args:
            attn_logits: B x max_mel_len x max_text_len
        """
        if self._numba_cuda and attn_logits.is_cuda:
            return self.cuda_binarize_attention_parallel(attn_logits, text_len, mel_len)
        else:
            return self.cpu_binarize_attention_parallel(attn_logits, text_len, mel_len)

    @staticmethod
    @torch.no_grad()
    def cpu_binarize_attention_parallel(attn_logits: Tensor, text_len: Tensor, mel_len: Tensor):
        from tts.modules.aligner import b_mas
        attn_out = b_mas(
            attn_logits.detach().cpu().numpy(),
            in_lens=text_len.cpu().numpy(),
            out_lens=mel_len.cpu().numpy()
        )
        return torch.from_numpy(attn_out).to(attn_logits.device)

    @staticmethod
    @torch.no_grad()
    def cuda_binarize_attention_parallel(attn_logits: Tensor, text_len: Tensor, mel_len: Tensor):
        # import locally to avoid kernel compilation after importing with other models
        from tts.modules.aligner.cuda_mas import cuda_b_mas

        # attention logits and pre-allocated hard attention
        log_p = attn_logits.clone().detach()
        attn_out = torch.zeros_like(attn_logits, dtype=torch.int16)

        # allocated temporary variables
        prev_log_p = torch.zeros_like(attn_logits)
        prev_ind = torch.zeros_like(attn_logits, dtype=torch.int16)

        cuda_b_mas[(max(64, attn_logits.shape[0]), 1), (1, 256)](
            log_p, prev_log_p, prev_ind, attn_out, text_len, mel_len
        )
        return attn_out
