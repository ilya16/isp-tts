from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from tts.modules.constructor import Constructor, ModuleConfig
from tts.modules.transformer.attend import Attend, AttentionIntermediates
from tts.modules.transformer.embeddings import LearnedALiBiPositionalBias


class AttentionSharedIntermediates(NamedTuple):
    rel_pos_bias: Optional[Tensor] = None


@dataclass
class AttentionConfig(ModuleConfig):
    dim: int = 256
    heads: int = 4
    head_dim: Optional[int] = 64
    causal: bool = False
    dropout: float = 0.
    one_kv_head: bool = False

    context_dim: Optional[int] = None

    alibi_pos_bias: bool = False
    alibi_heads: Optional[int] = None
    alibi_symmetric: bool = True


class Attention(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 256,
            heads: int = 4,
            head_dim: Optional[int] = 64,
            causal: bool = False,
            dropout: float = 0.,
            one_kv_head: bool = False,
            context_dim: Optional[int] = None,
            alibi_pos_bias: bool = False,
            alibi_heads: Optional[int] = None,
            alibi_symmetric: bool = True
    ):
        super().__init__()

        self.heads = heads
        self.causal = causal

        self.dim = dim
        self.head_dim = head_dim or dim // heads
        self.one_kv_head = one_kv_head

        self.out_dim = q_dim = head_dim * self.heads
        kv_dim = head_dim if one_kv_head else q_dim
        context_dim = context_dim or dim

        self.q_dim = q_dim
        self.kv_dim = kv_dim

        self.to_q = nn.Linear(dim, q_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * kv_dim, bias=False)

        self.scale = head_dim ** -0.5

        # relative positional bias

        self.rel_pos = None
        if alibi_pos_bias:
            alibi_heads = heads if alibi_heads is None else alibi_heads
            assert alibi_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = LearnedALiBiPositionalBias(heads=alibi_heads, total_heads=heads, symmetric=alibi_symmetric)

        # attend class - includes core attention algorithm

        self.attend = Attend(
            causal=causal,
            dropout=dropout,
            scale=self.scale
        )

        # output layer

        self.to_out = nn.Linear(self.out_dim, dim, bias=False)

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            cache: Optional[AttentionIntermediates] = None,
            shared_cache: Optional[AttentionSharedIntermediates] = None
    ):
        b, n = x.shape[:2]
        h, head_dim, device = self.heads, self.head_dim, x.device

        kv_input = x if context is None else context

        # project for queries, keys, values

        q = self.to_q(x)
        q = q.view(b, -1, h, head_dim).transpose(2, 1)  # b n (h d) -> b h n d

        if cache is not None and context is not None:
            k, v = cache.keys, cache.values
        else:
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            if not self.one_kv_head:
                k = k.view(b, -1, h, head_dim).transpose(2, 1)  # b n (h d) -> b h n d
                v = v.view(b, -1, h, head_dim).transpose(2, 1)  # b n (h d) -> b h n d

        # kv cache

        if cache is not None and context is None:
            k = torch.cat([cache.keys, k], dim=-2)
            v = torch.cat([cache.values, v], dim=-2)

        # handle all masks

        i, j = q.shape[-2], k.shape[-2]
        final_attn_mask: Optional[Tensor] = None

        input_mask: Optional[Tensor] = mask if context_mask is None else context_mask
        if input_mask is not None:
            input_mask = input_mask[:, None, None, :]  # b j -> b 1 1 j
            final_attn_mask = ~input_mask

        if attention_mask is not None:
            assert 2 <= attention_mask.ndim <= 4, "attention mask must have from 2 to 4 dimensions"
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[None, None]  # i j -> 1 1 i j
            elif attention_mask.ndim == 3:
                attention_mask = attention_mask[None]  # h i j -> 1 h i j
            final_attn_mask = final_attn_mask | (~attention_mask) if final_attn_mask is not None else ~attention_mask

        final_attn_mask = ~final_attn_mask if final_attn_mask is not None else None  # True for attended positions

        # prepare relative positional bias, if needed

        rel_pos_bias: Optional[Tensor] = None
        attn_bias: Optional[Tensor] = None
        if self.rel_pos is not None:
            if shared_cache is not None and shared_cache.rel_pos_bias is not None:
                rel_pos_bias = shared_cache.rel_pos_bias
            else:
                rel_pos_bias = self.rel_pos.get_bias(i, j, k=j - i).to(dtype=q.dtype)
            attn_bias = self.rel_pos(i, j, k=j - i, bias=rel_pos_bias)

        # attend

        out, intermediates = self.attend(
            q, k, v,
            mask=final_attn_mask,
            attn_bias=attn_bias
        )

        # merge heads

        out = out.transpose(2, 1).contiguous().view(b, -1, self.out_dim)  # b h n d -> b n (h d)

        # combine the heads

        out = self.to_out(out)

        if mask is not None:
            mask = mask[:, -1:] if cache is not None else mask
            out = out * mask[..., None]

        shared_intermediates = AttentionSharedIntermediates(rel_pos_bias=rel_pos_bias)

        return out, intermediates, shared_intermediates
