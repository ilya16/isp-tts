""" Transformer layer and model with optional style adaptive normalization. """
from dataclasses import dataclass, field
from typing import Union, Optional, NamedTuple

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from tts.modules.constructor import ModuleConfig, Constructor
from .attend import AttentionIntermediates
from .attention import AttentionConfig, Attention, AttentionSharedIntermediates
from .embeddings import FixedPositionalEmbedding
from .feedforward import FeedForwardConfig, FeedForward
from .normalization import LayerNorm, AdaptiveLayerNorm


class TransformerLayerIntermediates(NamedTuple):
    attention: Optional[AttentionIntermediates] = None


class TransformerLayerOutput(NamedTuple):
    out: Tensor
    intermediates: Optional[TransformerLayerIntermediates] = None
    shared_intermediates: Optional[AttentionSharedIntermediates] = None


@dataclass
class TransformerLayerConfig(ModuleConfig):
    dim: int = 384
    attention: Union[AttentionConfig, DictConfig] = field(default_factory=lambda: AttentionConfig())
    feed_forward: Union[FeedForwardConfig, DictConfig] = field(default_factory=lambda: FeedForwardConfig())
    pre_norm: bool = True
    adaptive_norm: bool = False
    condition_dim: Optional[int] = None


class TransformerLayer(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 384,
            attention: Union[AttentionConfig, DictConfig] = AttentionConfig(),
            feed_forward: Union[FeedForwardConfig, DictConfig] = FeedForwardConfig(),
            pre_norm: bool = True,
            adaptive_norm: bool = False,
            condition_dim: Optional[int] = None
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.adaptive_norm = adaptive_norm

        assert not adaptive_norm or condition_dim is not None
        norm_fn = AdaptiveLayerNorm if adaptive_norm else LayerNorm
        norm_kwargs = dict(condition_dim=condition_dim) if adaptive_norm else dict()

        self.attention_norm = norm_fn(dim, **norm_kwargs)
        self.attention = Attention.init(attention, dim=dim)

        self.feed_forward_norm = norm_fn(dim, **norm_kwargs)
        self.feed_forward = FeedForward.init(feed_forward, dim=dim)

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            adaptive_condition: Optional[Tensor] = None,
            cache: Optional[TransformerLayerIntermediates] = None,
            shared_cache: Optional[AttentionSharedIntermediates] = None
    ):
        assert not self.adaptive_norm or adaptive_condition is not None, \
            "`adaptive_condition` should be provided for AdaptiveLayerNorm"

        # attention layer + normalization
        out = residual = x
        if self.pre_norm:  # (pre-layer) normalization
            out = self.attention_norm(out, adaptive_condition)

        attention_cache = cache.attention if cache is not None else None
        out, attention_cache, shared_cache = self.attention(
            out,
            mask=mask,
            context=context,
            context_mask=context_mask,
            attention_mask=attention_mask,
            cache=attention_cache,
            shared_cache=shared_cache
        )
        out = out + residual  # residual connection

        if not self.pre_norm:  # (post-layer) normalization
            out = self.attention_norm(out, adaptive_condition)

        # position-wise feed-forward + normalization
        residual = out
        if self.pre_norm:  # (pre-layer) normalization
            out = self.feed_forward_norm(out, adaptive_condition)

        mask = mask[..., None] if mask is not None else None
        out = out * mask if mask is not None else out

        out = self.feed_forward(out)
        out = out + residual  # residual connection

        if not self.pre_norm:  # (post-layer) normalization
            out = self.feed_forward_norm(out, adaptive_condition)

        out = out * mask if mask is not None else out

        cache = TransformerLayerIntermediates(attention=attention_cache)

        return TransformerLayerOutput(
            out=out,
            intermediates=cache,
            shared_intermediates=shared_cache
        )


class TransformerOutput(NamedTuple):
    out: Tensor
    intermediates: Optional[list[TransformerLayerIntermediates]] = None


@dataclass
class TransformerConfig(ModuleConfig):
    dim: int = 384
    depth: int = 6
    transformer_layer: Union[TransformerLayerConfig, DictConfig] = field(
        default_factory=lambda: TransformerLayerConfig())
    emb_dim: Optional[int] = None
    use_abs_pos_emb: bool = True
    adaptive_norm: bool = False
    condition_dim: Optional[int] = None


class Transformer(nn.Module, Constructor):
    def __init__(
            self,
            dim: int = 384,
            depth: int = 6,
            transformer_layer: Union[TransformerLayerConfig, DictConfig] = TransformerLayerConfig(),
            emb_dim: Optional[int] = None,
            use_abs_pos_emb: bool = True,
            adaptive_norm: bool = False,
            condition_dim: Optional[int] = None
    ):
        super().__init__()

        self.dim = dim
        self.emb_dim = emb_dim = emb_dim or dim
        self.adaptive_norm = adaptive_norm

        self.layers = nn.ModuleList([
            TransformerLayer.init(
                transformer_layer,
                dim=dim,
                adaptive_norm=adaptive_norm,
                condition_dim=condition_dim
            )
            for _ in range(depth)
        ])

        pre_norm = self.layers[0].pre_norm
        project_emb = self.emb_dim != self.dim
        has_rel_pos_emb = self.layers[0].attention.rel_pos is not None

        self.pos_emb = FixedPositionalEmbedding(emb_dim) if use_abs_pos_emb and not has_rel_pos_emb else None
        self.project_emb = nn.Linear(emb_dim, dim) if project_emb else nn.Identity()

        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            adaptive_condition: Optional[Tensor] = None,
            return_intermediates: bool = False
    ):
        if self.pos_emb is not None:
            x = x + self.pos_emb(x)

        x = self.project_emb(x)

        out, cache, shared_cache = x, None, None
        intermediates: list[TransformerLayerIntermediates] = []
        for layer in self.layers:
            out, new_cache, shared_cache = layer(
                out,
                mask=mask,
                context=context,
                context_mask=context_mask,
                attention_mask=attention_mask,
                adaptive_condition=adaptive_condition,
                cache=cache,
                shared_cache=shared_cache
            )
            if return_intermediates and new_cache is not None:
                intermediates.append(new_cache)

        out = self.norm(out)
        out = out * mask[..., None] if mask is not None else out

        return TransformerOutput(
            out=out,
            intermediates=intermediates
        )
