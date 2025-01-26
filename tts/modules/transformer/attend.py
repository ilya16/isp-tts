from typing import Optional, NamedTuple

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn, einsum, Tensor

from tts.utils import min_dtype_value

is_torch2 = version.parse(torch.__version__) >= version.parse('2.0.0')


class AttentionIntermediates(NamedTuple):
    queries: Tensor
    keys: Tensor
    values: Tensor
    qk_similarities: Optional[Tensor] = None


def unused(fn):
    if is_torch2:
        return fn
    else:
        return torch.jit.unused(fn)


class Attend(nn.Module):
    def __init__(
            self,
            *,
            dropout: float = 0.,
            causal: bool = False,
            scale: Optional[float] = None,
    ):
        super().__init__()

        self.causal = causal
        self.scale = scale

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # efficient attention
        self.efficient = is_torch2
        if self.efficient:
            torch.backends.cuda.enable_flash_sdp(False)

    @unused
    def efficient_attn(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None,
            attn_bias: Optional[Tensor] = None,
            offset: int = 0
    ) -> tuple[Tensor, AttentionIntermediates]:
        batch, heads, q_len, _ = q.shape
        k_len, device = k.shape[-2], q.device

        intermediates = AttentionIntermediates(queries=q.detach(), keys=k.detach(), values=v.detach())

        if k.ndim == 3:
            k = k[:, None].expand(-1, heads, -1, -1)

        if v.ndim == 3:
            v = v[:, None].expand(-1, heads, -1, -1)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = torch.ones(
                    (q_len, k_len), dtype=torch.bool, device=device
                ).triu(k_len - q_len + 1 - offset)
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if attn_bias is not None:
            if attn_bias.ndim == 3:
                attn_bias = attn_bias[None].expand(batch, -1, -1, -1)  # h i j -> b h i j

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = min_dtype_value(q)

            if mask is not None:
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = torch.ones(
                    (q_len, k_len), dtype=torch.bool, device=device
                ).triu(k_len - q_len + 1 - offset)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 attention: q, k, v, mask, dropout, causal, softmax_scale

        out = F.scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.,
            is_causal=causal and q_len != 1 and q_len == k_len
        )

        return out, intermediates

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            mask: Optional[Tensor] = None,
            attn_bias: Optional[Tensor] = None,
            offset: int = 0
    ) -> tuple[Tensor, AttentionIntermediates]:
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        if self.efficient and q.shape[0] <= 65535:
            return self.efficient_attn(
                q, k, v, mask=mask, attn_bias=attn_bias, offset=offset
            )

        n, device = q.shape[-2], q.device
        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k * scale)

        qk_similarities = dots.clone().detach()

        if attn_bias is not None:
            dots = dots + attn_bias

        dtype = dots.dtype
        mask_value = min_dtype_value(dots)

        if mask is not None:
            dots = dots.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1 - offset)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = dots.softmax(dim=-1, dtype=dtype)

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = AttentionIntermediates(
            queries=q.detach(), keys=k.detach(), values=v.detach(),
            qk_similarities=qk_similarities
        )

        return out, intermediates
