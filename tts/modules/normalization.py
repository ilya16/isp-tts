from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

__all__ = [
    "MaskedBatchNorm1d", "MaskedBatchNorm2d",
    "MaskedInstanceNorm1d", "MaskedInstanceNorm2d"
]


class _MaskedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        self._check_input_dim(inputs)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            update_buffers = True
        else:
            update_buffers = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if mask is not None:
            self._check_input_dim(mask)
            return _masked_norm(
                "batch", inputs, mask,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias,
                update_buffers, exponential_average_factor, self.eps
            )
        else:
            return F.batch_norm(
                inputs,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias,
                update_buffers, exponential_average_factor, self.eps
            )


class MaskedBatchNorm1d(torch.nn.BatchNorm1d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm1d` for details.

    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )


class MaskedBatchNorm2d(torch.nn.BatchNorm2d, _MaskedBatchNorm):
    r"""Applies Batch Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension)..

    See documentation of :class:`~torch.nn.BatchNorm2d` for details.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )


class _MaskedInstanceNorm(_InstanceNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        self._check_input_dim(inputs)

        use_input_stats = True
        if mask is not None:
            self._check_input_dim(mask)
            return _masked_norm(
                "instance", inputs, mask,
                self.running_mean, self.running_var, self.weight, self.bias,
                use_input_stats, self.momentum, self.eps
            )
        else:
            return F.instance_norm(
                inputs, self.running_mean, self.running_var, self.weight, self.bias,
                use_input_stats, self.momentum, self.eps
            )


class MaskedInstanceNorm1d(torch.nn.InstanceNorm1d, _MaskedInstanceNorm):
    r"""Applies Instance Normalization over a masked 3D input
    (a mini-batch of 1D inputs with additional channel dimension)..
    See documentation of :class:`~torch.nn.InstanceNorm1d` for details.
    Shape:
        - Input: :math:`(N, C, L)`
        - Mask: :math:`(N, 1, L)`
        - Output: :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = False) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)


class MaskedInstanceNorm2d(torch.nn.InstanceNorm2d, _MaskedInstanceNorm):
    r"""Applies Instance Normalization over a masked 4D input
    (a mini-batch of 2D inputs with additional channel dimension).
    See documentation of :class:`~torch.nn.InstanceNorm2d` for details.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Mask: :math:`(N, 1, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = False) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats)


# Masked Normalization
def _masked_norm(type_: str, inputs: Tensor, mask: Tensor,
                 running_mean: Optional[Tensor] = None, running_var: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None, bias: Optional[Tensor] = None,
                 use_input_stats: bool = True,
                 momentum: float = 0.1, eps: float = 1e-8) -> Tensor:
    in_shape = inputs.shape
    b, c = in_shape[:2]
    dims = torch.tensor(in_shape[2:])

    if len(dims) == 1 or len(dims) == 2:
        view_shape = [1, c] + [1] * len(dims)
    else:
        raise RuntimeError(f"Unsupported number of feature dims of the inputs: expected 1 or 2, got {len(dims)}.")

    if type_ == "batch":
        _axis = [0, 2] if len(dims) == 1 else [0, 2, 3]  # [B, C, HW] -> mean.size = [1, C, 1]
    elif type_ == "instance":
        _axis = [2] if len(dims) == 1 else [2, 3]  # [B, C, HW] -> mean.size = [B, C, 1]
    else:
        raise ValueError("Unknown type.")

    if mask.shape[1] == 1 != c:
        mask = mask.repeat(view_shape)

    if use_input_stats:
        lengths = mask.sum(_axis, keepdim=True)
        masked_inputs = mask * inputs

        mean = masked_inputs.sum(_axis, keepdim=True) / lengths
        var = (((masked_inputs - mean) * mask) ** 2).sum(_axis, keepdim=True) / lengths

        if type_ == "batch":
            if running_mean is not None:
                running_mean_ = running_mean.view(view_shape) * (1 - momentum) + momentum * mean.detach()
                running_mean.copy_(running_mean_.squeeze())
            if running_var is not None:
                running_var_ = running_var.view(view_shape) * (1 - momentum) + momentum * var.detach()
                running_var.copy_(running_var_.squeeze())
    elif type_ == "batch" and running_mean is not None and running_var is not None:
        mean, var = running_mean.view(view_shape), running_var.view(view_shape)
    else:
        raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')

    out = (inputs - mean) / (var + eps).sqrt()

    if weight is not None and bias is not None:
        out = out * weight.view(view_shape) + bias.view(view_shape)

    return out
