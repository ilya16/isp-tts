""" A set of utility classes and functions used throughout the repository. """

import random
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def prob2bool(prob):
    return random.choices([True, False], weights=[prob, 1 - prob])[0]


def count_parameters(model, requires_grad: bool = False):
    if requires_grad:
        return sum(map(lambda p: p.numel() if p.requires_grad else 0, model.parameters()))
    return sum(map(lambda p: p.numel(), model.parameters()))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def min_dtype_value(tensor):
    # TorchScript does not work with `torch.finfo(tensor.dtype).min`
    if tensor.dtype == torch.float16:
        return -65504.0
    else:
        return -3.4028234663852886e+38


def max_dtype_value(tensor):
    # TorchScript does not work with `torch.finfo(tensor.dtype).max`
    if tensor.dtype == torch.float16:
        return 65504.0
    else:
        return 3.4028234663852886e+38


def masked_mean(tensor: Tensor, mask: Tensor):
    if tensor.ndim == 3 and mask.ndim == 2:
        mask = mask[..., None].expand_as(tensor)

    tensor = tensor.masked_fill(~mask, 0.)

    if tensor.ndim == 3:
        num = tensor.sum(dim=-1).sum(dim=-1)
        den = mask.sum(dim=-1).sum(dim=-1)
    else:
        num = tensor.sum(dim=-1)
        den = mask.sum(dim=-1)

    tensor = num / den.clamp(min=1e-5)
    return tensor.mean()


def get_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None):
    max_len = max_len if max_len is not None else lengths.max().item()
    ids = torch.arange(max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def get_float_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None):
    max_len = max_len if max_len is not None else lengths.max().item()
    ids = torch.arange(max_len, device=lengths.device)
    mask = torch.maximum(torch.tensor(0.), lengths.unsqueeze(1) - ids)
    mask = torch.minimum(torch.tensor(1.), mask)
    return mask


def get_mask_3d(widths, heights):
    mask_width = get_mask_from_lengths(widths)
    mask_height = get_mask_from_lengths(heights)
    mask_3d = mask_width.unsqueeze(2) & mask_height.unsqueeze(1)
    return mask_3d
