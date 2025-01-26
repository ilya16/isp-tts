from enum import Enum
from typing import Union

import torch

from .normalization import *


class Act(Enum):
    linear = "linear"
    relu = "relu"
    leaky_relu = "leaky_relu"
    selu = "selu"
    tanh = "tanh"
    mish = "mish"
    swish = "swish"
    gelu = "gelu"
    sigmoid = "sigmoid"


ACTIVATION_MAP = {
    Act.linear: torch.nn.Identity,
    Act.relu: torch.nn.ReLU,
    Act.leaky_relu: torch.nn.LeakyReLU,
    Act.selu: torch.nn.SELU,
    Act.tanh: lambda **kwargs: torch.nn.Tanh(),
    Act.mish: torch.nn.Mish,
    Act.swish: torch.nn.SiLU,
    Act.gelu: lambda **kwargs: torch.nn.GELU(),
    Act.sigmoid: lambda **kwargs: torch.nn.Sigmoid()
}


class Norm(Enum):
    batch = "batch"
    instance = "instance"


NORM_DIMENSION_TYPE_MAP = {
    1: {
        Norm.batch: MaskedBatchNorm1d,
        Norm.instance: MaskedInstanceNorm1d,
    },
    2: {
        Norm.batch: MaskedBatchNorm2d,
        Norm.instance: MaskedInstanceNorm2d,
    }
}


def choose_activation(act_type: Union[str, Act]):
    return ACTIVATION_MAP[Act(act_type)]


def choose_normalization(dims: int, norm_type: Union[str, Norm]):
    return NORM_DIMENSION_TYPE_MAP[dims][Norm(norm_type)]
