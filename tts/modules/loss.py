from dataclasses import dataclass
from typing import Union, Sequence

import torch

from tts.modules.constructor import ModuleConfig, Constructor


@dataclass
class WeightedLossConfig(ModuleConfig):
    weight: Union[float, Sequence[float]] = 1.0
    skip_steps: int = 0


class WeightedLoss(torch.nn.Module, Constructor):
    def __init__(
            self,
            weight: Union[float, Sequence[float]] = 1.0,
            skip_steps: int = 0
    ):
        super().__init__()

        self.weight = weight
        self.skip_steps = skip_steps

    def weight_loss(self, loss_value, step=None):
        if step is not None and step < self.skip_steps:
            return 0.

        return self.weight * loss_value
