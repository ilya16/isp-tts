from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn, Tensor

from tts.models.acoustic.model import AcousticModelInput, AcousticModelOutput
from tts.modules.constructor import ModuleConfig, Constructor
from tts.modules.loss import WeightedLossConfig, WeightedLoss
from tts.utils import masked_mean, get_mask_from_lengths


@dataclass
class MelLossConfig(WeightedLossConfig):
    ...


class MelLoss(WeightedLoss):
    def __init__(self, weight: float | Sequence[float] = 1.0):
        super().__init__(weight)
        self.criterion = torch.nn.MSELoss(reduction="none")

    def forward(self, mels_out, mels_target, mel_lengths, step=None):
        mel_loss = self.criterion(mels_out, mels_target)

        mask = get_mask_from_lengths(mel_lengths)[:, None].expand_as(mels_out)
        mel_loss = masked_mean(mel_loss, mask)

        return self.weight_loss(mel_loss, step=step)


@dataclass
class AttentionCTCLossConfig(WeightedLossConfig):
    blank_logprob: int = -1


class AttentionCTCLoss(WeightedLoss):
    def __init__(
            self,
            blank_logprob: int = -1,
            weight: float | Sequence[float] = 1.0,
            skip_steps: int = 0
    ):
        super().__init__(weight, skip_steps)
        self.blank_logprob = blank_logprob
        self.criterion = nn.CTCLoss(zero_infinity=True)

    @staticmethod
    def get_target_seqs(lengths):
        ids = torch.arange(1, lengths.max() + 1, device=lengths.device)
        ids = ids[None].expand(lengths.numel(), -1).clone()
        ids[ids > lengths.unsqueeze(1)] = 0
        return ids

    def forward(
            self,
            attn_logits: Tensor,
            text_lengths: Tensor,
            mel_lengths: Tensor,
            step: int | None = None
    ):
        attn_logits_padded = F.pad(input=attn_logits, pad=(1, 0), value=self.blank_logprob)

        attn_logprob_padded = F.log_softmax(attn_logits_padded, dim=2)
        attn_logprob_padded = attn_logprob_padded.transpose(0, 1)
        target_seqs = self.get_target_seqs(text_lengths)

        loss = self.criterion(
            log_probs=attn_logprob_padded,
            targets=target_seqs,
            input_lengths=mel_lengths,
            target_lengths=text_lengths
        )

        return self.weight_loss(loss, step)


@dataclass
class AttentionBinarizationLossConfig(WeightedLossConfig):
    eps: float = 1e-6


class AttentionBinarizationLoss(WeightedLoss):
    def __init__(
            self,
            weight: float | Sequence[float] = 1.0,
            skip_steps: int = 0,
            eps: float = 1e-6
    ):
        super().__init__(weight, skip_steps)
        self.eps = eps

    def forward(
            self,
            soft_attention: Tensor,
            hard_attention: Tensor,
            step: int | None = None
    ):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=self.eps)).sum()
        loss = -log_sum / hard_attention.sum()
        return self.weight_loss(loss, step)


@dataclass
class AcousticLossConfig(ModuleConfig):
    mel_loss: MelLossConfig | DictConfig = field(
        default_factory=lambda: MelLossConfig()
    ),
    attention_loss: AttentionCTCLossConfig | DictConfig | None = field(
        default_factory=lambda: AttentionCTCLossConfig()
    ),
    attention_kl_loss: AttentionBinarizationLossConfig | DictConfig | None = field(
        default_factory=lambda: AttentionBinarizationLossConfig()
    )


class AcousticModelLoss(torch.nn.Module, Constructor):
    def __init__(
            self,
            mel_loss: MelLossConfig | DictConfig = MelLossConfig(),
            attention_loss: AttentionCTCLossConfig | DictConfig | None = AttentionCTCLossConfig(),
            attention_kl_loss: AttentionBinarizationLossConfig | DictConfig | None = AttentionBinarizationLossConfig()
    ):
        super().__init__()

        self.mel_criterion = MelLoss.init(config=mel_loss)

        self.attention_criterion = None
        if attention_loss is not None:
            self.attention_criterion = AttentionCTCLoss.init(config=attention_loss)

        self.attention_kl_criterion = None
        if attention_kl_loss is not None:
            self.attention_kl_criterion = AttentionBinarizationLoss.init(config=attention_kl_loss)

    def forward(self, inputs: dict | AcousticModelInput, outputs: dict | AcousticModelOutput, step=None):
        if isinstance(inputs, dict):
            inputs = AcousticModelInput(**inputs)

        if isinstance(outputs, dict):
            outputs = AcousticModelOutput(**outputs)

        loss, losses = 0., {}

        mel_loss = self.mel_criterion(
            mels_out=outputs.mel,
            mels_target=inputs.mel,
            mel_lengths=inputs.mel_len,
            step=step
        )
        losses["model/mel_loss"] = mel_loss
        loss += mel_loss

        if outputs.adaptor_output.losses is not None:
            for key, loss_i in outputs.adaptor_output.losses.items():
                losses[f"adaptor/{key}"] = loss_i
                loss += loss_i

        if self.attention_criterion is not None:
            attn_loss = self.attention_criterion(
                attn_logits=outputs.aligner_output.attn_logits,
                text_lengths=inputs.text_len,
                mel_lengths=inputs.mel_len,
                step=step
            )
            losses["aligner/attention_loss"] = attn_loss
            loss += attn_loss

        if self.attention_kl_criterion is not None:
            attn_kl_loss = self.attention_kl_criterion(
                soft_attention=outputs.aligner_output.attn_soft,
                hard_attention=outputs.aligner_output.attn_hard,
                step=step
            )
            losses["aligner/kl_loss"] = attn_kl_loss
            loss += attn_kl_loss

        return loss, losses
