from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from tts.modules.constructor import ModuleConfig, Constructor
from tts.modules.transformer import TransformerConfig, Transformer
from tts.modules.transformer.embeddings import TimePositionalEmbedding
from tts.utils import masked_mean, get_float_mask_from_lengths, get_mask_3d


@dataclass
class TransformerTemporalModuleConfig(ModuleConfig):
    input_dim: int = 256
    output_dim: int = 256
    transformer: DictConfig | TransformerConfig = field(default_factory=lambda: TransformerConfig(dim=128, depth=2))
    detach_inputs: bool = False


class TransformerTemporalModule(nn.Module, Constructor):
    """ Base class for transformer temporal predictor/embedding modules. """

    def __init__(
            self,
            input_dim: int = 256,
            output_dim: int = 256,
            transformer: DictConfig | TransformerConfig = TransformerConfig(dim=128, depth=2),
            detach_inputs: bool = False
    ):
        super().__init__()

        self.transformer = Transformer.init(
            transformer,
            emb_dim=input_dim
        )

        self.linear_layer = nn.Linear(self.transformer.dim, output_dim, bias=True)

        self.detach_inputs = detach_inputs

    def forward(self, x: Tensor, mask: Tensor | None = None):
        x = x.detach() if self.detach_inputs else x

        out = self.transformer(x, mask=mask[..., 0] if mask is not None else None).out

        out = self.linear_layer(out)
        out = out * mask if mask is not None else out

        return out

    @torch.jit.export
    def infer(self, x: Tensor, mask: Tensor | None = None, steps: int = 4):
        return self.forward(x, mask)


@dataclass
class FlowTransformerTemporalModuleConfig(ModuleConfig):
    input_dim: int = 256
    output_dim: int = 256
    transformer: DictConfig | TransformerConfig = field(default_factory=lambda: TransformerConfig(dim=128, depth=2))
    time_embedding_dim: int | None = None
    sigma: float = 1e-5
    detach_inputs: bool = False


class FlowTransformerTemporalModule(nn.Module, Constructor):
    """ Base class for transformer temporal predictor/embedding modules. """

    def __init__(
            self,
            input_dim: int = 256,
            output_dim: int = 256,
            transformer: DictConfig | TransformerConfig = TransformerConfig(dim=128, depth=2),
            time_embedding_dim: int | None = None,
            sigma: float = 1e-5,
            detach_inputs: bool = False
    ):
        super().__init__()

        time_embedding_dim = time_embedding_dim or input_dim
        self.time_embedding = TimePositionalEmbedding(
            freq_dim=64, emb_dim=time_embedding_dim, with_steps=True
        )

        self.transformer = Transformer.init(
            transformer,
            emb_dim=output_dim + input_dim,
            adaptive_norm=True,
            condition_dim=time_embedding_dim
        )

        self.linear_layer = nn.Linear(self.transformer.dim, output_dim, bias=True)
        self.output_dim = output_dim
        self.sigma = sigma

        self.detach_inputs = detach_inputs

    def forward(self, x: Tensor, targets: Tensor, mask: Tensor | None = None):
        cond = x.detach() if self.detach_inputs else x

        if mask is None:
            mask = x.new_ones(x.shape[0], x.shape[1], dtype=torch.bool)
        elif mask.ndim == 3:
            mask = mask[..., 0]

        x1 = targets.detach()
        x0 = torch.randn_like(x1)

        time_steps = torch.rand((x1.shape[0],), dtype=x1.dtype, device=x1.device)
        time_emb = self.time_embedding(time_steps)

        t = time_steps[:, None, None]
        x_t = (1 - (1 - self.sigma) * t) * x0 + t * x1
        flow = x1 - (1 - self.sigma) * x0

        x = torch.cat([x_t, cond], dim=-1)

        out = self.transformer(x, mask=mask, adaptive_condition=time_emb).out
        pred_flow = self.linear_layer(out)

        mask = mask[..., None].expand_as(pred_flow)
        pred_flow = pred_flow * mask

        loss = F.mse_loss(pred_flow, flow, reduction="none")
        loss = masked_mean(loss, mask)

        losses = {"flow_loss": loss}

        with torch.no_grad():
            x_pred = (x0 + pred_flow) * mask

        return x_pred, losses

    @torch.jit.export
    def infer(self, x, mask: Tensor | None = None, steps: int = 4, step_factor: float = 0.75):
        if mask is None:
            mask = x.new_ones(x.shape[0], x.shape[1], dtype=torch.bool)
        elif mask.ndim == 3:
            mask = mask.squeeze(-1)

        cond = x
        x_t = torch.randn(x.shape[0], x.shape[1], self.output_dim, device=x.device)

        assert step_factor <= 1.
        if step_factor == 1.:
            time_steps = torch.linspace(0, 1, steps + 1, device=x_t.device)
        else:
            time_steps = -torch.diff(torch.logspace(0, steps, steps + 1, base=step_factor, device=x_t.device))
            time_steps = torch.cat([torch.tensor([0.], device=x_t.device), time_steps])
            time_steps = torch.cumsum(time_steps / time_steps.sum(), dim=0)

        for i, t in enumerate(time_steps[:-1]):
            dt = time_steps[i + 1] - time_steps[i]
            time_emb = self.time_embedding(t.view(1, 1))
            x = torch.cat([x_t, cond], dim=-1)

            out = self.transformer(x, mask=mask, adaptive_condition=time_emb).out
            pred_flow = self.linear_layer(out)

            x_t = x_t + pred_flow * dt

        x_t = x_t * mask[..., None]

        return x_t


class TemporalAdaptorOutput(NamedTuple):
    enc_out: Tensor
    log_duration: Optional[Tensor]
    duration: Tensor
    dec_lengths: Tensor
    pitch: Optional[Tensor]
    energy: Optional[Tensor]
    pitch_target: Optional[Tensor]
    energy_target: Optional[Tensor]
    losses: Optional[dict[str, Tensor]] = None


@dataclass
class FlowTemporalAdaptorConfig(ModuleConfig):
    encoder_dim: int = 384
    predictor: DictConfig | FlowTransformerTemporalModuleConfig = field(
        default_factory=lambda: FlowTransformerTemporalModuleConfig()
    )
    embedding: DictConfig | TransformerTemporalModuleConfig = field(
        default_factory=lambda: TransformerTemporalModuleConfig()
    )
    pitch: bool = True
    energy: bool = True
    soft_duration: bool = False


class FlowTemporalAdaptor(nn.Module, Constructor):
    def __init__(
            self,
            encoder_dim: int = 384,
            predictor: DictConfig | FlowTransformerTemporalModuleConfig = FlowTransformerTemporalModuleConfig(),
            embedding: DictConfig | TransformerTemporalModuleConfig = TransformerTemporalModuleConfig(),
            pitch: bool = True,
            energy: bool = True,
            soft_duration: bool = False
    ):
        super().__init__()

        self.length_regulator = LengthRegulator()
        self.averager = TemporalAverager()

        self.encoder_dim = encoder_dim
        self.feature_dim = 1 + int(pitch) + int(energy)

        self.pitch = pitch
        self.energy = energy

        self.pitch_idx = 1
        self.energy_idx = self.pitch_idx + 1 if self.energy else self.pitch_idx

        self.soft_duration = soft_duration

        self.predictor = FlowTransformerTemporalModule.init(
            predictor,
            input_dim=encoder_dim,
            output_dim=self.feature_dim,
        )

        self.embedding = TransformerTemporalModule.init(
            embedding,
            input_dim=self.feature_dim - 1,
            output_dim=encoder_dim
        )

    @torch.jit.unused
    def forward(
            self,
            enc_out: Tensor,
            enc_mask: Tensor,
            max_dec_len: Tensor,
            duration_target: Tensor | None = None,
            alignment: Tensor | None = None,
            pitch_target_dense: Tensor | None = None,
            energy_target_dense: Tensor | None = None
    ):
        enc_mask = enc_mask[..., None]  # (b, t, 1)

        assert alignment is not None or not self.soft_duration
        alignment = alignment if self.soft_duration else None

        # target features
        target_features = []

        if duration_target is not None:
            target_features.append(torch.log1p(duration_target)[..., None])

        pitch_target = None
        if self.pitch:
            pitch_target = self._process_target(pitch_target_dense, duration_target, alignment, enc_mask)
            target_features.append(pitch_target)

        energy_target = None
        if self.energy:
            energy_target = self._process_target(energy_target_dense, duration_target, alignment, enc_mask)
            target_features.append(energy_target)

        target_features = torch.cat(target_features, dim=-1)

        # make predictions
        pred, losses = self.predictor(enc_out, target_features, enc_mask)  # (b, t, c)

        # durations
        log_duration_pred = pred[..., :1].squeeze(-1)  # (b, t)
        duration_pred = torch.clamp(torch.exp(log_duration_pred) - 1, min=0)

        features = []

        # pitch
        pitch_pred = None
        if self.pitch:
            pitch_pred = pred[..., self.pitch_idx:self.pitch_idx + 1]
            pitch = pitch_target.detach() if pitch_target is not None else pitch_pred
            features.append(pitch)
            pitch_pred = pitch_pred.squeeze(-1)

        # energy
        energy_pred = None
        if self.energy:
            energy_pred = pred[..., self.energy_idx:self.energy_idx + 1]
            energy = energy_target.detach() if energy_target is not None else energy_pred
            features.append(energy)
            energy_pred = energy_pred.squeeze(-1)  # (b, t)

        features = torch.cat(features, dim=-1)
        enc_out = enc_out + self.embedding(features, mask=enc_mask)  # (b, t, enc_dim)

        # adapt encoder outputs to decoder lengths
        enc_out, dec_lens = self.length_regulator(enc_out, duration_target, max_len=max_dec_len, alignment=alignment)

        return TemporalAdaptorOutput(
            enc_out=enc_out,
            log_duration=log_duration_pred,
            duration=duration_pred,
            dec_lengths=dec_lens,
            pitch=pitch_pred,
            energy=energy_pred,
            pitch_target=pitch_target.squeeze(-1) if pitch_target is not None else None,
            energy_target=energy_target.squeeze(-1) if energy_target is not None else None,
            losses=losses
        )

    def _process_target(
            self,
            feature_target_dense: Tensor | None,
            duration_target: Tensor,
            alignment: Tensor | None = None,
            enc_mask: Tensor | None = None
    ):
        feature_target = None
        if feature_target_dense is not None:
            if feature_target_dense.ndim == 2:
                feature_target_dense = feature_target_dense[:, None]
            feature = self.averager(feature_target_dense, duration_target, alignment).transpose(1, 2)  # (b, t, 1)
            feature_target = feature * enc_mask if enc_mask is not None else feature

        return feature_target

    @torch.jit.export
    def infer(
            self,
            enc_out: Tensor,
            enc_mask: Tensor | None = None,
            duration_target: Tensor | None = None,
            duration_factor: float = 1.0,
            pitch_target: Tensor | None = None,
            pitch_factor: float = 1.0,
            pitch_delta: float = 0.,
            energy_target: Tensor | None = None,
            energy_factor: float = 1.0,
            energy_delta: float = 0.,
            steps: int = 4
    ):
        if enc_mask is not None:
            enc_mask = enc_mask[..., None]  # (b, t, 1)

        pred = self.predictor.infer(enc_out, mask=enc_mask, steps=steps)

        # durations
        no_duration_mask = duration_target < 0 if duration_target is not None else torch.zeros(1, dtype=torch.bool)
        if duration_target is None or torch.any(no_duration_mask):
            log_duration_pred = pred[..., :1].squeeze(-1)  # (b, t)
            duration_pred = duration_factor * (torch.exp(log_duration_pred) - 1)
            if not self.soft_duration:
                duration_pred = torch.round(duration_pred)
            duration_pred = torch.clamp(duration_pred, min=0)

            if duration_target is not None:
                duration_target = duration_target.float()
                duration_target[no_duration_mask] = duration_pred[no_duration_mask]
                duration_pred = duration_target
        else:
            duration_pred = duration_target

        features = []

        # pitch
        pitch = pred[..., self.pitch_idx:self.pitch_idx + 1] if pitch_target is None else pitch_target.unsqueeze(-1)
        pitch = pitch * pitch_factor + pitch_delta  # (b, t, 1)
        features.append(pitch)
        pitch = pitch.squeeze(-1)

        # energy over tokens
        energy: Tensor | None = None
        if self.energy:
            energy = pred[...,
                     self.energy_idx:self.energy_idx + 1] if energy_target is None else energy_target.unsqueeze(-1)
            energy = energy * energy_factor + energy_delta  # (b, t, 1)
            features.append(energy)
            energy = energy.squeeze(-1)

        features = torch.cat(features, dim=-1)
        enc_out = enc_out + self.embedding(features)

        # adapt encoder outputs to decoder lengths
        alignment: Tensor | None = None
        if self.soft_duration:
            if enc_mask is None:
                enc_lens = torch.full((enc_out.shape[0],), fill_value=enc_out.shape[1], device=enc_out.device)
            else:
                enc_lens = enc_mask.sum(dim=[1, 2])
            dec_lens = (duration_pred.sum(dim=1) + 0.5).long()
            mask = get_mask_3d(enc_lens, dec_lens).float()
            alignment = generate_soft_path(duration_pred, mask).transpose(1, 2)

        enc_out, dec_lens = self.length_regulator(enc_out, duration_pred, alignment=alignment)

        return TemporalAdaptorOutput(
            enc_out=enc_out,
            log_duration=None,
            duration=duration_pred,
            dec_lengths=dec_lens,
            pitch=pitch,
            energy=energy,
            pitch_target=pitch_target,
            energy_target=energy_target
        )


class LengthRegulator(nn.Module):
    def forward(
            self,
            x: Tensor,
            durations: Tensor,
            max_len: int | None = None,
            alignment: Tensor | None = None
    ):
        if alignment is not None:
            dec_lens = (durations.sum(dim=1) + 0.5).long()
            out = (x.transpose(1, 2) @ alignment.transpose(1, 2)).transpose(1, 2)
        else:
            reps = (durations.float() + 0.5).long()
            dec_lens = reps.sum(dim=1)

            reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1, dtype=x.dtype)[:, None, :]

            r = torch.arange(dec_lens.max(), device=x.device)[None, :, None]
            mult = ((reps_cumsum[:, :, :-1] <= r) & (reps_cumsum[:, :, 1:] > r)).to(x.dtype)
            out = torch.matmul(mult, x)

        if max_len is not None:
            out = out[:, :max_len]
            dec_lens = torch.clamp_max(dec_lens, max_len)

        return out, dec_lens


class TemporalAverager(nn.Module):
    def forward(
            self,
            x: Tensor,
            durations: Tensor,
            alignment: Tensor | None = None
    ):
        if alignment is not None:
            alignment_durations = alignment.sum(dim=1, keepdim=True)
            x_avg = (x @ alignment / (alignment_durations + 1e-5))
            return x_avg

        durs_cums_ends = torch.cumsum(durations, dim=1).long()
        durs_cums_starts = F.pad(durs_cums_ends[:, :-1], (1, 0))
        x_nonzero_cums = F.pad(torch.cumsum(x != 0.0, dim=2), (1, 0))
        x_cums = F.pad(torch.cumsum(x, dim=2), (1, 0))

        b, n = durs_cums_ends.size()
        n_formants = x.size(1)
        dcs = durs_cums_starts[:, None, :].expand(b, n_formants, n)
        dce = durs_cums_ends[:, None, :].expand(b, n_formants, n)

        x_sums = (torch.gather(x_cums, 2, dce) - torch.gather(x_cums, 2, dcs)).float()
        x_nelems = (torch.gather(x_nonzero_cums, 2, dce) - torch.gather(x_nonzero_cums, 2, dcs)).float()

        x_avg = torch.where(x_nelems == 0.0, x_nelems, x_sums / x_nelems)
        return x_avg


def generate_soft_path(duration, mask):
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = get_float_mask_from_lengths(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path * mask
    return path
