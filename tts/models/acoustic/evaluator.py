""" AcousticModel metrics evaluator. """
from __future__ import annotations

import numpy as np
import torch
from torchaudio.functional import create_dct

from .model import AcousticModelInput, AcousticModelOutput, AcousticModel
from ...data.collator import AcousticInputs
from ...utils import get_mask_from_lengths
from ...utils.plotting import plot_attention, plot_spectrogram


class MCD(torch.nn.Module):
    """Mel-cepstral distortion (MCD).

    Computes MCD for a batch of aligned mel spectrograms.
    """
    _logdb_const = 10.0 * np.sqrt(2.0) / np.log(10.0)

    def __init__(self, n_mel_channels: int = 80, n_mfcc: int = 13):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_mfcc = n_mfcc  # number of mel-frequency cepstral coefficients
        self._dct_matrix = create_dct(self.n_mfcc, self.n_mel_channels, norm='ortho')

    def _mfcc(self, mels):
        if mels.size(-1) != self.n_mel_channels:
            mels = mels.transpose(-2, -1)
        return mels @ self._dct_matrix.to(dtype=mels.dtype, device=mels.device)

    def _mcd(self, x, y, lengths):
        return self._logdb_const * torch.sqrt(((x - y) ** 2).sum(dim=2)).sum(dim=1) / lengths

    def forward(self, mels_out, mels_target, mel_lengths):
        mfcc_out = self._mfcc(mels_out)[..., 1:]  # zero channel is energy
        mfcc_target = self._mfcc(mels_target)[..., 1:]

        mcd = self._mcd(mfcc_out, mfcc_target, mel_lengths)
        return mcd.mean()


class AlignmentMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alignments, mel_lengths, text_lengths):
        # alignments: (batch_size, mel_len, text_len)
        batch_size, device = alignments.size(0), alignments.device

        max_indices = alignments.argmax(dim=2)
        max_indices_diff = max_indices[:, 1:] - max_indices[:, :-1]

        mask = get_mask_from_lengths(mel_lengths)[:, 1:]
        steps_all = (1 + max_indices_diff[mask] ** 2).float().pow(0.5)

        ind = torch.arange(batch_size, device=device).repeat_interleave(mel_lengths - 1)
        lengths = torch.zeros(batch_size, device=device).index_add_(0, ind, steps_all)

        diagonal_lengths = torch.sqrt(text_lengths.float() ** 2 + mel_lengths.float() ** 2)
        alignment_length = (lengths / diagonal_lengths).mean()
        alignment_strength = alignments.max(dim=2).values.sum() / mel_lengths.sum()

        return alignment_length, alignment_strength


class AcousticModelEvaluator:
    def __init__(self, model: AcousticModel):
        self.model = model

        self.mcd_evaluator = MCD()
        self.alignment_evaluator = AlignmentMetric()

    @torch.no_grad()
    def __call__(self, inputs: dict | AcousticModelInput, outputs: dict | AcousticModelOutput):
        if isinstance(inputs, dict):
            inputs = AcousticModelInput(**inputs)

        if isinstance(outputs, dict):
            outputs = AcousticModelOutput(**outputs)

        mcd = self.mcd_evaluator(
            mels_out=outputs.mel,
            mels_target=inputs.mel,
            mel_lengths=inputs.mel_len
        )

        alignment_length, alignment_strength = self.alignment_evaluator(
            alignments=outputs.aligner_output.attn_soft,
            mel_lengths=inputs.mel_len,
            text_lengths=inputs.text_len
        )

        metrics = {
            f"metrics/mcd_{self.mcd_evaluator.n_mfcc}": mcd,
            "metrics/alignment_length": alignment_length,
            "metrics/alignment_strength": alignment_strength,
        }

        return metrics

    @torch.no_grad()
    def on_eval_epoch_end(
            self,
            inputs: dict | AcousticInputs,
            outputs: dict | AcousticModelOutput
    ):
        if isinstance(inputs, dict):
            inputs = AcousticInputs(**inputs)

        if isinstance(outputs, dict):
            outputs = AcousticModelOutput(**outputs)

        idx = 0
        name = inputs.filename[idx]

        text_len = inputs.text_vector_len[idx]
        mel_spec_len = inputs.mel_len[idx]

        alignment_dict = {
            "soft": outputs.aligner_output.attn_soft[idx, :mel_spec_len, :text_len].T,
            "hard": outputs.aligner_output.attn_hard[idx, :mel_spec_len, :text_len].T
        }

        mel = inputs.mel[idx, :, :mel_spec_len]
        clamp_mel = lambda x: x.clamp(min=mel.min(), max=mel.max())
        spec_dict = {
            "target": mel,
            "predicted": clamp_mel(outputs.mel[idx, :, :mel_spec_len])
        }

        image_dict = {
            "images/eval/alignment": plot_attention(alignment_dict, title=name),
            "images/eval/mel_spectrogram": plot_spectrogram(spec_dict, title=name)
        }

        return image_dict
