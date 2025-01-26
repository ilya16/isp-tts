from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, MISSING
from torch import Tensor

from tts.models.base import Model
from tts.modules.constructor import ModuleConfig
from tts.modules.transformer import Transformer, TransformerConfig
from .modules import (
    TemporalAdaptorOutput, FlowTemporalAdaptorConfig, FlowTemporalAdaptor,
    Aligner, AlignerConfig, AlignerOutput
)
from ...data.collator import AcousticInputs
from ...data.dataset import AcousticDataset
from ...utils import get_mask_from_lengths
from ...utils.config import asdict


class AcousticModelInput(NamedTuple):
    text: Tensor
    text_len: Tensor
    mel: Tensor
    mel_len: int | Tensor

    pitch: Tensor | None = None
    energy: Tensor | None = None
    speaker: Tensor | None = None


class AcousticModelOutput(NamedTuple):
    mel: Tensor
    adaptor_output: TemporalAdaptorOutput
    aligner_output: AlignerOutput

    loss: Tensor | None = None
    losses: dict[str, Tensor] | None = None


@dataclass
class AcousticModelConfig(ModuleConfig):
    encoding_map: dict = MISSING
    mel_dim: int = MISSING
    text_dim: int = 384
    encoder: TransformerConfig | DictConfig = field(default_factory=lambda: TransformerConfig())
    decoder: TransformerConfig | DictConfig = field(default_factory=lambda: TransformerConfig())
    temporal_adaptor: FlowTemporalAdaptorConfig | DictConfig = field(
        default_factory=lambda: FlowTemporalAdaptorConfig()
    )
    aligner: AlignerConfig | DictConfig = field(default_factory=lambda: AlignerConfig())
    num_speakers: int | None = 0
    pitch_mean: int | None = None
    pitch_std: int | None = None


class AcousticModel(Model):
    def __init__(
            self,
            encoding_map: dict,
            mel_dim: int,
            text_dim: int = 384,
            encoder: TransformerConfig | DictConfig = TransformerConfig(),
            decoder: TransformerConfig | DictConfig = TransformerConfig(),
            temporal_adaptor: FlowTemporalAdaptorConfig | DictConfig = FlowTemporalAdaptorConfig(),
            aligner: AlignerConfig | DictConfig = AlignerConfig(),
            num_speakers: int | None = 0,
            pitch_mean: int | None = None,
            pitch_std: int | None = None
    ):
        super().__init__()

        self.encoding_map = dict(encoding_map)
        self.mel_dim = mel_dim
        self.text_dim = text_dim
        self.text_embedding = nn.Embedding(len(encoding_map), text_dim, padding_idx=0)

        self.encoder = Transformer.init(
            config=encoder,
            emb_dim=text_dim
        )
        self._encoder_output_dim = encoder.dim

        self.aligner = Aligner.init(
            config=aligner,
            mel_dim=mel_dim,
            text_dim=encoder.dim
        )

        num_speakers = num_speakers or 0
        self.speaker_embedding = None
        if num_speakers > 0:
            self.speaker_embedding = nn.Embedding(num_speakers, encoder.dim)
            torch.nn.init.xavier_uniform_(self.speaker_embedding.weight)

        self.temporal_adaptor = FlowTemporalAdaptor.init(
            temporal_adaptor,
            encoder_dim=self._encoder_output_dim,
        )

        self.decoder = Transformer.init(
            config=decoder,
            emb_dim=self._encoder_output_dim
        )

        self.to_mel = nn.Linear(decoder.dim, mel_dim)

        # store values precomputed for training data within the model
        self.register_buffer("pitch_mean", torch.tensor(pitch_mean or 0.))
        self.register_buffer("pitch_std", torch.tensor(pitch_std or 1.))

    @torch.jit.unused
    def forward(
            self,
            text: Tensor,
            text_len: int | Tensor,
            mel: Tensor,
            mel_len: int | Tensor,

            pitch: Tensor | None = None,
            energy: Tensor | None = None,

            speaker: Tensor | None = None,
            sigma: float = 0.,
            steps: int = 1
    ):
        # token embedding
        token_emb = self.text_embedding(text)  # (B, L) -> (B, L, emb_dim)

        # encoder
        enc_mask = get_mask_from_lengths(text_len)
        enc_out = self.encoder(token_emb, mask=enc_mask).out

        # alignment
        aligner_output = self.aligner(
            mel=mel, enc_text=enc_out.transpose(1, 2).detach(),
            mel_len=mel_len, text_len=text_len
        )
        duration_target = aligner_output.attn_hard_duration

        # speaker embedding
        if self.speaker_embedding is not None:
            enc_out = enc_out + self.speaker_encoder(speaker)

        # temporal adaptor with prosody features
        adaptor_output = self.temporal_adaptor(
            enc_out=enc_out,
            enc_mask=enc_mask,
            max_dec_len=mel.size(2),
            duration_target=duration_target,
            alignment=aligner_output.attn_soft,
            pitch_target_dense=pitch,
            energy_target_dense=energy
        )

        # decoder
        dec_mask = get_mask_from_lengths(adaptor_output.dec_lengths)
        dec_out = self.decoder(
            adaptor_output.enc_out,
            mask=dec_mask
        ).out

        # transform to mels
        mel_out = self.to_mel(dec_out).transpose(1, 2)
        mel_out = mel_out * dec_mask[:, None]

        return AcousticModelOutput(
            mel=mel_out,
            adaptor_output=adaptor_output,
            aligner_output=aligner_output
        )

    @torch.jit.export
    def infer(
            self,
            input_sequence: Tensor,
            text_lengths: Tensor | None = None,
            duration_target: Tensor | None = None,
            duration_factor: float = 1.0,
            pitch_target: Tensor | None = None,
            pitch_factor: float = 1.0,
            pitch_delta: float = 0.,
            pitch_normalize: bool = False,
            energy_target: Tensor | None = None,
            steps: int = 4,
            speaker: Tensor | None = None
    ):
        batch_infer = input_sequence.shape[0] > 1

        # token embedding
        token_emb = self.text_embedding(input_sequence)

        # encoder
        enc_mask: Tensor | None = None
        if batch_infer:
            if text_lengths is None:
                text_lengths = torch.tensor([input_sequence.shape[1]], device=input_sequence.device)[None]
            enc_mask = get_mask_from_lengths(text_lengths)

        enc_out = self.encoder(token_emb, mask=enc_mask).out

        # speaker embeddings
        if self.speaker_embedding is not None and speaker is not None:
            enc_out = enc_out + self.speaker_embedding(speaker)

        # temporal adaptor with prosody features
        if pitch_normalize:
            if pitch_target is not None:
                pitch_target = (pitch_target - self.pitch_mean) / self.pitch_std
            pitch_delta /= self.pitch_std

        adaptor_output = self.temporal_adaptor.infer(
            enc_out=enc_out,
            enc_mask=enc_mask,
            duration_target=duration_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            duration_factor=duration_factor,
            pitch_factor=pitch_factor,
            pitch_delta=pitch_delta,
            steps=steps
        )

        # decoder
        dec_mask = get_mask_from_lengths(adaptor_output.dec_lengths) if batch_infer else None
        dec_out = self.decoder(
            adaptor_output.enc_out,
            mask=dec_mask
        ).out

        # transform to mels
        mel_out = self.to_mel(dec_out).transpose(1, 2)
        mel_out = mel_out * dec_mask[:, None] if dec_mask is not None else mel_out

        return mel_out, adaptor_output

    @staticmethod
    def get_criterion(criterion: DictConfig):
        from .loss import AcousticModelLoss
        return AcousticModelLoss.init(criterion)

    def prepare_inputs(self, inputs: dict | AcousticInputs):
        if isinstance(inputs, AcousticInputs):
            inputs = asdict(inputs)

        inputs_dict = {
            "text": inputs["text_vector"],
            "text_len": inputs["text_vector_len"],
            "mel": inputs["mel"],
            "mel_len": inputs["mel_len"],
            "pitch": inputs["pitch"],
            "energy": inputs["energy"],
            "speaker": inputs["speaker"],
        }

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: DictConfig | AcousticModelConfig | None,
            dataset: AcousticDataset | None
    ) -> DictConfig | ModuleConfig | None:
        assert isinstance(dataset, AcousticDataset)

        config["encoding_map"] = dict(dataset.text_provider.coding_table.encoding_map)
        if dataset.stats is not None:
            config["pitch_mean"] = dataset.stats.pitch.mean
            config["pitch_std"] = dataset.stats.pitch.std

        return config
