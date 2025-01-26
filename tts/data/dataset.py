from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from . import providers as dp
from .functions import remove_outliers, FeatureStats, StandardScaler
from .metadata import TTSMeta, Field


@dataclass
class AcousticDatasetStats:
    pitch: FeatureStats = field(default_factory=lambda: FeatureStats())
    energy: FeatureStats = field(default_factory=lambda: FeatureStats())

    def to_dict(self):
        return {
            Field.pitch: self.pitch.to_dict(),
            Field.energy: self.energy.to_dict()
        }

    @staticmethod
    def from_dict(dct):
        return AcousticDatasetStats(
            pitch=FeatureStats(*dct.get(Field.pitch, FeatureStats().to_dict()).values()),
            energy=FeatureStats(*dct.get(Field.energy, FeatureStats().to_dict()).values())
        )

    def reset(self):
        self.pitch = FeatureStats()
        self.energy = FeatureStats()


@dataclass
class AcousticSample:
    filename: str
    text: str
    text_vector: torch.Tensor
    text_vector_len: torch.Tensor

    mel: torch.Tensor
    mel_len: torch.Tensor

    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    speaker: Optional[torch.Tensor] = None


class AcousticDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            meta_name: str,
            meta_column_names: list[str],
            text: DictConfig,
            audio: DictConfig,
            spec: DictConfig,
            mel_scale: DictConfig,
            pitch: Optional[DictConfig] = None,
            energy: Optional[DictConfig] = None,
            speaker: Optional[Union[str, dict[str, int]]] = None,
            text_length_limits: Optional[tuple[int, int]] = None,
            audio_length_limits: Optional[tuple[float, float]] = None,
            pitch_from_disk: bool = False,
            stats: AcousticDatasetStats | dict[str, dict[str, float]] | str | None = None
    ):
        self.data_root = root

        meta_path = os.path.join(root, meta_name)
        meta = TTSMeta.load(meta_path, meta_column_names)

        if text_length_limits is not None:
            meta = meta.filter_length(Field.text, *list(text_length_limits))
            logger.debug(f"text_length_limits: {len(meta)} samples")

        if audio_length_limits is not None:
            meta = meta.filter_audio_length(root, *list(audio_length_limits))
            logger.debug(f"audio_length_limits: {len(meta)} samples")

        self.meta = meta

        self.text_provider = dp.TextProvider.init(text)

        self.audio_provider = dp.AudioProvider.init(audio)
        self.spec_provider = dp.SpectrogramProvider.init(spec)
        self.mel_scale_provider = dp.MelScaleProvider.init(mel_scale)

        self.pitch_provider = dp.PitchProvider.init(pitch) if pitch is not None else None
        self.energy_provider = dp.EnergyProvider.init(energy) if energy is not None else None

        self.pitch_from_disk = pitch_from_disk

        if not self.pitch_from_disk:
            assert self.audio_provider is not None

        self.speaker_map = None
        if speaker is not None:
            if isinstance(speaker, str):
                with open(speaker, encoding="utf-8") as f:
                    speaker = json.load(f)
            elif isinstance(speaker, (DictConfig, dict)):
                speaker = dict(speaker)

            self.speaker_map = speaker

        if stats is None:
            self.stats = AcousticDatasetStats()
        elif isinstance(stats, AcousticDatasetStats):
            self.stats = stats
        elif isinstance(stats, (dict, DictConfig)):
            self.stats = AcousticDatasetStats(
                pitch=FeatureStats(**stats.get("pitch", {})),
                energy=FeatureStats(**stats.get("energy", {}))
            )
        else:
            with open(stats, "r") as f:
                stats_dict = json.load(f)
                self.stats = AcousticDatasetStats.from_dict(stats_dict)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, item):
        sample = self.meta[item]

        text_data = self.text_provider(sample[Field.text])

        audio_data = self.audio_provider(Path(self.data_root, sample[Field.audio_path]))

        filename = sample[Field.audio_path]
        spec = self.spec_provider(audio_data)

        if self.pitch_from_disk:
            source_pitch = Path(self.data_root, sample[Field.pitch]) if self.pitch_provider is not None else None
        else:
            source_pitch = audio_data

        mel = self.mel_scale_provider(spec)

        pitch = None
        if self.pitch_provider is not None:
            pitch = self.pitch_provider(source_pitch, self.stats.pitch.mean, self.stats.pitch.std)
            pitch = torch.nn.functional.pad(pitch, (0, mel.shape[1] - pitch.size(0)))

        energy = None
        if self.energy_provider is not None:
            energy = self.energy_provider(spec)

        speaker = None
        if self.speaker_map is not None:
            speaker = self.speaker_map[sample[Field.speaker]]

        return AcousticSample(
            filename=filename,
            text=text_data.string,
            text_vector=text_data.vector,
            text_vector_len=text_data.vector_len,
            mel=mel,
            mel_len=mel.size(1),
            pitch=pitch,
            energy=energy,
            speaker=speaker
        )

    def compute_stats(self, disable_tqdm: bool = True, save_stats: bool = True, save_path: Optional[str] = None):
        self.stats.reset()

        pitch_min = energy_min = np.finfo(np.float64).max
        pitch_max = energy_max = np.finfo(np.float64).min

        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        iterator = self if disable_tqdm else tqdm(self, desc="Computing dataset statistics")
        for inputs in iterator:
            pitch = remove_outliers(inputs.pitch)
            pitch = pitch[pitch > 0.]
            if len(pitch) > 0:
                pitch_scaler.partial_fit(pitch.numpy().reshape((-1, 1)))
                pitch_min = min(pitch_min, pitch.min().item())
                pitch_max = max(pitch_max, pitch.max().item())

            energy = remove_outliers(inputs.energy)
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.numpy().reshape((-1, 1)))
                energy_min = min(energy_min, energy.min().item())
                energy_max = max(energy_max, energy.max().item())

        pitch_mean, pitch_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
        energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]

        self.stats = AcousticDatasetStats(
            pitch=FeatureStats(
                min=pitch_min,
                max=pitch_max,
                mean=pitch_mean,
                std=pitch_std
            ),
            energy=FeatureStats(
                min=energy_min,
                max=energy_max,
                mean=energy_mean,
                std=energy_std
            )
        )

        if save_stats:
            save_path = save_path or os.path.join(self.data_root, "stats.json")
            with open(save_path, "w") as f:
                f.write(json.dumps(self.stats.to_dict()))

        return self.stats
