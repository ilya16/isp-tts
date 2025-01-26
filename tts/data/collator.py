from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from tts.data.dataset import AcousticSample
from tts.utils.config import asdict


@dataclass
class AcousticInputs:
    filename: list[str]
    text: list[str]
    text_vector: torch.Tensor
    text_vector_len: torch.Tensor

    mel: torch.Tensor
    mel_len: torch.Tensor

    pitch: Optional[torch.Tensor] = None
    energy: Optional[torch.Tensor] = None
    speaker: Optional[torch.Tensor] = None


class AcousticCollator:
    @staticmethod
    def reserve(batch: Sequence[AcousticSample]) -> AcousticInputs:
        batch_size = len(batch)

        reference = batch[0]
        num_mels = reference.mel.size(0)

        filenames = [elem.filename for elem in batch]
        texts = [elem.text for elem in batch]

        text_vector_len = torch.tensor([x.text_vector_len for x in batch], dtype=torch.long)
        max_text_vec_len = text_vector_len.max()
        if reference.text_vector.ndim == 2:
            text_vectors = torch.zeros(
                size=(batch_size, max_text_vec_len, reference.text_vector.shape[-1]), dtype=torch.long
            )
        else:
            text_vectors = torch.zeros(size=(batch_size, max_text_vec_len), dtype=torch.long)

        mel_lengths = torch.tensor([x.mel_len for x in batch], dtype=torch.long)
        max_mel_len = mel_lengths.max()
        mels = torch.zeros(size=(batch_size, num_mels, max_mel_len), dtype=torch.float)

        pitch = None
        if reference.pitch is not None:
            pitch = torch.zeros(size=(batch_size, max_mel_len), dtype=torch.float)

        energy = None
        if reference.energy is not None:
            energy = torch.zeros(size=(batch_size, max_mel_len), dtype=torch.float)

        speaker = None
        if reference.speaker is not None:
            speaker = torch.zeros(size=(batch_size, 1), dtype=torch.long)

        return AcousticInputs(
            filename=filenames,
            text=texts,
            text_vector=text_vectors,
            text_vector_len=text_vector_len,
            mel=mels,
            mel_len=mel_lengths,
            pitch=pitch,
            energy=energy,
            speaker=speaker
        )

    @staticmethod
    def process_sample(idx: int, sample: AcousticSample, data: AcousticInputs):
        data.text_vector[idx, :sample.text_vector_len] = sample.text_vector
        data.text_vector_len[idx] = sample.text_vector_len

        data.mel[idx, :, :sample.mel_len] = sample.mel
        data.mel_len[idx] = sample.mel_len

        if data.pitch is not None:
            data.pitch[idx, :sample.mel_len] = sample.pitch

        if data.energy is not None:
            data.energy[idx, :sample.mel_len] = sample.energy

        if data.speaker is not None:
            data.energy[idx] = sample.speaker

    def __call__(self, batch: Sequence[AcousticSample], return_dict: bool = True):
        data = self.reserve(batch)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return asdict(data) if return_dict else data
