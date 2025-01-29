from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, NamedTuple

import torch
import torchaudio
from omegaconf import MISSING
from torchaudio import transforms as T

from tts.data import functions as F
from tts.data.text.processor import TextProcessor
from tts.data.text.table import CodingTable
from tts.modules.constructor import Constructor, ModuleConfig
from tts.utils import prob2bool


class _Provider(Constructor):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...


@dataclass
class SpectrogramProviderConfig(ModuleConfig):
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    pad: Optional[int] = None
    power: Optional[float] = 1.
    normalized: bool = False
    center: bool = False


class SpectrogramProvider(T.Spectrogram, _Provider):
    def __init__(
            self,
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: int = 1024,
            pad: Optional[int] = None,
            power: Optional[float] = 1.,
            normalized: bool = False,
            center: bool = False
    ):
        super().__init__(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad if pad is not None else int((n_fft - hop_length) / 2),
            power=power,
            normalized=normalized,
            center=center
        )

    def __call__(self, source: Union[Path, torch.Tensor]):
        if isinstance(source, Path):  # spec tensor
            spec = F.load_array(source)
        elif isinstance(source, torch.Tensor):  # audio tensor
            assert source.ndim <= 2
            spec = super().__call__(source)
        else:
            raise ValueError

        return spec


@dataclass
class MelScaleProviderConfig(ModuleConfig):
    sample_rate: int
    n_fft: int = 1024
    n_mels: int = 80
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    norm: Optional[str] = "slaney"
    mel_scale: str = "slaney"


class MelScaleProvider(T.MelScale, _Provider):
    def __init__(
            self,
            sample_rate: int,
            n_fft: int = 1024,
            n_mels: int = 80,
            f_min: float = 0.0,
            f_max: Optional[float] = 8000.0,
            norm: Optional[str] = "slaney",
            mel_scale: str = "slaney"
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            norm=norm,
            mel_scale=mel_scale
        )

    def __call__(self, source: Union[Path, torch.Tensor]):
        if isinstance(source, Path):  # mel spec tensor
            spec = F.load_array(source)
        elif isinstance(source, torch.Tensor):  # spec tensor
            assert source.ndim == 2
            spec = F.dynamic_range_compression(
                super().__call__(source)
            )
        else:
            raise ValueError

        return spec


@dataclass
class MelSpecProviderConfig(ModuleConfig):
    sample_rate: int
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: float = 0.0
    f_max: Optional[float] = 8000.0
    pad: Optional[int] = None
    n_mels: int = 80
    power: Optional[float] = 1.
    normalized: bool = False
    center: bool = False
    norm: Optional[str] = "slaney"
    mel_scale: str = "slaney"


class MelSpecProvider(T.MelSpectrogram, _Provider):
    def __init__(
            self,
            sample_rate: int,
            n_fft: int = 400,
            win_length: Optional[int] = None,
            hop_length: Optional[int] = None,
            f_min: float = 0.0,
            f_max: Optional[float] = None,
            pad: int = 0,
            n_mels: int = 128,
            power: float = 2.0,
            normalized: bool = False,
            center: bool = True,
            norm: Optional[str] = None,
            mel_scale: str = "htk",
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad if pad is not None else int((n_fft - hop_length) / 2),
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=center,
            norm=norm,
            mel_scale=mel_scale,
        )

    def __call__(self, source: Union[Path, torch.Tensor]):
        if isinstance(source, Path):  # mel spec tensor
            spec = F.load_array(source)
        elif isinstance(source, torch.Tensor):  # audio tensor
            assert source.ndim <= 2
            spec = F.dynamic_range_compression(
                super().__call__(source)
            )
        else:
            raise ValueError

        return spec


class EnergyProvider(_Provider):
    def __call__(self, source: Union[Path, torch.Tensor]):
        if isinstance(source, Path):
            energy = F.load_array(source)
        elif isinstance(source, torch.Tensor):
            assert source.ndim == 2
            energy = torch.norm(source, dim=0)
        else:
            raise ValueError

        return torch.log1p(energy)


@dataclass
class AudioProviderConfig(ModuleConfig):
    sample_rate: int = MISSING


class AudioProvider(_Provider):
    def __init__(
            self,
            sample_rate: int = MISSING
    ):
        self.sample_rate = sample_rate

    def __call__(self, file_path: Path):
        audio, rate = torchaudio.load(file_path, backend="soundfile")

        if rate != self.sample_rate:
            audio = T.Resample(rate, self.sample_rate, dtype=audio.dtype)(audio)

        if audio.shape[0] != 1:
            audio = audio.mean(0)

        return audio.squeeze()


class TextData(NamedTuple):
    string: Union[str, list[str]]
    vector: torch.Tensor
    vector_len: torch.Tensor


@dataclass
class TextProviderConfig(ModuleConfig):
    charset: list[str]
    phonemizer: bool = False
    mask_phonemes: Union[bool, float] = False
    word_level_prob: bool = True


class TextProvider(_Provider):
    def __init__(
            self,
            charset: list[str],
            phonemizer: bool = False,
            mask_phonemes: Union[bool, float] = False,
            word_level_prob: bool = True
    ):
        self.coding_table = CodingTable.from_charset(charset)

        self.mask_phonemes = mask_phonemes
        self.word_level_prob = word_level_prob

        self.text_processor = TextProcessor(phonemizer=phonemizer)

    def __call__(self, text: str):
        options = dict(
            mask_phonemes=self._prob2bool(self.mask_phonemes, self.word_level_prob),
        )

        preprocessed_text = self.text_processor(text, **options)
        preprocessed_text = self.coding_table.check_eos(preprocessed_text)

        text_vector = self.coding_table.text_to_vector(preprocessed_text)
        text_vector = torch.tensor(text_vector, dtype=torch.long)

        return TextData(
            string=text,
            vector=text_vector,
            vector_len=text_vector.size(0),
        )

    @staticmethod
    def _prob2bool(prob, word_level):
        return prob2bool(prob) if not word_level else prob


@dataclass
class PitchProviderConfig(ModuleConfig):
    sample_rate: int = MISSING
    hop_length: int = 256
    win_length: int = 1024
    f_min: int = 40
    f_max: int = 800
    method: str = "torch-yin"
    center: bool = True
    pad: Optional[int] = None
    threshold: float = 0.15
    norm: str = "standard"
    device: int = None


class PitchProvider(_Provider):
    def __init__(
            self,
            sample_rate: int,
            hop_length: int = 256,
            win_length: int = 1024,
            f_min: int = 40,
            f_max: int = 800,
            method: str = "torch-yin",
            center: bool = True,
            pad: Optional[int] = None,
            threshold: float = 0.15,
            norm: str = "standard",
            device: int = None
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_shift = hop_length / self.sample_rate
        if method == "torch-yin":
            f_min = 2 * int(sample_rate / win_length)  # for torch-yin to match mel_spec frames
        self.f_min = f_min
        self.f_max = f_max

        self.method = method
        self.threshold = threshold

        self.center = center
        self.pad = pad if pad is not None else int((win_length - hop_length) / 2)

        assert norm in ("standard", "log")
        self.norm = norm
        self.device = device

    def __call__(self, source: Union[Path, torch.Tensor], mean: float = 0.0, std: float = 1.0):
        if isinstance(source, Path):
            pitch = F.load_array(source)
        elif isinstance(source, torch.Tensor):
            if self.method == "torch-yin":
                source = torch.nn.functional.pad(source, (self.pad, self.pad), "constant")
                from .pitch import pitch_yin
                pitch = pitch_yin(
                    source,
                    sample_rate=self.sample_rate,
                    pitch_min=self.f_min,
                    pitch_max=self.f_max,
                    frame_stride=self.frame_shift,
                    threshold=self.threshold
                )
            elif self.method == "penn":
                import penn
                pitch, periodicity = penn.from_audio(
                    source[None],
                    self.sample_rate,
                    hopsize=self.frame_shift,
                    fmin=self.f_min,
                    fmax=self.f_max,
                    batch_size=128,
                    center='zero' if self.center else 'half-window',
                    gpu=self.device
                )
                pitch, periodicity = map(lambda x: x[0].float(), (pitch, periodicity))
                pitch[periodicity < self.threshold] = 0.
            else:
                raise ValueError
        else:
            raise ValueError

        return (pitch - mean) / std
