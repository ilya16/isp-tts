import os
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


class Field(str, Enum):
    audio_path = "audio_path"
    text = "text"
    mel = "mel"
    pitch = "pitch"
    energy = "energy"
    speaker = "speaker"

    @classmethod
    def values(cls):
        return [f.value for f in cls]

    @classmethod
    def nested(cls, value):
        return value in cls.values()


class TTSMeta(np.ndarray):
    sep = "|"

    def __new__(cls, data):
        obj = np.asarray(data).view(cls)
        obj._fields = obj.dtype.names
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._fields = getattr(obj, '_fields', None)

    @property
    def fields(self):
        return self.dtype.names

    def tolist(self):
        return ["|".join((str(p) for p in elem)) for elem in self]

    @classmethod
    def load(cls, source, fields=(Field.audio_path, Field.text)):
        source = Path(source)

        if not all([Field.nested(field) for field in fields]):
            raise ValueError(f"Unsupported field names were encountered. "
                             f"Got {fields}, supported fields: {Field.values()}")

        data = np.genfromtxt(
            fname=source,
            delimiter=TTSMeta.sep,
            names=list(fields),
            autostrip=True,
            encoding="utf-8",
            dtype=None
        )

        return TTSMeta(data)

    def save(self, path):
        np.savetxt(
            path,
            self,
            delimiter=TTSMeta.sep,
            fmt='%s'
        )

    def filter_length(self, field, minimum: int = 0, maximum: int = 1e3):
        assert field in self.fields

        lengths = np.vectorize(len)(self[field])
        mask = (lengths > minimum) & (lengths < maximum)

        return self[mask]

    def filter_audio_length(self, root_path: str, minimum: float = 0., maximum: float = 60):
        assert Field.audio_path in self._fields

        def _audio_length(audio_path, root_path):
            import torchaudio
            audio_info = torchaudio.info(os.path.join(root_path, audio_path), backend="soundfile")
            return audio_info.num_frames / audio_info.sample_rate

        audio_length = partial(_audio_length, root_path=root_path)

        def wrapped(audio_path, pbar):
            pbar.update(1)
            return audio_length(audio_path)

        with tqdm(total=len(self), leave=False) as pbar:
            lengths = np.vectorize(wrapped)(self[Field.audio_path], pbar)

        mask = (lengths > minimum) & (lengths < maximum)

        return self[mask]
