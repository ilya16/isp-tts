from __future__ import annotations

from typing import Iterable, Union, Callable, Sequence

from .phonemes import Phonemizer
from .cleaners import punct_corrector

class TextProcessor:
    def __init__(
            self,
            cleaners: list[str] | None = None,
            language: str = "en-us",
            phonemizer: bool = False
    ):
        self.cleaners = (cleaners or []) + [punct_corrector]
        self.language = language
        self.phonemizer = Phonemizer(language=language) if phonemizer else None

    def __call__(self, text, mask_phonemes: Union[bool, float] = False):
        for cleaner in self.cleaners:
            text = cleaner(text)

        text = text.lower()

        if self.phonemizer is not None:
            text = self.phonemizer(text, mask_phonemes=mask_phonemes)

        return text