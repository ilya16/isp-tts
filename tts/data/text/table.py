import re
from collections import OrderedDict
from string import punctuation
from typing import List, Union, Sequence

from loguru import logger

from . import symbols as sym

_CHAR_SPLIT_PATTERN = re.compile(rf"(</?\w+>|[{punctuation}]|\w)")


def re_split(text, pattern):
    return [t for t in pattern.split(text) if t]


def split_to_chars(text, split_pattern=None):
    return re_split(text, _CHAR_SPLIT_PATTERN if split_pattern is None else split_pattern)


charset_map = {
    "#punct": sym.PUNCTUATION,
    "#marks": sym.MARKS,
    "#ru": sym.RUSSIAN,
    "#en": sym.ENGLISH,
    "#ipa_ph": sym.IPA_PHONEMES
}


class CodingTable:
    def __init__(
            self,
            vocab: tuple,
            pad: str = None,
            eos: str = None
    ):
        _service = tuple(item for item in (pad, eos) if item)
        for item in _service:
            assert item.startswith("<") and item.endswith(">"), "Service tokens must be formatted as xml-tags."

        self.pad = pad
        self.eos = eos

        self.service = _service
        self.vocab = vocab

        self.encoding_map = OrderedDict({label: idx for idx, label in enumerate(self.service + vocab)})
        self.decoding_map = OrderedDict({idx: label for label, idx in self.encoding_map.items()})

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.encode(item)
        elif isinstance(item, int):
            return self.decode(item)

    def __iter__(self):
        yield from self.encoding_map.keys()

    def __len__(self):
        return len(self.encoding_map)

    def __repr__(self):
        return " ".join(list(self.charset))

    @property
    def charset(self):
        return tuple(self.encoding_map.keys())

    def encode(self, char: str):
        return self.encoding_map[char]

    def decode(self, idx: int):
        return self.decoding_map[idx]

    def text_to_vector(self, text: str):
        not_valid = set()
        vector = []
        for s in split_to_chars(text):
            if s in self.encoding_map:
                vector.append(self[s])
            else:
                not_valid.add(s)

        if not_valid:
            logger.warning(f"The coding table does not contain the following characters: {not_valid}")

        return vector

    def vector_to_text(self, vector: list[int]):
        text = ""
        for elem_idx in vector:
            elem = self[elem_idx]
            text += elem
        return text

    def check_eos(self, text: str):
        if self.eos is not None and not text.endswith(self.eos):
            return text + self.eos
        return text

    @staticmethod
    def from_charset(charset: Union[str, Sequence[str]]):
        charset = charset if isinstance(charset, Sequence) else [charset]

        kwargs_mapping = {sym.PAD: "pad", sym.EOS: "eos"}

        vocab, extra = [], []
        kwargs = {"pad": None, "eos": None}

        for item in charset:
            if item.startswith("<") and item.endswith(">"):
                if item in kwargs_mapping:
                    kwargs[kwargs_mapping[item]] = item
                else:
                    extra.append(item)
            else:
                try:
                    _charset = charset_map[item]
                except KeyError:
                    _charset = list(item)

                vocab.extend(_charset)
        unique_vocab = list(OrderedDict((c, None) for c in vocab))

        return CodingTable(
            vocab=tuple(unique_vocab),
            **kwargs
        )
