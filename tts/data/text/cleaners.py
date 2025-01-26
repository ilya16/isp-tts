"""
Source: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/text/cleaners.py
"""

import re

from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\{\}]")

# Simplify quotes and hyphens
_redundant_quotes_re = re.compile(r"(“|”|„|«|»)")
_redundant_hyphen_re = re.compile(r"(‐|‑|‒|–|―)")

# Correct punctuation
_wrong_dot_re = re.compile(r"[, \n]+\.$")
_wrong_punct_re = re.compile(r"(^[?!., -]+)")


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def punct_corrector(text):
    text = text.strip()

    text = _brackets_re.sub("", text)

    text = _redundant_hyphen_re.sub("-", text)
    text = _redundant_quotes_re.sub('"', text)

    text = text.replace("…", "...")

    text = _wrong_dot_re.sub(".", text)
    text = _wrong_punct_re.sub("", text)

    text = collapse_whitespace(text)

    return text