"""Dependency-free tokenizer for mixed Japanese / English text (used by BM25).

* NFKC normalisation   → full-width "２０２３" and half-width "2023" become identical.
* CJK character bigrams → robust for kanji/kana without MeCab or Sudachi.
* ASCII word tokens    → "GX", "COP15", "Web3" stay as single tokens.
"""

import re
import unicodedata

_CJK = "぀-ヿ㐀-䶿一-鿿ー々"  # kana, kanji, ー, 々
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])")
_CJK_RUN = re.compile(rf"[{_CJK}]+")
_ASCII_WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def normalize(text: str) -> str:
    """NFKC + lowercase, and join CJK characters split by PDF line breaks/spacing."""
    text = unicodedata.normalize("NFKC", text).lower()
    return _CJK_GAP.sub("", text)


def tokenize(text: str) -> list[str]:
    text = normalize(text)
    tokens = _ASCII_WORD.findall(text)
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens
