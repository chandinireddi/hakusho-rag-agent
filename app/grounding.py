"""Deterministic numeric grounding check (no LLM call).

The LLM fact-checker (`verify` node) can miss a single wrong digit, e.g. 「199億円」 written
as 「197億円」. In white papers the numbers are usually the most important part of the
answer, so every number in the answer is compared with the passages it cites:

* a sentence with a number but no citation      → flagged
* a number that is not in any cited passage      → flagged

Numbers are compared after NFKC normalisation (２０２３ → 2023) and after removing thousands
separators (1万7,129 → 1万7129), so formatting differences do not cause false alarms.

Known blind spot (documented in the README): a wrong number that happens to appear elsewhere
in the cited passage passes this check. The LLM judge in the `verify` node covers that case.
"""

import re
import unicodedata

_CITE = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]\s+|[-・*]\s*)")  # "1. " / "- " at the start of a line
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])|(?<=\.)\s+")


def _canonical(number: str) -> str:
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number.lstrip("0") or "0"


def extract_numbers(text: str) -> set[str]:
    text = _THOUSANDS.sub("", unicodedata.normalize("NFKC", text))
    return {_canonical(n) for n in _NUMBER.findall(text)}


def _sentences(text: str) -> list[str]:
    lines = (_LIST_MARKER.sub("", line) for line in unicodedata.normalize("NFKC", text).splitlines())
    return [s.strip() for line in lines for s in _SENTENCE_SPLIT.split(line) if s and s.strip()]


def check_numbers(answer: str, context: list, cited_ids: list[int]) -> list[str]:
    """Return one message per sentence whose numbers cannot be found in its cited passages.

    `answer` uses the renumbered citations [1]..[m]; `cited_ids[m-1]` is the 1-based
    position of that passage in `context` (see `graph.renumber_citations`)."""
    issues: list[str] = []
    for sentence in _sentences(answer):
        refs = [int(r) for r in _CITE.findall(sentence)]
        numbers = extract_numbers(_CITE.sub("", sentence))
        if not numbers:
            continue
        if not refs:
            issues.append(f"出典のない数値を含む記述: {sentence}")
            continue
        source: set[str] = set()
        for ref in refs:
            if 1 <= ref <= len(cited_ids):
                source |= extract_numbers(context[cited_ids[ref - 1] - 1].text)
        missing = sorted(numbers - source)
        if missing:
            issues.append(f"出典と一致しない数値 {', '.join(missing)}: {sentence}")
    return issues
