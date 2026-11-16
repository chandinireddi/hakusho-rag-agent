"""Fault-injection benchmark for the answer verifier (no new data needed).

Question: "If the LLM writes one wrong digit, which verifier notices?"

1. Take real sentences that contain numbers from the golden-set pages (the correct source).
2. Change exactly one digit (e.g. 199億円 → 197億円) to create a realistic hallucination.
3. Run both verifiers on the correct and the corrupted sentence:
     * LLM judge   (the `verify` node prompt, gpt-4o-mini)  — skipped with --offline
     * Rule check  (app/grounding.py, deterministic)
4. Report a confusion matrix (positive = corrupted sentence) with Precision / Recall / F1
   and the false-alarm rate (correct sentence flagged) — the same metrics used for
   visual-inspection models.

Two difficulty levels are measured, because the easy one flatters the rule check:
* easy : the wrong number appears nowhere in the source          (e.g. 199 → 197)
* hard : the wrong number is another number from the SAME passage (e.g. 2008 → 2022).
         This is the documented blind spot of `app/grounding.py`; the LLM judge has to cover it.
"""

import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app import prompts
from app.grounding import check_numbers, extract_numbers
from app.ingest import Chunk
from app.schemas import GroundednessCheck

_SENTENCE = re.compile(r"[^。\n]{15,150}。")  # one line, one sentence, ends with 。
_DIGIT = re.compile(r"\d")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Sample:
    chunk: Chunk
    correct: str  # "...199億円を措置[1]。"
    corrupted: str  # "...197億円を措置[1]。"


def _cite(sentence: str) -> str:
    """Put the citation before the final 。 so the checker treats it as one cited sentence."""
    return sentence[:-1] + "[1]。"


def _mutate(sentence: str, rng: random.Random) -> str | None:
    positions = [m.start() for m in _DIGIT.finditer(sentence)]
    if not positions:
        return None
    i = rng.choice(positions)
    new_digit = str((int(sentence[i]) + rng.randint(1, 9)) % 10)
    return sentence[:i] + new_digit + sentence[i + 1 :]


def _swap_with_existing(sentence: str, source_numbers: set[str], rng: random.Random) -> str | None:
    """Hard case: replace one number with a *different number that also occurs in the passage*."""
    matches = list(_NUMBER.finditer(sentence))
    if not matches:
        return None
    m = rng.choice(matches)
    (own,) = extract_numbers(m.group())
    # same length only (2008 → 2022, not 6 → 1007), so the error looks like a real LLM slip
    candidates = sorted(n for n in source_numbers if n.isdigit() and n != own and len(n) == len(own))
    if not candidates:
        return None
    return sentence[: m.start()] + rng.choice(candidates) + sentence[m.end() :]


def _is_valid_error(sentence: str, mutated: str | None, source_numbers: set[str], hard: bool) -> bool:
    """Keep only real errors. easy: the new number is absent from the source.
    hard: every number is still in the source, but the sentence's numbers have changed."""
    if not mutated:
        return False
    new = extract_numbers(mutated)
    if hard:
        return new <= source_numbers and new != extract_numbers(sentence)
    return bool(new - source_numbers)


def build_samples(
    chunks: list[Chunk], cases: list[dict], per_case: int = 4, seed: int = 42, hard: bool = False
) -> list[Sample]:
    rng = random.Random(seed)  # fixed seed → the same samples every run (reproducible)
    by_page: dict[tuple[str, int], list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_page[(c.doc_id, c.page)].append(c)

    samples: list[Sample] = []
    seen: set[str] = set()  # chunks overlap, so the same sentence can appear twice
    for case in cases:
        taken = 0
        for page in case["pages"]:
            for chunk in by_page[(case["doc_id"], page)]:
                source_numbers = extract_numbers(chunk.text)
                for s in _SENTENCE.findall(unicodedata.normalize("NFKC", chunk.text)):
                    s = s.strip()
                    if taken >= per_case or s in seen:
                        continue
                    mutated = _swap_with_existing(s, source_numbers, rng) if hard else _mutate(s, rng)
                    if _is_valid_error(s, mutated, source_numbers, hard):
                        samples.append(Sample(chunk, _cite(s), _cite(mutated)))
                        seen.add(s)
                        taken += 1
    return samples


def _rule_flags(sentence: str, chunk: Chunk) -> bool:
    return bool(check_numbers(sentence, [chunk], cited_ids=[1]))


def _llm_flags(checker, sentence: str, chunk: Chunk) -> bool:
    result: GroundednessCheck = checker.invoke(
        [
            SystemMessage(prompts.VERIFY_SYSTEM),
            HumanMessage(f"<passages>\n{prompts.format_passages([chunk])}\n</passages>\n\n回答:\n{sentence}"),
        ]
    )
    return not result.grounded


def confusion_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, float]:
    """`pairs` = (flags_correct, flags_corrupted) per sample. Positive class = corrupted."""
    n = len(pairs)
    tp = sum(bad for _, bad in pairs)
    fp = sum(ok for ok, _ in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": n - tp,
        "tn": n - fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm": fp / n if n else 0.0,
    }


def run(samples: list[Sample], llm=None, hard_samples: list[Sample] | None = None) -> dict[str, dict[str, float]]:
    """Return {verifier: confusion_metrics(...)}. Without `llm` only the rule check runs.
    Results for `hard_samples` are added with the suffix " (hard)"."""
    checker = llm.with_structured_output(GroundednessCheck, method="function_calling") if llm else None
    results: dict[str, dict[str, float]] = {}
    for suffix, group in (("", samples), (" (hard)", hard_samples or [])):
        flags: dict[str, list[tuple[bool, bool]]] = defaultdict(list)  # (flags_correct, flags_corrupted)
        for s in group:
            rule = (_rule_flags(s.correct, s.chunk), _rule_flags(s.corrupted, s.chunk))
            flags["Rule-based number check"].append(rule)
            if checker:
                judge = (_llm_flags(checker, s.correct, s.chunk), _llm_flags(checker, s.corrupted, s.chunk))
                flags["LLM judge"].append(judge)
                flags["LLM judge + rule check (deployed)"].append((judge[0] or rule[0], judge[1] or rule[1]))
        results.update({f"{name}{suffix}": confusion_metrics(f) for name, f in flags.items()})
    return results
