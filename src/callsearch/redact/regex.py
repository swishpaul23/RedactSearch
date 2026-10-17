"""Regex-based detection of structured PII: cards, SINs, accounts, emails, phones (no PERSON)."""

from __future__ import annotations

import re

from callsearch.redact.base import Span
from callsearch.redact.merge import resolve_overlaps
from callsearch.validators import luhn_ok, sin_ok

# Maximal run of digit groups joined by a single space or dash, never inside a longer run.
DIGIT_RUN = re.compile(r"(?<!\d)\d+(?:[ -]\d+)*(?!\d)")
GROUP = re.compile(r"\d+")

EMAIL = re.compile(r"(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")

# NANP: area code and exchange start with 2-9. Optional +1 / 1 country prefix.
PHONE = re.compile(
    r"(?<![\d+])(?:\+?1[ .-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[ .-]?[2-9]\d{2}[ .-]?\d{4}(?!\d)"
)

# Context word (account / acct, optionally "number", "no.", "num", "#") then 8-12 digits.
ACCOUNT = re.compile(
    r"\b(?:account|acct)\b\.?(?:\s*(?:number|num\b\.?|no\b\.?|#))?\s*(?:is\b|was\b|[:#=-])?\s*#?\s*"
    r"(?<!\d)(\d{8,12})(?!\d)",
    re.IGNORECASE,
)


def _windows(run: str, offset: int):
    """Yield (start, end, digits, group_sizes) for every window of whole consecutive groups."""
    groups = [(m.start() + offset, m.end() + offset, m.group()) for m in GROUP.finditer(run)]
    for i in range(len(groups)):
        digits = ""
        for j in range(i, len(groups)):
            digits += groups[j][2]
            yield groups[i][0], groups[j][1], digits, [len(g[2]) for g in groups[i:j + 1]]


class RegexDetector:
    name = "regex"

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for run in DIGIT_RUN.finditer(text):
            for start, end, digits, sizes in _windows(run.group(), run.start()):
                if 13 <= len(digits) <= 19 and luhn_ok(digits):
                    spans.append(Span(start, end, "CARD", self.name))
                elif len(digits) == 9 and sizes in ([9], [3, 3, 3]) and sin_ok(digits):
                    spans.append(Span(start, end, "SIN", self.name))
        for m in ACCOUNT.finditer(text):
            spans.append(Span(m.start(1), m.end(1), "ACCOUNT", self.name))
        for m in EMAIL.finditer(text):
            spans.append(Span(m.start(), m.end(), "EMAIL", self.name))
        for m in PHONE.finditer(text):
            spans.append(Span(m.start(), m.end(), "PHONE", self.name))
        return resolve_overlaps(spans)
