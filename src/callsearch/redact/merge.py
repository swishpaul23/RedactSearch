"""Resolve overlapping PII spans by type priority and replace spans with typed placeholders."""

from __future__ import annotations

from callsearch.redact.base import Span

# Higher-priority types win overlaps. Types not listed (e.g. PERSON) rank below all of these.
PRIORITY = ["CARD", "SIN", "ACCOUNT", "EMAIL", "PHONE"]


def _rank(span: Span) -> int:
    return PRIORITY.index(span.type) if span.type in PRIORITY else len(PRIORITY)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Drop any span overlapping a higher-priority one; ties go to the longer span."""
    kept: list[Span] = []
    for s in sorted(spans, key=lambda s: (_rank(s), -(s.end - s.start), s.start)):
        if not any(s.overlaps(k) for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def redact(text: str, spans: list[Span]) -> str:
    """Replace each span with [TYPE], right-to-left so earlier offsets stay valid.

    Any spans that still overlap are unioned under the highest-priority type rather
    than dropped, so no characters of a detected span survive (recall over precision).
    """
    merged: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if merged and s.start < merged[-1].end:
            prev = merged[-1]
            winner = prev if _rank(prev) <= _rank(s) else s
            merged[-1] = Span(prev.start, max(prev.end, s.end), winner.type, winner.source)
        else:
            merged.append(s)
    for s in reversed(merged):
        text = text[:s.start] + f"[{s.type}]" + text[s.end:]
    return text
