"""Phase 2: combined detector that runs every layer and merges spans by type priority."""

from __future__ import annotations

from callsearch.redact.base import Detector, Span
from callsearch.redact.merge import resolve_overlaps


class PipelineDetector:
    def __init__(self, detectors: list[Detector] | None = None, name: str = "pipeline"):
        if detectors is None:
            from callsearch.redact.ner import SpacyNerDetector
            from callsearch.redact.regex import RegexDetector

            detectors = [RegexDetector(), SpacyNerDetector()]
        self.name = name
        self.detectors = detectors

    def detect(self, text: str) -> list[Span]:
        return resolve_overlaps([s for d in self.detectors for s in d.detect(text)])
