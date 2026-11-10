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

    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        """Same as detect() per text; layers with detect_batch (the HF model) run batched."""
        per_layer = [d.detect_batch(texts) if hasattr(d, "detect_batch") else [d.detect(t) for t in texts]
                     for d in self.detectors]
        return [resolve_overlaps([s for layer in per_layer for s in layer[i]]) for i in range(len(texts))]


class PipelineHFDetector(PipelineDetector):
    """regex + the DistilBERT NER (ner_hf) instead of spaCy, for the NER comparison."""

    def __init__(self, detectors: list[Detector] | None = None, name: str = "pipeline_hf"):
        if detectors is None:
            from callsearch.redact.ner_hf import HFNerDetector
            from callsearch.redact.regex import RegexDetector

            detectors = [RegexDetector(), HFNerDetector()]
        super().__init__(detectors, name)
