"""Shared redaction interface: Span, the Detector protocol, and the detector registry."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    source: str

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


class Detector(Protocol):
    name: str

    def detect(self, text: str) -> list[Span]: ...


# name -> "module:Class". Imported lazily so heavy Phase 2 deps load only when used.
# Phase 2 adds: "llm".
_REGISTRY: dict[str, str] = {
    "regex": "callsearch.redact.regex:RegexDetector",
    "ner": "callsearch.redact.ner:SpacyNerDetector",
    "pipeline": "callsearch.redact.pipeline:PipelineDetector",
    "presidio": "callsearch.redact.presidio_baseline:PresidioDetector",
    "presidio_casin": "callsearch.redact.presidio_baseline:PresidioCaSinDetector",
}


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def get_detector(name: str, **kwargs) -> Detector:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; available: {available_detectors()}")
    module, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)(**kwargs)
