"""Tests for the Presidio baseline detector and compare_results."""

import importlib.util
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from callsearch.redact.base import Span, get_detector
from callsearch.redact.presidio_baseline import ENTITY_MAP, PresidioCaSinDetector, PresidioDetector

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class R:  # stands in for presidio_analyzer.RecognizerResult
    entity_type: str
    start: int
    end: int
    score: float


class StubAnalyzer:
    def __init__(self, results):
        self.results = results

    def analyze(self, text, language):
        assert language == "en"
        return self.results


def detect(results):
    return PresidioDetector(analyzer=StubAnalyzer(results)).detect("x" * 100)


def test_entity_map():
    assert ENTITY_MAP == {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE",
                          "CREDIT_CARD": "CARD", "CA_SIN": "SIN", "US_BANK_NUMBER": "ACCOUNT"}
    assert "US_SSN" not in ENTITY_MAP  # a US SSN is not a Canadian SIN


def test_mapped_types_become_spans():
    spans = detect([R("PERSON", 0, 5, 0.85), R("EMAIL_ADDRESS", 10, 20, 1.0), R("PHONE_NUMBER", 30, 40, 0.75),
                    R("CREDIT_CARD", 50, 66, 1.0), R("CA_SIN", 70, 79, 0.5), R("US_BANK_NUMBER", 85, 95, 0.4)])
    assert [(s.start, s.end, s.type, s.source) for s in spans] == [
        (0, 5, "PERSON", "presidio"), (10, 20, "EMAIL", "presidio"), (30, 40, "PHONE", "presidio"),
        (50, 66, "CARD", "presidio"), (70, 79, "SIN", "presidio"), (85, 95, "ACCOUNT", "presidio")]


def test_unmapped_types_produce_no_spans():
    assert detect([R("LOCATION", 0, 5, 0.85), R("DATE_TIME", 6, 10, 0.85), R("US_SSN", 11, 20, 0.5),
                   R("URL", 21, 30, 0.5), R("NRP", 31, 35, 0.85)]) == []


def test_overlaps_resolved_by_presidio_score():
    spans = detect([R("US_BANK_NUMBER", 0, 16, 0.05), R("CREDIT_CARD", 0, 16, 1.0)])
    assert [s.type for s in spans] == ["CARD"]


def test_interface_and_registry_lazy():
    det = get_detector("presidio")  # must not load spaCy/Presidio yet
    assert det.name == "presidio" and det._analyzer is None
    assert callable(det.detect)


def test_casin_variant_registry_lazy():
    det = get_detector("presidio_casin")
    assert det.name == "presidio_casin" and det._analyzer is None
    spans = PresidioCaSinDetector(analyzer=StubAnalyzer([R("CA_SIN", 3, 14, 0.5)])).detect("x" * 20)
    assert [(s.type, s.source) for s in spans] == [("SIN", "presidio_casin")]


@pytest.mark.slow
def test_casin_variant_is_default_plus_one_recognizer():
    default = PresidioDetector().analyzer.registry.recognizers
    casin = PresidioCaSinDetector().analyzer.registry.recognizers
    assert Counter(r.name for r in casin) - Counter(r.name for r in default) == Counter({"CaSinRecognizer": 1})
    assert not Counter(r.name for r in default) - Counter(r.name for r in casin)


@pytest.mark.slow
def test_real_presidio_default_engine():
    det = PresidioDetector()
    text = "Hi, I'm Danielle Johnson, email danielle.johnson@gmail.com, card 4539 1488 0343 6467."
    spans = det.detect(text)
    assert all(isinstance(s, Span) for s in spans)
    found = {(s.type, text[s.start:s.end]) for s in spans}
    assert ("EMAIL", "danielle.johnson@gmail.com") in found
    assert ("CARD", "4539 1488 0343 6467") in found
    assert det.model_name().startswith("en_core_web_")


# --- compare_results ------------------------------------------------------------

def _load_compare():
    spec = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts" / "compare_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_eval(detector):
    m = {"precision": 1.0, "recall": 0.5, "f1": 0.667, "exact_recall": 0.5, "leak_rate": 0.5,
         "tp": 1, "fp": 0, "fn": 1, "support": 2, "leaked": 1}
    return {"detector": detector, "split": "dev", "n_calls": 1,
            "per_type": {t: dict(m) for t in ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]},
            "micro": dict(m), "micro_excl_person": dict(m), "distractor_fp": {}}


def test_compare_handles_missing_detectors(tmp_path):
    cmp = _load_compare()
    (tmp_path / "eval_regex_dev.json").write_text(json.dumps(_fake_eval("regex")), encoding="utf-8")
    md = cmp.build_comparison("dev", tmp_path)
    assert "| regex | CARD | 1.000 | 0.500 |" in md
    assert "Missing results (not run yet): ner, pipeline, presidio, presidio_casin" in md
    assert "n/a: needs both pipeline and presidio results." in md


def test_compare_summary(tmp_path):
    cmp = _load_compare()
    for d in ("pipeline", "presidio"):
        (tmp_path / f"eval_{d}_dev.json").write_text(json.dumps(_fake_eval(d)), encoding="utf-8")
    md = cmp.build_comparison("dev", tmp_path)
    assert "| SIN | 0.500 | 0.500 | 50.0% | 50.0% |" in md
    assert "| overall | 0.500 | 0.500 | 50.0% | 50.0% |" in md
