"""Tests for the spaCy NER layer and the regex+ner pipeline."""

import importlib.util
from pathlib import Path

import pytest
import spacy

from callsearch.data.generator import generate_calls
from callsearch.redact.base import Span, get_detector
from callsearch.redact.ner import SpacyNerDetector, build_examples, make_reference_doc
from callsearch.redact.pipeline import PipelineDetector
from callsearch.redact.regex import RegexDetector

ROOT = Path(__file__).resolve().parents[1]


def label(text, sub, typ="PERSON", start=None):
    i = text.index(sub) if start is None else start
    return {"turn_id": 0, "start": i, "end": i + len(sub), "type": typ, "value": sub}


def test_label_to_span_conversion():
    nlp = spacy.blank("en")
    text = "It's under the name Danielle Johnson. Thanks, Danielle!"
    labels = [label(text, "Danielle Johnson"), label(text, "Danielle", start=46),
              label(text, "604-555-0182", typ="PHONE", start=0)]  # non-PERSON labels ignored
    doc, failed = make_reference_doc(nlp, text, labels)
    assert failed == 0
    assert [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents] == [
        ("Danielle Johnson", 20, 36, "PERSON"), ("Danielle", 46, 54, "PERSON")]


def test_label_alignment_failure_counted():
    nlp = spacy.blank("en")
    text = "Hi Danielle Johnson"
    # "Dan" sits inside one token, so contract mode leaves no whole token -> failure.
    doc, failed = make_reference_doc(nlp, text, [label(text, "Dan"), label(text, "Johnson")])
    assert failed == 1 and [e.text for e in doc.ents] == ["Johnson"]


def test_build_examples_counts():
    nlp = spacy.blank("en")
    calls = generate_calls(n=10, seed=1, multiwoz_frac=0, test_frac=0, dev_frac=0)
    examples, n_labels, failed = build_examples(nlp, calls)
    assert len(examples) == sum(len(c["turns"]) for c in calls)
    assert n_labels == sum(l["type"] == "PERSON" for c in calls for l in c["labels"])
    assert failed == 0


def test_missing_model_error(tmp_path):
    det = SpacyNerDetector(model_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="scripts/train_ner.py"):
        det.detect("Hello Danielle")


def test_registry_has_ner_and_pipeline():
    assert get_detector("ner").name == "ner"  # lazy: constructing doesn't load a model
    assert get_detector("pipeline").name == "pipeline"


class StubNer:
    name = "ner"

    def __init__(self, *names):
        self.names = names

    def detect(self, text):
        out = []
        for n in self.names:
            i = text.index(n)
            out.append(Span(i, i + len(n), "PERSON", "ner"))
        return out


def test_pipeline_name_and_card_both_detected():
    text = "This is Danielle Johnson, card 4539 1488 0343 6467."
    spans = PipelineDetector([RegexDetector(), StubNer("Danielle Johnson")]).detect(text)
    assert [(text[s.start:s.end], s.type, s.source) for s in spans] == [
        ("Danielle Johnson", "PERSON", "ner"), ("4539 1488 0343 6467", "CARD", "regex")]
    assert not any(a.overlaps(b) for a in spans for b in spans if a is not b)


def test_pipeline_person_loses_overlap_to_card():
    text = "card 4539 1488 0343 6467 ok"
    spans = PipelineDetector([RegexDetector(), StubNer("1488 0343")]).detect(text)
    assert [s.type for s in spans] == ["CARD"]


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_ner", ROOT / "scripts" / "train_ner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_tiny_training_run(tmp_path):
    train_ner = _load_train_script()
    calls = generate_calls(n=5, seed=3, multiwoz_frac=0, test_frac=0, dev_frac=0.2)  # 4 train + 1 dev call
    train_calls = [c for c in calls if c["split"] == "train"]
    dev_calls = [c for c in calls if c["split"] == "dev"]
    summary = train_ner.train(train_calls, dev_calls, base="blank", epochs=2, out=tmp_path / "m", verbose=False)
    assert summary["epochs_run"] == 2 and summary["n_train_calls"] == 4 and summary["n_dev_calls"] == 1
    assert 15 <= summary["n_train_examples"] + summary["n_dev_examples"] <= 60
    spans = SpacyNerDetector(model_dir=tmp_path / "m").detect("Hi, this is Danielle Johnson.")
    assert all(s.type == "PERSON" and s.source == "ner" for s in spans)


def test_train_rejects_wrong_splits(tmp_path):
    train_ner = _load_train_script()
    calls = generate_calls(n=6, seed=3, multiwoz_frac=0, test_frac=0.34, dev_frac=0.34)
    by = {s: [c for c in calls if c["split"] == s] for s in ("train", "dev", "test")}
    with pytest.raises(ValueError, match="train split"):
        train_ner.train(by["train"] + by["test"], by["dev"], base="blank", epochs=1, out=tmp_path / "m", verbose=False)
    with pytest.raises(ValueError, match="dev split"):
        train_ner.train(by["train"], by["test"], base="blank", epochs=1, out=tmp_path / "m", verbose=False)


def _run_show_errors(*args):
    import subprocess
    import sys
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "show_errors.py"), *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_show_errors_refuses_test_without_flag():
    r = _run_show_errors("--detector", "regex", "--split", "test")
    assert r.returncode != 0 and "--allow-test" in r.stderr
