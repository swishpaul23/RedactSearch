"""Phase 2: spaCy NER detection of PERSON names (structured PII is left to the regex layer)."""

from __future__ import annotations

from pathlib import Path

from callsearch.redact.base import Span

DEFAULT_MODEL_DIR = Path("models/ner_spacy")
LABEL = "PERSON"


def make_reference_doc(nlp, text: str, labels: list[dict]):
    """Gold Doc for one turn: PERSON labels as entity spans. Returns (doc, n_failed_alignments).

    Character offsets are snapped to whole tokens with alignment_mode="contract"; a label
    that contains no whole token can't be represented and is counted as a failure.
    """
    doc = nlp.make_doc(text)
    spans, failed = [], 0
    for label in labels:
        if label["type"] != LABEL:
            continue
        span = doc.char_span(label["start"], label["end"], label=LABEL, alignment_mode="contract")
        if span is None:
            failed += 1
        else:
            spans.append(span)
    doc.ents = spans
    return doc, failed


def build_examples(nlp, calls: list[dict]):
    """One spacy Example per turn. Returns (examples, n_person_labels, n_failed_alignments)."""
    from spacy.training import Example

    examples, n_labels, n_failed = [], 0, 0
    for call in calls:
        for turn in call["turns"]:
            labels = [l for l in call["labels"] if l["turn_id"] == turn["turn_id"] and l["type"] == LABEL]
            ref, failed = make_reference_doc(nlp, turn["text"], labels)
            examples.append(Example(nlp.make_doc(turn["text"]), ref))
            n_labels += len(labels)
            n_failed += failed
    return examples, n_labels, n_failed


class SpacyNerDetector:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, name: str = "ner"):
        self.name = name
        self.model_dir = Path(model_dir)
        self._nlp = None

    def _load(self):
        if self._nlp is None:
            if not (self.model_dir / "meta.json").exists():
                raise FileNotFoundError(
                    f"No trained NER model at {self.model_dir}. Train one first: python scripts/train_ner.py"
                )
            import spacy

            self._nlp = spacy.load(self.model_dir)
        return self._nlp

    def detect(self, text: str) -> list[Span]:
        doc = self._load()(text)
        return [Span(e.start_char, e.end_char, LABEL, self.name) for e in doc.ents if e.label_ == LABEL]
