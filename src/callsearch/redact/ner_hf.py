"""Phase 2, NER comparison: a fine-tuned Hugging Face token classifier (DistilBERT) for PERSON.

Same job as the spaCy layer (ner.py), different model family, so the two can be compared
on the same splits. Train it with scripts/train_ner_hf.py.

Labels are BIO over subword tokens: O, B-PERSON, I-PERSON. Gold spans are character
offsets, so labels are aligned through the fast tokenizer's offset mapping, and predictions
are mapped back to character offsets the same way. No word splitting of our own.
"""

from __future__ import annotations

from pathlib import Path

from callsearch.redact.base import Span

DEFAULT_MODEL_DIR = Path("models/ner_distilbert")
DEFAULT_BASE_MODEL = "distilbert/distilbert-base-cased"  # cased: capitalisation is a strong name cue
LABEL = "PERSON"
TAGS = ["O", f"B-{LABEL}", f"I-{LABEL}"]
O, B, I = 0, 1, 2
IGNORE = -100  # special and padding tokens: no loss, no prediction
MAX_LENGTH = 256
STRIDE = 32


def align_labels(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Gold BIO tag ids for one tokenized text. Returns (tag_ids, n_spans_without_a_token).

    A token is inside a span when their characters overlap. The first token of each span
    gets B, the rest I. Every subword is labeled, so decoding needs no word table.
    Special tokens have offset (0, 0) and get IGNORE.
    """
    tags = [IGNORE if s == e else O for s, e in offsets]
    missed = 0
    for ss, se in spans:
        first = True
        for i, (s, e) in enumerate(offsets):
            if s == e or e <= ss or s >= se:
                continue
            tags[i] = B if first else I
            first = False
        missed += first
    return tags, missed


def _expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow a span to whole words, so "Dan" predicted inside "Danielle" covers "Danielle"."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    return start, end


def decode(offsets: list[tuple[int, int]], tag_ids: list[int], text: str) -> list[tuple[int, int]]:
    """Predicted BIO tag ids -> character spans.

    B starts a span. I continues the open span, or starts one if none is open (models
    sometimes skip the B). Spans are then grown to word edges and merged if they touch.
    """
    raw, cur = [], None
    for (s, e), t in zip(offsets, tag_ids):
        if s == e or t == IGNORE:
            continue
        if t == B or (t == I and cur is None):
            if cur:
                raw.append(cur)
            cur = [s, e]
        elif t == I:
            cur[1] = e
        else:
            if cur:
                raw.append(cur)
            cur = None
    if cur:
        raw.append(cur)

    return _merge([_expand_to_word(text, s, e) for s, e in raw])


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def predict_spans(model, tokenizer, texts: list[str], batch_size: int = 32,
                  max_length: int = MAX_LENGTH, stride: int = STRIDE) -> list[list[tuple[int, int]]]:
    """Character spans per text. Long texts are split into overlapping windows."""
    import torch

    device = next(model.parameters()).device
    results: list[list[tuple[int, int]]] = [[] for _ in texts]
    model.eval()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, truncation=True, max_length=max_length, stride=stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True,
                        padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        owners = enc.pop("overflow_to_sample_mapping").tolist()
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in enc.items() if k in tokenizer.model_input_names}
            logits = model(**inputs).logits
        preds = logits.argmax(-1).tolist()
        for w, owner in enumerate(owners):
            text = chunk[owner]
            results[i + owner].extend(decode([tuple(o) for o in offsets[w]], preds[w], text))
    return [_merge(r) for r in results]


class HFNerDetector:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, name: str = "ner_hf",
                 device: str | None = None, batch_size: int = 32):
        self.name = name
        self.model_dir = Path(model_dir)
        self.device = device
        self.batch_size = batch_size
        self._model = self._tok = None

    def _load(self):
        if self._model is None:
            if not (self.model_dir / "config.json").exists():
                raise FileNotFoundError(
                    f"No trained HF NER model at {self.model_dir}. Train one first: python scripts/train_ner_hf.py"
                )
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(device)
        return self._model, self._tok

    def detect_batch(self, texts: list[str]) -> list[list[Span]]:
        model, tok = self._load()
        return [[Span(s, e, LABEL, self.name) for s, e in spans]
                for spans in predict_spans(model, tok, texts, self.batch_size)]

    def detect(self, text: str) -> list[Span]:
        return self.detect_batch([text])[0]
