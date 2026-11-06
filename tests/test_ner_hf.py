"""Tests for the Hugging Face (DistilBERT) NER layer. Offline: uses a tiny random model and vocab."""

import importlib.util
import string
from pathlib import Path

import pytest

from callsearch.data.generator import generate_calls
from callsearch.redact.base import get_detector
from callsearch.redact.ner_hf import B, I, IGNORE, O, HFNerDetector, align_labels, decode

ROOT = Path(__file__).resolve().parents[1]


def test_align_labels_subwords_and_specials():
    text = "Hi Danielle Johnson"
    # [CLS] Hi Dan ##ielle John ##son [SEP]
    offsets = [(0, 0), (0, 2), (3, 6), (6, 11), (12, 16), (16, 19), (0, 0)]
    tags, missed = align_labels(offsets, [(3, 19)])
    assert tags == [IGNORE, O, B, I, I, I, IGNORE] and missed == 0
    assert text[3:19] == "Danielle Johnson"


def test_align_labels_two_spans_and_missed():
    offsets = [(0, 0), (0, 4), (5, 8), (9, 13), (0, 0)]
    tags, missed = align_labels(offsets, [(0, 4), (9, 13), (20, 25)])  # last span is past the tokens
    assert tags == [IGNORE, B, O, B, IGNORE] and missed == 1


def test_decode_inverts_alignment():
    text = "Hi Danielle Johnson and Mark"
    offsets = [(0, 0), (0, 2), (3, 6), (6, 11), (12, 16), (16, 19), (20, 23), (24, 28), (0, 0)]
    spans = [(3, 19), (24, 28)]
    tags, _ = align_labels(offsets, spans)
    assert decode(offsets, tags, text) == spans


def test_decode_i_without_b_and_partial_word():
    text = "call Danielle now"
    offsets = [(0, 0), (0, 4), (5, 8), (8, 13), (14, 17), (0, 0)]
    # model tagged only the second subword, and with I instead of B
    assert decode(offsets, [IGNORE, O, O, I, O, IGNORE], text) == [(5, 13)]


def test_decode_adjacent_b_tags_stay_separate():
    text = "Ann Lee"
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
    assert decode(offsets, [IGNORE, B, B, IGNORE], text) == [(0, 3), (4, 7)]


def test_missing_model_error(tmp_path):
    det = HFNerDetector(model_dir=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="scripts/train_ner_hf.py"):
        det.detect("Hello Danielle")


def test_registry_has_hf_detectors():
    assert get_detector("ner_hf").name == "ner_hf"  # lazy: constructing doesn't load a model
    assert get_detector("pipeline_hf").name == "pipeline_hf"


# --- tiny offline model --------------------------------------------------------------------

def _tiny_base(tmp_path: Path) -> Path:
    """A random 1-layer DistilBERT with a character-level WordPiece vocab, saved like a Hub model."""
    pytest.importorskip("torch")
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertModel

    chars = string.ascii_letters + string.digits + string.punctuation
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *chars, *("##" + c for c in chars)]
    d = tmp_path / "tiny_base"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=False,
                            model_input_names=["input_ids", "attention_mask"])
    tok.save_pretrained(d)
    cfg = DistilBertConfig(vocab_size=len(vocab), dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                           max_position_embeddings=512)
    DistilBertModel(cfg).save_pretrained(d)  # body only, like the Hub base model
    return d


def test_gold_roundtrip_through_real_tokenizer(tmp_path):
    """Every PERSON label survives tokenize -> BIO -> decode with exact offsets."""
    _tiny = _tiny_base(tmp_path)
    from transformers import AutoTokenizer

    train_ner_hf = _load_train_script()
    tok = AutoTokenizer.from_pretrained(_tiny)
    calls = generate_calls(n=8, seed=5, multiwoz_frac=0.5, test_frac=0, dev_frac=0)
    for text, spans in train_ner_hf.turns_with_spans(calls):
        enc = tok(text, return_offsets_mapping=True)  # no truncation: compare whole turns
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        tags, missed = align_labels(offsets, spans)
        assert missed == 0
        assert decode(offsets, tags, text) == sorted(spans)


def _load_train_script():
    spec = importlib.util.spec_from_file_location("train_ner_hf", ROOT / "scripts" / "train_ner_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exact_prf():
    train_ner_hf = _load_train_script()
    p, r, f = train_ner_hf.exact_prf([[(0, 3), (5, 8)], []], [[(0, 3)], [(1, 2)]])
    assert (p, r) == (0.5, 0.5) and f == 0.5


def test_train_rejects_wrong_splits(tmp_path):
    train_ner_hf = _load_train_script()
    calls = generate_calls(n=6, seed=3, multiwoz_frac=0, test_frac=0.34, dev_frac=0.34)
    by = {s: [c for c in calls if c["split"] == s] for s in ("train", "dev", "test")}
    with pytest.raises(ValueError, match="train split"):
        train_ner_hf.train(by["train"] + by["test"], by["dev"], out=tmp_path / "m", verbose=False)
    with pytest.raises(ValueError, match="dev split"):
        train_ner_hf.train(by["train"], by["test"], out=tmp_path / "m", verbose=False)


@pytest.mark.slow
def test_tiny_training_run(tmp_path):
    base = _tiny_base(tmp_path)
    train_ner_hf = _load_train_script()
    calls = generate_calls(n=5, seed=3, multiwoz_frac=0, test_frac=0, dev_frac=0.2)  # 4 train + 1 dev call
    train_calls = [c for c in calls if c["split"] == "train"]
    dev_calls = [c for c in calls if c["split"] == "dev"]
    summary = train_ner_hf.train(train_calls, dev_calls, base_model=str(base), epochs=2, patience=5,
                                 out=tmp_path / "m", device="cpu", verbose=False)
    assert summary["epochs_run"] == 2 and summary["n_train_calls"] == 4 and summary["n_dev_calls"] == 1
    assert summary["failed_alignments"] == 0
    det = HFNerDetector(model_dir=tmp_path / "m", device="cpu")
    texts = ["Hi, this is Danielle Johnson.", "x" * 2000]  # second one needs several windows
    for text, spans in zip(texts, det.detect_batch(texts)):
        assert all(s.type == "PERSON" and s.source == "ner_hf" and 0 <= s.start < s.end <= len(text)
                   for s in spans)
