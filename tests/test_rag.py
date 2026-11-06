"""Tests for the redaction boundary, chunking, the Chroma index and the index guard.

Offline: a hashing bag-of-words embedder stands in for sentence-transformers.
"""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest

from callsearch.data.generator import generate_calls
from callsearch.rag.chunk import chunk_calls
from callsearch.rag.guard import pattern_scan, planted_scan, verdict
from callsearch.redact.base import Span
from callsearch.redact.export import ManifestError, load_redacted, redact_calls, write_redacted
from callsearch.redact.pipeline import PipelineDetector
from callsearch.redact.regex import RegexDetector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


class FakeEmbedder:
    name = "fake"
    dim = 128

    def _vec(self, text):
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class LabelDetector:
    """Finds exactly the labeled spans: a perfect redactor for tests."""
    name = "labels"

    def __init__(self, calls):
        self.by_text = {}
        for c in calls:
            for t in c["turns"]:
                self.by_text[t["text"]] = [Span(l["start"], l["end"], l["type"], "labels")
                                           for l in c["labels"] if l["turn_id"] == t["turn_id"]]

    def detect(self, text):
        return self.by_text.get(text, [])


class NoDetector:
    name = "none"

    def detect(self, text):
        return []


def call(call_id="call_9001", split="test", turns=("Hi there", "I want a refund for my subscription")):
    return {"call_id": call_id, "split": split, "source": "template",
            "turns": [{"turn_id": i, "speaker": "agent" if i % 2 == 0 else "customer", "start_sec": 2.0 * i,
                       "text": t} for i, t in enumerate(turns)]}


# --- chunking ------------------------------------------------------------------------------

def test_chunk_ids_metadata_and_context():
    chunks = chunk_calls([call(turns=("Hello", "  ", "I need a refund"))], context_turns=1)
    assert [c.id for c in chunks] == ["call_9001:0", "call_9001:2"]  # blank turn skipped
    assert chunks[1].text == "I need a refund"
    assert chunks[1].embed_text == "customer:   \nagent: I need a refund"
    assert chunks[1].metadata() == {"call_id": "call_9001", "turn_id": 2, "speaker": "agent", "start_sec": 4.0,
                                    "split": "test", "source": "template"}


def test_chunk_without_context():
    chunks = chunk_calls([call()], context_turns=0)
    assert all(c.embed_text == f"{c.speaker}: {c.text}" for c in chunks)
    with pytest.raises(ValueError):
        chunk_calls([call()], context_turns=-1)


# --- redaction boundary --------------------------------------------------------------------

def test_export_drops_labels_and_masks_values(tmp_path):
    raw = generate_calls(n=6, seed=2, multiwoz_frac=0, test_frac=0, dev_frac=0)
    red = redact_calls(raw, LabelDetector(raw))
    assert all(set(c) == {"call_id", "split", "source", "turns"} for c in red)
    text = json.dumps(red)
    assert all(l["value"] not in text for c in raw for l in c["labels"] if len(l["value"]) > 3)
    manifest = write_redacted(red, tmp_path, "labels")
    calls, m = load_redacted(tmp_path)
    assert calls == red and m == manifest and m["redacted"] is True


def test_load_refuses_missing_tampered_or_unmarked(tmp_path):
    with pytest.raises(ManifestError, match="no manifest.json"):
        load_redacted(tmp_path)
    write_redacted([call()], tmp_path, "labels")
    path = tmp_path / "calls.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("refund", "REFUND"), encoding="utf-8")
    with pytest.raises(ManifestError, match="changed"):
        load_redacted(tmp_path)
    write_redacted([call()], tmp_path, "labels")
    m = json.loads((tmp_path / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps({**m, "redacted": False}))
    with pytest.raises(ManifestError, match="does not mark"):
        load_redacted(tmp_path)


def test_load_refuses_extra_fields(tmp_path):
    write_redacted([{**call(), "labels": []}], tmp_path, "labels")
    with pytest.raises(ManifestError, match="unexpected fields"):
        load_redacted(tmp_path)


def test_pipeline_detect_batch_matches_detect():
    texts = ["card 4539 1488 0343 6467 ok", "mail me at a.b@example.com", "nothing here"]
    det = PipelineDetector([RegexDetector()])
    assert det.detect_batch(texts) == [det.detect(t) for t in texts]


# --- index ---------------------------------------------------------------------------------

@pytest.fixture
def built(tmp_path):
    from build_index import build

    calls = [call("call_9001", "test", ("Thanks for calling", "I want a refund for my subscription")),
             call("call_9002", "dev", ("Hello", "Can I book a table for two tonight")),
             call("call_9003", "train", ("Hi", "My parcel never arrived at the house"))]
    write_redacted(calls, tmp_path / "red", "labels")
    summary = build(tmp_path / "red", tmp_path / "idx", FakeEmbedder(), context_turns=0)
    return tmp_path, summary


def test_build_and_search(built):
    from callsearch.rag.store import CallIndex

    tmp_path, summary = built
    assert summary["n_chunks"] == 6 and summary["detector"] == "labels"
    index = CallIndex(tmp_path / "idx", embedder=FakeEmbedder())
    assert index.count() == 6
    assert index.info["embed_model"] == "fake" and index.info["context_turns"] == 0
    top = index.search("refund subscription", k=1)[0]
    assert (top.chunk_id, top.call_id, top.turn_id, top.speaker) == ("call_9001:1", "call_9001", 1, "customer")
    assert top.text == "I want a refund for my subscription" and 0 < top.score <= 1.0001
    assert all(h.split == "dev" for h in index.search("refund", k=3, split="dev"))


def test_rebuild_replaces_collection(built):
    from build_index import build
    from callsearch.rag.store import CallIndex

    tmp_path, _ = built
    write_redacted([call("call_9100", "test")], tmp_path / "red", "labels")
    build(tmp_path / "red", tmp_path / "idx", FakeEmbedder())
    assert CallIndex(tmp_path / "idx", embedder=FakeEmbedder()).count() == 2


def test_query_embedder_must_match_index(built):
    from callsearch.rag.store import CallIndex

    tmp_path, _ = built
    other = FakeEmbedder()
    other.name = "other"
    with pytest.raises(ValueError, match="built with 'fake'"):
        CallIndex(tmp_path / "idx", embedder=other).search("refund")


def test_missing_index_error(tmp_path):
    from callsearch.rag.store import CallIndex

    with pytest.raises(FileNotFoundError, match="build_index.py"):
        CallIndex(tmp_path / "nope").count()


def test_build_refuses_structured_pii(tmp_path):
    from build_index import build

    raw = generate_calls(n=6, seed=2, multiwoz_frac=0, test_frac=0, dev_frac=0)
    write_redacted(redact_calls(raw, NoDetector()), tmp_path / "red", "none")  # nothing masked
    with pytest.raises(RuntimeError, match="nothing was written"):
        build(tmp_path / "red", tmp_path / "idx", FakeEmbedder())
    assert not (tmp_path / "idx").exists()


def test_build_refuses_without_manifest(tmp_path):
    from build_index import build

    (tmp_path / "red").mkdir()
    (tmp_path / "red" / "calls.jsonl").write_text(json.dumps(call()) + "\n")
    with pytest.raises(ManifestError):
        build(tmp_path / "red", tmp_path / "idx", FakeEmbedder())


# --- guard ---------------------------------------------------------------------------------

def test_pattern_scan_counts_structured_types():
    hits = pattern_scan(["card 4539 1488 0343 6467", "[CARD] and [EMAIL]", "write to a.b@example.com"])
    assert hits["CARD"] == 1 and hits["EMAIL"] == 1 and "PERSON" not in hits


def test_planted_scan_and_verdict():
    raw = generate_calls(n=8, seed=4, multiwoz_frac=0, test_frac=0.5, dev_frac=0)
    clean = {c.id: c.text for c in chunk_calls(redact_calls(raw, LabelDetector(raw)))}
    res = planted_scan(clean, raw)
    assert sum(res["leaks"].values()) == 0 and res["labels_without_chunk"] == 0
    assert res["support"]["PERSON"] == sum(l["type"] == "PERSON" for c in raw for l in c["labels"])
    assert verdict({"CARD": 0}, res, allow_person_leaks=False) == []

    # leak one name and one card back in
    person = next(l for c in raw for l in c["labels"] if l["type"] == "PERSON")
    card = next((c, l) for c in raw for l in c["labels"] if l["type"] == "CARD")
    pc = next(c for c in raw if person in c["labels"])
    clean[f"{pc['call_id']}:{person['turn_id']}"] += " " + person["value"]
    clean[f"{card[0]['call_id']}:{card[1]['turn_id']}"] += " " + card[1]["value"]
    res = planted_scan(clean, raw)
    assert res["leaks"]["PERSON"] >= 1 and res["leaks"]["CARD"] >= 1
    assert sum(res["leaks_by_split"][s]["CARD"] for s in res["leaks_by_split"]) == res["leaks"]["CARD"]
    reasons = verdict({"CARD": 0}, res, allow_person_leaks=True)
    assert any("CARD" in r for r in reasons) and not any("PERSON" in r for r in reasons)
    assert any("PERSON" in r for r in verdict({"CARD": 0}, res, allow_person_leaks=False))


INDEX_DIR = ROOT / "data" / "index" / "chroma"


@pytest.mark.skipif(not INDEX_DIR.exists(), reason="no built index; run scripts/build_index.py")
def test_built_index_has_no_structured_pii():
    """The real index: the regex layer finds nothing in any stored document."""
    from callsearch.rag.store import CallIndex

    docs = [t for _, t in CallIndex(INDEX_DIR).documents()]
    assert docs and not any(pattern_scan(docs).values())


# --- keyword and hybrid search -------------------------------------------------------------

def test_tokenize_keeps_times_prices_ids_and_drops_placeholders():
    from callsearch.rag.bm25 import tokenize

    assert tokenize("TR7423 leaves at 05:00, 18.88 pounds. Thanks [PERSON]! saint john's") == [
        "tr7423", "leaves", "05:00", "18.88", "pounds", "thanks", "saint", "john's"]
    assert tokenize("[CARD] [PHONE]") == []


def test_bm25_prefers_rare_exact_tokens():
    from callsearch.rag.bm25 import BM25

    docs = ["the train leaves at 05:00", "the train leaves at 07:35", "a table for two", "train train train"]
    top = BM25(docs).top("train at 05:00", 2)
    assert top[0][0] == 0 and len(top) == 2
    assert BM25(docs).top("05:00", 5, allowed={1, 2}) == []
    assert BM25([]).top("anything", 3) == []


def test_rrf_rewards_agreement_and_keeps_first_seen_order_on_ties():
    from callsearch.rag.bm25 import RRF_K, rrf

    fused = rrf([["a", "b", "c"], ["b", "d"]])
    assert [cid for cid, _ in fused] == ["b", "a", "d", "c"]
    assert fused[0][1] == 1 / (RRF_K + 2) + 1 / (RRF_K + 1)
    assert [cid for cid, _ in rrf([["x"], ["y"]])] == ["x", "y"]


def test_search_modes(built):
    from callsearch.rag.store import CallIndex

    tmp_path, _ = built
    index = CallIndex(tmp_path / "idx", embedder=FakeEmbedder())
    for mode in ("dense", "bm25", "hybrid"):
        top = index.search("refund subscription", k=2, mode=mode)
        assert top[0].chunk_id == "call_9001:1" and len(top) <= 2, mode
    assert all(h.split == "train" for h in index.search("parcel house", k=3, split="train", mode="bm25"))
    assert all(h.split == "dev" for h in index.search("parcel house", k=3, split="dev", mode="hybrid"))
    assert index.search("zzzz", k=3, mode="bm25") == []
    with pytest.raises(ValueError, match="unknown search mode"):
        index.search("refund", mode="magic")


def test_with_neighbors_returns_ordered_windows(built):
    from callsearch.rag.store import CallIndex

    tmp_path, _ = built
    index = CallIndex(tmp_path / "idx", embedder=FakeEmbedder())
    hit = index.search("refund subscription", k=1)[0]  # call_9001:1, the last turn of its call
    assert [w.chunk_id for w in index.with_neighbors([hit], 1)[0]] == ["call_9001:0", "call_9001:1"]
    assert [w.chunk_id for w in index.with_neighbors([hit], 0)[0]] == ["call_9001:1"]
    first = index.turns(["call_9002:0"])["call_9002:0"]
    assert [w.chunk_id for w in index.with_neighbors([first], 2)[0]] == ["call_9002:0", "call_9002:1"]
    assert index.turns([]) == {} and index.turns(["call_9999:0"]) == {}
