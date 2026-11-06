"""Tests for the RAG eval: question-set validation, retrieval and answer scoring, committed question files."""

import json
from pathlib import Path

import pytest

from callsearch.eval.rag_eval import (QuestionSetError, answer_outcome, load_questions, retrieval_outcome, run,
                                      summarize, validate)
from callsearch.rag.answer import NOT_FOUND, Answer, Citation
from callsearch.rag.store import Hit

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "data" / "questions"

DOCS = {
    "call_0001:0": "Hello, which hotel?", "call_0001:1": "The Bridge Guest House, reference ABC123.",
    "call_0002:0": "Hello, which hotel?", "call_0002:1": "The Bridge Guest House, reference XYZ999.",
    "call_0003:0": "Train TR1234 leaves at 09:15.",
}


def aq(**kw):
    return {"id": "a1", "question": "q", "call_id": "call_0001", "turn_ids": [1], "evidence": ["abc123"],
            "answer_terms": ["abc123"], **kw}


def hit(cid):
    call, turn = cid.split(":")
    return Hit(cid, call, int(turn), "agent", 0.0, "test", DOCS.get(cid, ""), 0.5)


def test_validate_accepts_unique_evidence():
    assert validate([aq(), {"id": "u1", "question": "q", "call_id": None, "absent": ["helicopter"]}], DOCS) == []


def test_validate_rejects_shared_evidence():
    probs = validate([aq(evidence=["bridge guest house"])], DOCS)
    assert len(probs) == 1 and "also matches 1 other calls" in probs[0]


def test_validate_rejects_missing_evidence_turns_and_calls():
    assert "not in call_0001" in validate([aq(evidence=["nope"])], DOCS)[0]
    assert "turns [7]" in validate([aq(turn_ids=[7])], DOCS)[0]
    assert "not in the index" in validate([aq(call_id="call_0099")], DOCS)[0]
    assert "no answer_terms" in validate([aq(answer_terms=[])], DOCS)[0]


def test_validate_rejects_answerable_unanswerable():
    probs = validate([{"id": "u1", "question": "q", "call_id": None, "absent": ["tr1234", "09:15"]}], DOCS)
    assert probs and "call_0003:0" in probs[0]


def test_load_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps(aq()) + "\n" + json.dumps(aq()) + "\n")
    with pytest.raises(QuestionSetError):
        load_questions(p)


def test_retrieval_outcome_ranks():
    hits = [hit("call_0002:1"), hit("call_0001:0"), hit("call_0001:1")]
    assert retrieval_outcome(aq(), hits) == {"rank": 2, "turn_ranks": [3]}
    assert retrieval_outcome(aq(call_id="call_0003", turn_ids=[0]), hits) == {"rank": None, "turn_ranks": []}


def answer(text, cited=(), rejected=None):
    cits = [Citation(c, c.split(":")[0], int(c.split(":")[1]), "agent", 0.0, "") for c in cited]
    return Answer("q", text, text != NOT_FOUND, cits, [], rejected)


def test_answer_outcome():
    assert answer_outcome(aq(), answer("Reference ABC123.", ["call_0001:1"])) == {
        "found": True, "rejected": False, "citation_ok": True, "term_match": True}
    assert answer_outcome(aq(), answer("Reference XYZ999.", ["call_0002:1"]))["citation_ok"] is False
    assert answer_outcome(aq(), answer(NOT_FOUND))["term_match"] is False
    u = {"id": "u1", "call_id": None}
    assert answer_outcome(u, answer(NOT_FOUND))["abstained"] is True
    assert answer_outcome(u, answer("Something.", ["call_0003:0"]))["abstained"] is False


def test_summarize_rates():
    rows = [{"answerable": True, "rank": 1, "turn_ranks": [1], "found": True, "citation_ok": True, "term_match": True,
             "rejected": False},
            {"answerable": True, "rank": 4, "turn_ranks": [], "found": False, "citation_ok": False,
             "term_match": False, "rejected": True},
            {"answerable": False, "found": False, "abstained": True, "rejected": False}]
    s = summarize(rows, k=5, with_answers=True)
    assert (s["recall@1"], s["recall@3"], s["recall@5"]) == (0.5, 0.5, 1.0)
    assert s["turn_hit@5"] == 0.5 and s["mrr@10"] == round((1 + 0.25) / 2, 4)
    assert s["citation_accuracy"] == 0.5 and s["false_not_found"] == 0.5 and s["abstention_rate"] == 1.0
    assert s["rejected_replies"] == 1 and s["n_answerable"] == 2 and s["n_unanswerable"] == 1
    assert "citation_accuracy" not in summarize(rows, k=5, with_answers=False)


class FakeIndex:
    def search(self, query, k=5, split=None, mode="dense"):
        assert split is None and mode == "hybrid"  # the eval searches the whole index, in the mode asked for
        return [hit("call_0001:1"), hit("call_0002:1")][:k]


class FakeAnswerer:
    def ask(self, question):
        return answer("Reference ABC123.", ["call_0001:1"]) if "ref" in question else answer(NOT_FOUND)


def test_run_end_to_end():
    qs = [aq(question="ref?"), {"id": "u1", "question": "boat?", "call_id": None, "absent": ["boat"]}]
    metrics, rows = run(qs, FakeIndex(), FakeAnswerer(), k=5, mode="hybrid")
    assert metrics["recall@1"] == 1.0 and metrics["citation_accuracy"] == 1.0 and metrics["abstention_rate"] == 1.0
    assert [r["id"] for r in rows] == ["a1", "u1"]
    assert all(set(r) <= {"id", "answerable", "rank", "turn_ranks", "found", "rejected", "citation_ok", "term_match",
                          "abstained"} for r in rows)


# --- committed question files ------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(QDIR.glob("rag_*.jsonl")), ids=lambda p: p.name)
def test_question_files_are_well_formed(path):
    qs = load_questions(path)
    split = path.stem.removeprefix("rag_")
    for q in qs:
        assert q["id"].startswith(f"{split}_") and q["question"].endswith("?")
        if q["call_id"]:
            assert set(q) == {"id", "question", "call_id", "turn_ids", "evidence", "answer_terms"}
            assert q["evidence"] and q["answer_terms"] and all(e == e.lower() for e in q["evidence"])
        else:
            assert set(q) == {"id", "question", "call_id", "absent"} and q["absent"]


RAW = ROOT / "data" / "raw" / "calls.jsonl"


@pytest.mark.skipif(not RAW.exists(), reason="no generated data; run scripts/generate_data.py")
@pytest.mark.parametrize("path", sorted(QDIR.glob("rag_*.jsonl")), ids=lambda p: p.name)
def test_question_files_match_their_split(path):
    """Each answerable question points at a call of its own split, and the set is valid on raw text."""
    calls = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    split = {c["call_id"]: c["split"] for c in calls}
    docs = {f"{c['call_id']}:{t['turn_id']}": t["text"] for c in calls for t in c["turns"]}
    qs = load_questions(path)
    assert all(split[q["call_id"]] == path.stem.removeprefix("rag_") for q in qs if q["call_id"])
    assert validate(qs, docs) == []
