"""Tests for cited answering: prompt contents, server-side citation checks, NOT_FOUND, cache."""

import json

import pytest

from callsearch.rag.answer import NOT_FOUND, PROMPTS, Answerer, build_user_message, check_reply, parse_reply
from callsearch.rag.store import Hit


def hit(call_id, turn_id, text, speaker="customer"):
    return Hit(f"{call_id}:{turn_id}", call_id, turn_id, speaker, 3.0 * turn_id, "test", text, 0.8)


HITS = [hit("call_0001", 0, "I want a refund for the subscription I cancelled."),
        hit("call_0002", 3, "My card [CARD] was charged twice for the gym."),
        hit("call_0003", 1, "Can I book a table for two?")]


class FakeIndex:
    def __init__(self, hits=HITS):
        self.hits = hits
        self.calls = []

    def search(self, query, k=5, split=None):
        self.calls.append((query, k, split))
        return self.hits[:k]


class FakeProvider:
    name, model = "fake", "fake-1"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, system, user):
        self.seen.append((system, user))
        return self.replies.pop(0)


def reply(answer, citations):
    return json.dumps({"answer": answer, "citations": citations})


def test_user_message_numbers_excerpts_with_ids():
    msg = build_user_message("why a refund?", HITS)
    assert msg.startswith("Question: why a refund?")
    assert "[1] call call_0001, turn 0, customer at 0s: I want a refund" in msg
    assert "[3] call call_0003, turn 1" in msg and "[4]" not in msg


@pytest.mark.parametrize("raw", ['```json\n{"answer": "x", "citations": [1]}\n```', 'Sure! {"answer": "x", "citations": [1]}'])
def test_parse_reply_tolerates_wrapping(raw):
    assert parse_reply(raw)["answer"] == "x"


def test_check_reply_cases():
    assert check_reply(reply("A refund.", [1, "2", 1]), HITS) == ("A refund.", [0, 1], None)  # dedupe, digit str
    assert check_reply(reply("A refund.", ["call_0003:1"]), HITS) == ("A refund.", [2], None)  # chunk id
    assert check_reply(reply("NOT_FOUND", []), HITS) == (NOT_FOUND, [], None)
    assert check_reply(reply("not_found.", [1]), HITS) == (NOT_FOUND, [], None)
    assert check_reply(reply("A refund.", []), HITS)[2] == "answer had no citations"
    assert check_reply(json.dumps({"answer": "A refund."}), HITS)[2] == "answer had no citations"
    assert check_reply(reply("A refund.", [1, 4]), HITS)[2] == "cited 4, which was not retrieved"
    assert check_reply(reply("A refund.", [0]), HITS)[2] == "cited 0, which was not retrieved"
    assert check_reply(reply("A refund.", ["call_0999:0"]), HITS)[2].startswith("cited 'call_0999:0'")
    assert check_reply(reply("A refund.", [True]), HITS)[2] is not None
    assert check_reply("no json here", HITS) == (NOT_FOUND, [], "unparseable reply")


def test_ask_maps_citations_to_calls(tmp_path):
    prov = FakeProvider(reply("They cancelled a subscription and want a refund.", [1]))
    idx = FakeIndex()
    a = Answerer(idx, prov, k=3, cache_dir=tmp_path).ask("why a refund?", split="test")
    assert a.found and a.rejected is None
    assert [(c.call_id, c.turn_id, c.snippet) for c in a.citations] == [
        ("call_0001", 0, "I want a refund for the subscription I cancelled.")]
    assert a.retrieved == ["call_0001:0", "call_0002:3", "call_0003:1"]
    assert idx.calls == [("why a refund?", 3, "test")]
    assert a.provenance == {"provider": "fake", "model": "fake-1", "prompt_version": "v1", "k": 3}
    system, user = prov.seen[0]
    assert system == PROMPTS["v1"] and "[CARD]" in user  # only redacted index text goes in


def test_bad_citation_is_discarded(tmp_path):
    a = Answerer(FakeIndex(), FakeProvider(reply("Made up.", [9])), cache_dir=tmp_path).ask("q")
    assert not a.found and a.answer == NOT_FOUND and a.citations == [] and "not retrieved" in a.rejected


def test_nothing_retrieved_skips_the_model(tmp_path):
    prov = FakeProvider()
    a = Answerer(FakeIndex([]), prov, cache_dir=tmp_path).ask("q")
    assert not a.found and a.rejected == "nothing retrieved" and prov.seen == []


def test_cache_reuses_reply(tmp_path):
    prov = FakeProvider(reply("A refund.", [1]))
    first = Answerer(FakeIndex(), prov, cache_dir=tmp_path)
    first.ask("why?")
    second = Answerer(FakeIndex(), FakeProvider(), cache_dir=tmp_path)  # would fail if it called the model
    assert second.ask("why?").found and (second.requests, second.cache_hits) == (0, 1)
    assert (first.requests, first.cache_hits) == (1, 0)


def test_unknown_prompt_version():
    with pytest.raises(KeyError):
        Answerer(prompt_version="v9")


def test_to_dict_is_json():
    a = Answerer(FakeIndex(), FakeProvider(reply("A refund.", [2])), cache_dir=None).ask("q")
    d = json.loads(json.dumps(a.to_dict()))
    assert d["citations"][0]["chunk_id"] == "call_0002:3" and d["found"] is True
