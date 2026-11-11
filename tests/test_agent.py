"""Agent: tools over redacted data, the loop's rules, and the PII guard. Offline (scripted model)."""

import builtins
import json
import sys
from pathlib import Path

import pytest

from callsearch.agent.loop import Agent, build_message, first_json
from callsearch.agent.tools import Tools, stats
from callsearch.api.services import CallStore, ResultsReader
from callsearch.rag.answer import NOT_FOUND
from callsearch.redact.export import redact_calls, write_redacted

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_rag import FakeEmbedder, LabelDetector  # noqa: E402


def turn(i, speaker, text, sec=None):
    return {"turn_id": i, "speaker": speaker, "start_sec": float(sec if sec is not None else 5 * i), "text": text}


CALLS = [
    {"call_id": "call_0001", "split": "test", "source": "template", "turns": [
        turn(0, "agent", "Accounts, this is [PERSON]."),
        turn(1, "customer", "I want to cancel my card, it was stolen."),
        turn(2, "agent", "I can cancel it now. A new card ships Friday.")]},
    {"call_id": "call_0002", "split": "test", "source": "multiwoz", "turns": [
        turn(0, "customer", "I need a train to Ely on Saturday."),
        turn(1, "agent", "The 05:59 train arrives at 06:15."),
        turn(2, "customer", "Book it, and I also need a hotel with parking.")]},
    {"call_id": "call_0003", "split": "dev", "source": "multiwoz", "turns": [
        turn(0, "customer", "Is there a train to London?"),
        turn(1, "agent", "Yes, hourly.")]},
]


class Script:
    """A fake model that replies from a list, recording every prompt it saw."""
    name, model = "script", "script-1"

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.seen = []

    def complete(self, system, user):
        self.seen.append(user)
        return self.replies.pop(0) if self.replies else json.dumps({"answer": NOT_FOUND, "citations": []})


@pytest.fixture
def tools(tmp_path):
    from build_index import build
    from callsearch.rag.store import CallIndex

    write_redacted(CALLS, tmp_path / "red", "pipeline_hf")
    build(tmp_path / "red", tmp_path / "idx", FakeEmbedder(), context_turns=1)
    return Tools(CallStore(redacted_dir=tmp_path / "red"), CallIndex(tmp_path / "idx", FakeEmbedder()),
                 ResultsReader(ROOT / "results"))


def agent(tools, *replies, max_steps=6):
    return Agent(tools, Script(*replies), max_steps=max_steps, cache_dir=None)


# --- tools ----------------------------------------------------------------------------------

def test_stats_counts_speakers_matches_and_redactions():
    s = stats(CALLS[:2], "train")
    assert s["calls"] == 2 and s["turns"] == 6
    assert s["turns_by_speaker"] == {"agent": 3, "customer": 3}
    assert s["matching"] == {"query": "train", "query_terms": ["train"], "calls": 1,
                             "turns_by_speaker": {"agent": 1, "customer": 1}, "example_calls": ["call_0002"]}
    assert s["redactions"] == {"PERSON": 1}
    assert 0 < s["agent_talk_share"] < 1 and "matching" not in stats(CALLS)


def test_tools_filter_and_report_errors_the_model_can_fix(tools):
    r = tools.run("compute_stats", {"query": "train", "split": "test"})
    assert json.loads(r.text)["matching"]["calls"] == 1 and r.calls == ["call_0002"]
    hits = tools.run("search_calls", {"query": "cancel card", "k": 2})
    assert "call_0001:1" in hits.chunks and hits.summary.endswith("from 1 calls")
    t = tools.run("get_transcript", {"call_id": "call_0002"})
    assert t.text.startswith("call_0002: multiwoz call, test split, 3 turns") and len(t.chunks) == 3
    assert not tools.run("get_transcript", {"call_id": "call_9999"}).ok
    bad = tools.run("search_calls", {"query": "x", "path": "data/raw/calls.jsonl"})
    assert not bad.ok and bad.text.startswith("error: search_calls")
    assert not tools.run("read_file", {"path": "data/raw/calls.jsonl"}).ok
    report = json.loads(tools.run("redaction_report", {}).text)
    assert report["split"] == "test" and {m["method"] for m in report["methods"]} >= {"Regex + DistilBERT"}


def test_tools_refuse_a_raw_call_store():
    with pytest.raises(ValueError, match="redacted"):
        Tools(CallStore(redacted_dir=Path("data/raw")), None, None)


def test_summaries_never_contain_transcript_text(tools):
    for name, args in [("search_calls", {"query": "stolen card"}), ("get_transcript", {"call_id": "call_0001"}),
                       ("compute_stats", {"query": "stolen"})]:
        assert "stolen" not in tools.run(name, args).summary


# --- the loop -------------------------------------------------------------------------------

def test_multi_step_run_with_turn_call_and_result_citations(tools):
    a = agent(tools,
              {"tool": "compute_stats", "args": {"query": "train", "split": "test"}},
              {"tool": "search_calls", "args": {"query": "train Saturday", "k": 3}},
              {"tool": "get_transcript", "args": {"call_id": "call_0002"}},
              {"answer": "One test call mentions trains; the 05:59 train arrives at 06:15.",
               "citations": ["r1", "call_0002:1", "[call_0002]"]})
    out = a.run("How many test calls mention trains, and when does the Saturday train arrive?")
    assert out.found and out.tools_used == ["compute_stats", "search_calls", "get_transcript"]
    assert [c.chunk_id for c in out.citations] == ["call_0002:1"] and out.cited_calls == ["call_0002"]
    assert out.cited_results == [1] and out.llm_calls == 4
    assert "[r3] get_transcript" in a.provider.seen[-1] and "Tool calls left: 3." in a.provider.seen[-1]


@pytest.mark.parametrize("cite,why", [(["call_0002:9"], "was not in any tool result"), (["r7"], "not in any"),
                                      ([], "no citations"), ([True], "is not an id")])
def test_answers_with_unseen_citations_are_discarded(tools, cite, why):
    out = agent(tools, {"tool": "search_calls", "args": {"query": "train"}},
                {"answer": "Trains run hourly.", "citations": cite}).run("q?")
    assert not out.found and out.answer == NOT_FOUND and why in out.rejected


def test_error_results_cannot_be_cited(tools):
    out = agent(tools, {"tool": "get_transcript", "args": {"call_id": "call_9999"}},
                {"answer": "It does not exist.", "citations": ["r1"]}).run("q?")
    assert not out.found and "r1" in out.rejected


def test_step_cap_stops_the_loop(tools):
    replies = [{"tool": "search_calls", "args": {"query": f"train {i}"}} for i in range(10)]
    a = agent(tools, *replies, max_steps=3)
    out = a.run("q?")
    assert not out.found and len(out.steps) == 3 and "step cap of 3" in out.rejected
    assert "You must answer now." in a.provider.seen[-1] and out.llm_calls == 4


def test_bad_replies_and_repeats_use_steps_without_running_tools(tools):
    a = agent(tools, "I think I should search", {"tool": "search_calls", "args": {"query": "train"}},
              {"tool": "search_calls", "args": {"query": "train"}}, {"tool": "nope", "args": {}},
              {"answer": "There is a train to London.", "citations": ["call_0003:0"]})
    out = a.run("Is there a London train?")
    assert out.found and [s.summary for s in out.steps] == ["unreadable reply", "3 turns from 2 calls", "repeat of r2",
                                                            "unknown tool 'nope'"]
    assert "its result is [r2]" in a.provider.seen[3]


def test_not_found_is_a_valid_outcome(tools):
    out = agent(tools, {"tool": "search_calls", "args": {"query": "sauna"}},
                {"answer": "NOT_FOUND", "citations": []}).run("Which call mentions a sauna?")
    assert not out.found and out.rejected is None


def test_message_and_json_helpers():
    assert first_json('sure: ```json\n{"tool": "x"}\n```') == {"tool": "x"} and first_json("nope") is None
    assert build_message("q", [], 6).endswith("Tool calls left: 6.")


# --- PII guard ------------------------------------------------------------------------------

PLANTED = {"PERSON": "Marguerite Okonkwo", "CARD": "4539 1488 0343 6467", "EMAIL": "m.okonkwo@example.org"}


def test_agent_never_sees_or_returns_planted_values(tmp_path, monkeypatch):
    """Raw calls with planted values go through redaction; then the agent runs every tool over
    the result while any attempt to open data/raw fails the test."""
    from build_index import build
    from callsearch.rag.store import CallIndex

    raw_text = f"I'm {PLANTED['PERSON']}, card {PLANTED['CARD']}, email {PLANTED['EMAIL']}, refund please."
    raw = [{"call_id": "call_0100", "split": "test", "source": "template",
            "turns": [turn(0, "agent", "Refunds desk."), turn(1, "customer", raw_text)],
            "labels": [{"turn_id": 1, "start": raw_text.index(v), "end": raw_text.index(v) + len(v), "type": t}
                       for t, v in PLANTED.items()]}]
    write_redacted(redact_calls(raw, LabelDetector(raw)), tmp_path / "red", "pipeline_hf")
    build(tmp_path / "red", tmp_path / "idx", FakeEmbedder(), context_turns=1)
    tools = Tools(CallStore(redacted_dir=tmp_path / "red"), CallIndex(tmp_path / "idx", FakeEmbedder()),
                  ResultsReader(ROOT / "results"))

    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        assert "data/raw" not in str(file).replace("\\", "/"), f"agent tried to open {file}"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    a = agent(tools, {"tool": "search_calls", "args": {"query": "refund card email"}},
              {"tool": "get_transcript", "args": {"call_id": "call_0100"}},
              {"tool": "compute_stats", "args": {"query": "refund"}}, {"tool": "redaction_report", "args": {}},
              {"answer": "The customer asked for a refund.", "citations": ["call_0100:1"]})
    out = a.run("What did the refunds caller want?")
    trace = json.dumps(out.to_dict()) + "".join(a.provider.seen) + "".join(s.result.text for s in out.steps)
    assert out.found and out.tools_used == ["search_calls", "get_transcript", "compute_stats", "redaction_report"]
    for value in PLANTED.values():
        assert value not in trace and value.split()[0] not in trace
    assert "[PERSON]" in trace and "[CARD]" in trace and "[EMAIL]" in trace


# --- HTTP -----------------------------------------------------------------------------------

def test_agent_route_returns_steps_and_citations(tools, tmp_path):
    from fastapi.testclient import TestClient

    from callsearch.api.main import create_app
    from callsearch.api.services import Services

    replies = [{"tool": "compute_stats", "args": {"query": "train", "split": "test"}},
               {"tool": "search_calls", "args": {"query": "train Saturday"}},
               {"answer": "1 test call; the 05:59 train.", "citations": ["r1", "call_0002:1"]}]
    svc = Services(calls=tools.calls, index=tools.index, results=tools.results,
                   agent=lambda t: Agent(t, Script(*replies), cache_dir=None))
    c = TestClient(create_app(svc, web_dist=Path("/nonexistent"), ask_per_minute=1))
    r = c.post("/api/agent", json={"question": "  How many test calls mention trains?  "})
    body = r.json()
    assert r.status_code == 200 and body["found"] and body["question"] == "How many test calls mention trains?"
    assert [s["tool"] for s in body["steps"]] == ["compute_stats", "search_calls"]
    assert body["steps"][0]["summary"] == "2 calls (test split), 1 matching the query"
    assert body["citations"][0]["chunk_id"] == "call_0002:1" and body["cited_results"] == [1]
    assert body["provenance"]["max_steps"] == 6 and body["llm_calls"] == 3
    assert c.post("/api/agent", json={"question": "again?"}).status_code == 429
    assert c.post("/api/agent", json={"question": "x"}).status_code in (422, 429)


# --- eval scoring and the question set ------------------------------------------------------

def test_term_matching_respects_number_boundaries():
    from callsearch.eval.agent_eval import has_term, is_subsequence

    assert has_term("Customers did, 18 to 8.", ["8"]) and not has_term("There were 18.", ["8"])
    assert has_term("Recall is 95.9%.", ["95.9%"]) and not has_term("0.956", ["0.95"])
    assert is_subsequence(["a", "c"], ["a", "b", "c"]) and not is_subsequence(["c", "a"], ["a", "b", "c"])


def test_scoring_success_needs_terms_and_a_good_citation(tools):
    from callsearch.eval.agent_eval import score, summarize

    q = {"id": "x", "type": "t", "question": "q", "answerable": True, "expected_tools": ["compute_stats", "search_calls"],
         "answer_terms": [["customer", "customers"], ["3"]], "cite_any": ["call_0002"]}
    good = agent(tools, {"tool": "compute_stats", "args": {}}, {"tool": "search_calls", "args": {"query": "train"}},
                 {"answer": "Customers, 3 turns.", "citations": ["call_0002:0"]}).run("q")
    wrong_call = agent(tools, {"tool": "search_calls", "args": {"query": "cancel"}},
                       {"answer": "Customers, 3 turns.", "citations": ["call_0001:1"]}).run("q")
    rows = [score(q, good), score(q, wrong_call), score({**q, "answerable": False, "id": "y"}, wrong_call)]
    assert [r["success"] for r in rows] == [True, False, False]
    assert rows[0]["tool_order"] and not rows[1]["tool_selection"]
    s = summarize(rows)
    assert s["tool_selection"] == 0.5 and s["task_success"] == round(1 / 3, 4) and s["unanswerable_refused"] == 0.0


def test_agent_question_set_is_consistent():
    from callsearch.agent.tools import TOOL_NAMES
    from callsearch.redact.export import load_redacted

    path = ROOT / "data/questions/agent_test.jsonl"
    if not path.exists() or not (ROOT / "data/redacted/manifest.json").exists():
        pytest.skip("question set or redacted data not present")
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test_ids = {c["call_id"] for c in load_redacted(ROOT / "data/redacted")[0] if c["split"] == "test"}
    assert len(qs) == 20 and len({q["id"] for q in qs}) == 20
    assert sum(len(q["expected_tools"]) >= 2 for q in qs if q["answerable"]) == 18
    for q in qs:
        assert set(q["expected_tools"]) <= set(TOOL_NAMES)
        assert set(q["cite_any"]) <= test_ids
        assert all(group for group in q["answer_terms"])
