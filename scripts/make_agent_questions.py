"""Build the multi-step agent question set from the redacted test split.

python scripts/make_agent_questions.py   -> data/questions/agent_test.jsonl

Expected answers are computed from the data, not written by hand, so they are exact:
speaker counts and call counts come from the same counting the compute_stats tool does
(unit-tested separately), turn counts from the transcripts, and redaction numbers from
results/. Each question lists the tools it needs (expected_tools), answer_terms (every
group must match one alternative, case-insensitive) and cite_any (the answer must cite at
least one of these calls when the list is not empty). Uploaded calls are in their own
split, so they never change these answers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from callsearch.agent.tools import stats
from callsearch.api.services import ResultsReader
from callsearch.rag.bm25 import terms
from callsearch.redact.export import load_redacted

OUT = Path("data/questions/agent_test.jsonl")
SPEAKER_TERMS = ["restaurant", "hotel", "train", "parking", "fee", "address"]
COUNT_TERMS = ["wifi", "museum", "payment", "late"]
ABSENT = [("sauna", "Which test-split call mentions a sauna, and what did the agent offer the customer?"),
          ("lottery", "In the test split, how did agents respond when a customer said they won the lottery?")]


def pct_forms(x: float) -> list[str]:
    return list(dict.fromkeys([f"{x:.2f}", f"{x:.3f}", f"{x * 100:.1f}%", f"{round(x * 100)}%", f"{x * 100:.1f} %"]))


def main() -> None:
    calls, _ = load_redacted(Path("data/redacted"))
    test = [c for c in calls if c["split"] == "test"]
    by_id = {c["call_id"]: c for c in test}
    vocab = {t for c in calls for turn in c["turns"] for t in terms(turn["text"])}
    qs = []

    def add(kind, question, tools, answer_terms, cite_any=(), answerable=True):
        qs.append({"id": f"agent_{len(qs) + 1:02d}", "type": kind, "question": question, "expected_tools": tools,
                   "answer_terms": answer_terms, "cite_any": list(cite_any), "answerable": answerable})

    def matching_calls(word):
        return [c["call_id"] for c in test if any(word in terms(t["text"]) for t in c["turns"])]

    for word in SPEAKER_TERMS:
        m = stats(test, word)["matching"]["turns_by_speaker"]
        win, lose = sorted(m, key=m.get, reverse=True)
        if m[win] < 1.3 * m[lose]:
            sys.exit(f"{word}: speakers too close ({m}); pick another word")
        add("speaker_compare",
            f"In the test split, do agents or customers mention '{word}' in more turns, and how many turns from "
            f"each? Give one example of what was said.",
            ["compute_stats", "search_calls"], [[win, f"{win}s"], [str(m[win])], [str(m[lose])]], matching_calls(word))

    for word in COUNT_TERMS:
        n = stats(test, word)["matching"]["calls"]
        add("count_example", f"How many test-split calls mention '{word}'? Name one of them and say what was said.",
            ["compute_stats", "search_calls"], [[str(n)]], matching_calls(word))

    rag = [json.loads(line) for line in Path("data/questions/rag_test.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    for q in [q for q in rag if q.get("call_id") in by_id][:5]:
        n = len(by_id[q["call_id"]]["turns"])
        add("locate_then_read", f"{q['question']} Also, how many turns does that call have?",
            ["search_calls", "get_transcript"], [q["answer_terms"], [str(n)]], [q["call_id"]])

    hf = next(d for d in ResultsReader().redaction() if d.detector == "pipeline_hf")
    persons = stats(test)["redactions"].get("PERSON", 0)
    add("report_and_stats",
        "What name recall does the Regex + DistilBERT method get on the test split, and how many [PERSON] "
        "placeholders are there in test-split calls?",
        ["redaction_report", "compute_stats"], [pct_forms(hf.person_recall), [str(persons)]])
    share = stats([c for c in test if c["source"] == "template"])["agent_talk_share"]
    add("report_and_stats",
        "Did the search index check pass, and what share of the words in test-split template calls do agents speak?",
        ["redaction_report", "compute_stats"], [["pass", "passed", "yes"], pct_forms(share)])

    m = stats(test, "cancel")["matching"]["turns_by_speaker"]
    win = max(m, key=m.get)
    add("three_tools",
        "In the test split, which speaker says 'cancel' in more turns? Then read one of those calls in full and "
        "say what the customer wanted cancelled and why.",
        ["compute_stats", "search_calls", "get_transcript"], [[win, f"{win}s"]], matching_calls("cancel"))

    for word, question in ABSENT:
        if word in vocab:
            sys.exit(f"{word!r} occurs in the data; pick another unanswerable topic")
        add("unanswerable", question, ["search_calls"], [], answerable=False)

    OUT.write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8")
    print(f"wrote {len(qs)} questions to {OUT}")


if __name__ == "__main__":
    main()
