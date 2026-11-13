"""Evaluate the agent: tool selection, task success and steps, and agent vs the fixed pipeline.

python -m callsearch.eval.agent_eval                       # data/questions/agent_test.jsonl
python -m callsearch.eval.agent_eval --compare-rag         # also the 54 single-step test questions

Multi-step questions come from scripts/make_agent_questions.py (answers computed from the data).
  tool_selection   every expected tool was called (answerable questions)
  tool_order       ... and in the expected order (a subsequence of the calls made)
  task_success     answerable: found, every answer_terms group matched, and a cited call is in
                   cite_any when that list is set; unanswerable: the agent said not found
  avg_steps        tool calls per question; avg_llm_calls also counts the final answer
With --compare-rag the single-step test questions run through the agent and are scored the
same way as rag_eval (citation_accuracy, answer_term_match, abstention_rate), next to the
saved fixed-pipeline results, to show whether the agent helps or only adds model calls.
Model replies are cached in .cache/agent/, so a re-run is free. Results hold ids and metrics,
never transcript text.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_QUESTIONS = Path("data/questions/agent_test.jsonl")
RAG_QUESTIONS = Path("data/questions/rag_test.jsonl")
PIPELINE_RESULTS = Path("results/rag_eval_test_bm25_n1.json")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_term(answer: str, alternatives: list[str]) -> bool:
    text = answer.lower()
    return any(re.search(rf"(?<![\w.]){re.escape(a.lower())}(?!\w)", text) for a in alternatives)


def is_subsequence(want: list[str], got: list[str]) -> bool:
    it = iter(got)
    return all(w in it for w in want)


def score(q: dict, a) -> dict:
    used = a.tools_used
    row = {"id": q["id"], "type": q.get("type"), "answerable": q["answerable"], "found": a.found,
           "tools": used, "steps": len(a.steps), "llm_calls": a.llm_calls, "rejected": a.rejected,
           "cited_calls": a.cited_calls, "cited_results": a.cited_results}
    if q["answerable"]:
        row["tool_selection"] = set(q["expected_tools"]) <= set(used)
        row["tool_order"] = is_subsequence(q["expected_tools"], used)
        row["terms_ok"] = a.found and all(has_term(a.answer, g) for g in q["answer_terms"])
        row["cite_ok"] = not q["cite_any"] or bool(set(q["cite_any"]) & set(a.cited_calls))
        row["success"] = row["terms_ok"] and row["cite_ok"]
    else:
        row["success"] = not a.found
    return row


def rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    out = {"n": len(rows), "n_answerable": len(ans),
           "tool_selection": rate(sum(r["tool_selection"] for r in ans), len(ans)),
           "tool_order": rate(sum(r["tool_order"] for r in ans), len(ans)),
           "task_success": rate(sum(r["success"] for r in rows), len(rows)),
           "answerable_success": rate(sum(r["success"] for r in ans), len(ans)),
           "unanswerable_refused": rate(sum(r["success"] for r in rows if not r["answerable"]),
                                        len(rows) - len(ans)),
           "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 2) if rows else 0.0,
           "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / len(rows), 2) if rows else 0.0,
           "step_cap_hit": sum(bool(r["rejected"] and "step cap" in r["rejected"]) for r in rows),
           "rejected": sum(r["rejected"] is not None for r in rows)}
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    out["by_type"] = {t: {"n": len(rs), "success": rate(sum(r["success"] for r in rs), len(rs)),
                          "avg_steps": round(sum(r["steps"] for r in rs) / len(rs), 2)} for t, rs in by_type.items()}
    out["tool_counts"] = dict(Counter(t for r in rows for t in r["tools"]))
    return out


def rag_outcome(q: dict, a) -> dict:
    """Same scoring as rag_eval.answer_outcome, for the agent's answers."""
    out = {"id": q["id"], "found": a.found, "steps": len(a.steps), "llm_calls": a.llm_calls, "tools": a.tools_used}
    if q.get("call_id"):
        out["citation_ok"] = q["call_id"] in a.cited_calls
        out["term_match"] = a.found and any(t.lower() in a.answer.lower() for t in q["answer_terms"])
    else:
        out["abstained"] = not a.found
    return out


def rag_summary(rows: list[dict]) -> dict:
    ans = [r for r in rows if "citation_ok" in r]
    una = [r for r in rows if "abstained" in r]
    return {"n_answerable": len(ans), "n_unanswerable": len(una),
            "answer_found": rate(sum(r["found"] for r in ans), len(ans)),
            "citation_accuracy": rate(sum(r["citation_ok"] for r in ans), len(ans)),
            "answer_term_match": rate(sum(r["term_match"] for r in ans), len(ans)),
            "abstention_rate": rate(sum(r["abstained"] for r in una), len(una)),
            "avg_llm_calls": round(sum(r["llm_calls"] for r in rows) / len(rows), 2) if rows else 0.0,
            "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 2) if rows else 0.0}


def run_all(agent, questions: list[dict], scorer) -> tuple[list[dict], float]:
    rows, t0 = [], time.perf_counter()
    for i, q in enumerate(questions, 1):
        a = agent.run(q["question"])
        rows.append(scorer(q, a))
        print(f"  [{i}/{len(questions)}] {q['id']}: {'ok ' if rows[-1].get('success', a.found) else '-- '}"
              f"{len(a.steps)} steps {'/'.join(a.tools_used) or '-'}", flush=True)
    return rows, time.perf_counter() - t0


def main() -> None:
    from callsearch.agent.loop import MAX_STEPS, Agent
    from callsearch.agent.tools import Tools
    from callsearch.api.services import CallStore, ResultsReader
    from callsearch.rag.store import CallIndex

    p = argparse.ArgumentParser(description="Evaluate the tool-using agent.")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--compare-rag", action="store_true", help="also run the single-step test questions")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    agent = Agent(Tools(CallStore(), CallIndex(), ResultsReader()), max_steps=args.max_steps)
    print(f"agent: {agent.provider.name} {agent.provider.model}, prompt {agent.prompt_version}, "
          f"step cap {args.max_steps}")
    rows, secs = run_all(agent, load(args.questions), score)
    result = {"questions": str(args.questions), "agent": agent.provenance(), **summarize(rows),
              "seconds": round(secs, 1), "llm_requests": agent.requests, "llm_cache_hits": agent.cache_hits,
              "rows": rows}
    if args.compare_rag:
        rag_rows, rag_secs = run_all(agent, load(RAG_QUESTIONS), rag_outcome)
        pipeline = json.loads(PIPELINE_RESULTS.read_text(encoding="utf-8")) if PIPELINE_RESULTS.exists() else {}
        keys = ["answer_found", "citation_accuracy", "answer_term_match", "abstention_rate"]
        result["single_step"] = {"agent": {**rag_summary(rag_rows), "seconds": round(rag_secs, 1)},
                                 "pipeline": {**{k: pipeline.get(k) for k in keys}, "avg_llm_calls": 1.0,
                                              "model": (pipeline.get("llm") or {}).get("model")},
                                 "rows": rag_rows}
    out = args.out or Path(f"results/agent_eval_{args.questions.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\ntool selection {result['tool_selection']:.0%} (in order {result['tool_order']:.0%}), "
          f"task success {result['task_success']:.0%} "
          f"(answerable {result['answerable_success']:.0%}, unanswerable refused {result['unanswerable_refused']:.0%}), "
          f"avg steps {result['avg_steps']}, step cap hit {result['step_cap_hit']}x")
    for t, s in result["by_type"].items():
        print(f"  {t:<18} {s['success']:.0%} of {s['n']}, avg steps {s['avg_steps']}")
    if args.compare_rag:
        a, pl = result["single_step"]["agent"], result["single_step"]["pipeline"]
        print("\nsingle-step test questions    agent   pipeline")
        for k in ["answer_found", "citation_accuracy", "answer_term_match", "abstention_rate", "avg_llm_calls"]:
            print(f"  {k:<26} {a[k]:>6}   {pl.get(k)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
