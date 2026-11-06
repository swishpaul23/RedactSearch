"""Phase 3: Evaluate retrieval and answer quality with citation checks.

python -m callsearch.eval.rag_eval --split test [--mode hybrid] [--retrieval-only] [-k 5]

Question file: data/questions/rag_<split>.jsonl, one object per line.
  answerable:   {"id", "question", "call_id", "turn_ids", "evidence", "answer_terms"}
  unanswerable: {"id", "question", "call_id": null, "absent"}

Before scoring, every question is checked against the text actually stored in the index:
- answerable: the gold turns exist, every evidence phrase appears in the gold call, and no
  other call contains all of them. So each question has exactly one right call, even though
  template calls repeat their phrasing word for word.
- unanswerable: no stored turn contains all of the "absent" phrases.
A question that fails is an error in the question set, so the run stops and lists it.

Retrieval searches the whole index (all splits), as a user would. Metrics:
- recall@1/3/5/10: a chunk from the gold call is in the top n; turn_hit@k: a gold turn is in the top k
- MRR@10 over the gold call's first rank
Answers (unless --retrieval-only), using the top k chunks:
- answerable: found rate, citation accuracy (a cited chunk is from the gold call),
  answer-term match (any expected term in the answer text), rejected replies
- unanswerable: abstention rate (the reply is NOT_FOUND)
Saves results/rag_eval_<split>.json: counts, rates and per-question ids with outcomes; no text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

QUESTIONS_DIR = Path("data/questions")
RANKS = (1, 3, 5, 10)
SEARCH_DEPTH = max(RANKS)


class QuestionSetError(ValueError):
    pass


def load_questions(path: Path) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        raise QuestionSetError("duplicate question ids")
    return qs


def validate(questions: list[dict], docs: dict[str, str]) -> list[str]:
    """Problems with the question set against the stored chunks (chunk id -> text). Empty means valid."""
    by_call: dict[str, list[str]] = defaultdict(list)
    for cid, text in docs.items():
        by_call[cid.rsplit(":", 1)[0]].append(text.lower())
    problems = []
    for q in questions:
        if q.get("call_id"):
            call, ev = q["call_id"], [e.lower() for e in q["evidence"]]
            if call not in by_call:
                problems.append(f"{q['id']}: {call} is not in the index")
                continue
            missing_turns = [t for t in q["turn_ids"] if f"{call}:{t}" not in docs]
            if missing_turns:
                problems.append(f"{q['id']}: turns {missing_turns} of {call} are not in the index")
            text = "\n".join(by_call[call])
            missing = [e for e in ev if e not in text]
            if missing:
                problems.append(f"{q['id']}: evidence {missing} not in {call} (redacted away or mistyped?)")
            others = [c for c, ts in by_call.items() if c != call and all(e in "\n".join(ts) for e in ev)]
            if others:
                problems.append(f"{q['id']}: evidence also matches {len(others)} other calls, e.g. {others[:3]}")
            if not q.get("answer_terms"):
                problems.append(f"{q['id']}: no answer_terms")
        else:
            ab = [a.lower() for a in q["absent"]]
            hits = [cid for cid, t in docs.items() if all(a in t.lower() for a in ab)]
            if hits:
                problems.append(f"{q['id']}: unanswerable, but {hits[:3]} contain {ab}")
    return problems


def retrieval_outcome(q: dict, hits) -> dict:
    calls = [h.call_id for h in hits]
    rank = calls.index(q["call_id"]) + 1 if q["call_id"] in calls else None
    gold_turns = {f"{q['call_id']}:{t}" for t in q["turn_ids"]}
    return {"rank": rank, "turn_ranks": [i + 1 for i, h in enumerate(hits) if h.chunk_id in gold_turns]}


def answer_outcome(q: dict, a) -> dict:
    out = {"found": a.found, "rejected": a.rejected is not None}
    if q.get("call_id"):
        out["citation_ok"] = any(c.call_id == q["call_id"] for c in a.citations)
        out["term_match"] = a.found and any(t.lower() in a.answer.lower() for t in q["answer_terms"])
    else:
        out["abstained"] = not a.found
    return out


def summarize(rows: list[dict], k: int, with_answers: bool) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    rate = lambda xs, n: round(xs / n, 4) if n else 0.0
    res = {"n_answerable": len(ans), "n_unanswerable": len(una)}
    for n in RANKS:
        res[f"recall@{n}"] = rate(sum(r["rank"] is not None and r["rank"] <= n for r in ans), len(ans))
    res[f"turn_hit@{k}"] = rate(sum(any(t <= k for t in r["turn_ranks"]) for r in ans), len(ans))
    res["mrr@10"] = rate(sum(1 / r["rank"] for r in ans if r["rank"]), len(ans))
    if with_answers:
        res["answer_found"] = rate(sum(r["found"] for r in ans), len(ans))
        res["citation_accuracy"] = rate(sum(r["citation_ok"] for r in ans), len(ans))
        res["answer_term_match"] = rate(sum(r["term_match"] for r in ans), len(ans))
        res["false_not_found"] = rate(sum(not r["found"] for r in ans), len(ans))
        res["abstention_rate"] = rate(sum(r["abstained"] for r in una), len(una))
        res["rejected_replies"] = sum(r["rejected"] for r in rows)
    return res


def run(questions: list[dict], index, answerer=None, k: int = 5, mode: str = "dense") -> tuple[dict, list[dict]]:
    rows = []
    for q in questions:
        hits = index.search(q["question"], k=SEARCH_DEPTH, mode=mode)
        row = {"id": q["id"], "answerable": bool(q.get("call_id"))}
        if row["answerable"]:
            row.update(retrieval_outcome(q, hits))
        if answerer is not None:
            row.update(answer_outcome(q, answerer.ask(q["question"])))
        rows.append(row)
    return summarize(rows, k, answerer is not None), rows


def main() -> None:
    from callsearch.rag.answer import Answerer
    from callsearch.rag.store import DEFAULT_INDEX_DIR, CallIndex

    p = argparse.ArgumentParser(description="Evaluate retrieval and cited answers on a question set.")
    p.add_argument("--split", default="dev", choices=["dev", "test"])
    p.add_argument("--questions", type=Path, help="default: data/questions/rag_<split>.jsonl")
    p.add_argument("-k", type=int, default=5, help="chunks given to the answering model")
    p.add_argument("--retrieval-only", action="store_true", help="skip the LLM; retrieval metrics only")
    p.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"], help="retrieval mode")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--json", type=Path, help="default: results/rag_eval_<split>[_<mode>][_retrieval].json "
                                             "(no mode suffix for dense)")
    args = p.parse_args()

    questions = load_questions(args.questions or QUESTIONS_DIR / f"rag_{args.split}.jsonl")
    index = CallIndex(args.index_dir)
    problems = validate(questions, dict(index.documents()))
    if problems:
        print("question set does not match the index:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(1)

    answerer = None if args.retrieval_only else Answerer(index=index, k=args.k, mode=args.mode)
    metrics, rows = run(questions, index, answerer, args.k, args.mode)
    info = index.info
    res = {"split": args.split, "k": args.k, "retrieval": args.mode, "index": {key: info.get(key) for key in
                                                      ("embed_model", "context_turns", "detector", "n_chunks")},
           "llm": None if answerer is None else answerer.provenance(), **metrics,
           "llm_requests": 0 if answerer is None else answerer.requests,
           "llm_cache_hits": 0 if answerer is None else answerer.cache_hits, "questions": rows}

    print(f"RAG EVAL  split {args.split}  {metrics['n_answerable']} answerable + {metrics['n_unanswerable']} "
          f"unanswerable  {args.mode} retrieval  index {res['index']['embed_model']} "
          f"context {res['index']['context_turns']}")
    for key in [f"recall@{n}" for n in RANKS] + [f"turn_hit@{args.k}", "mrr@10", "answer_found", "citation_accuracy",
                                                   "answer_term_match", "false_not_found", "abstention_rate"]:
        if key in metrics:
            print(f"  {key:<20}{metrics[key]:.3f}")
    if answerer is not None:
        pv = answerer.provenance()
        print(f"  rejected replies    {metrics['rejected_replies']}   ({pv['provider']}:{pv['model']} prompt "
              f"{pv['prompt_version']}, {answerer.requests} requests, {answerer.cache_hits} cached)")
    missed = [r["id"] for r in rows if r["answerable"] and not (r["rank"] and r["rank"] <= args.k)]
    if missed:
        print(f"  not in top {args.k}: {', '.join(missed)}")
    suffix = ("" if args.mode == "dense" else f"_{args.mode}") + ("_retrieval" if args.retrieval_only else "")
    out = args.json or Path("results") / f"rag_eval_{args.split}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
