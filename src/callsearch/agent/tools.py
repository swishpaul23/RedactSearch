"""The agent's four tools. Every one reads redacted sources only.

search_calls(query, k, split, source)   turns from the search index
get_transcript(call_id)                 one redacted call, turn by turn
compute_stats(query, split, source)     counts by speaker, talk share, and matches for a query
redaction_report()                      saved test-split redaction results and the index check

What they can reach is fixed in code, not in the prompt: the tools get a CallStore (which
reads data/redacted/ through its manifest check), the search index and the saved results.
Tools refuses a call store pointed into data/raw/, and none of them takes a path argument.
Arguments are validated before a tool runs; unknown tools and extra fields are errors the
model sees, so it can correct itself.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from callsearch.rag.bm25 import terms

Split = Literal["train", "dev", "test", "upload"]
Source = Literal["template", "multiwoz", "upload"]
RAW_DIR = Path("data/raw")
MAX_TRANSCRIPT_TURNS = 80
SNIPPET_CHARS = 300
PLACEHOLDER_TYPES = ("PERSON", "CARD", "PHONE", "EMAIL", "SIN", "ACCOUNT")


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArgs(_Args):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(default=5, ge=1, le=10)
    split: Split | None = None
    source: Source | None = None


class TranscriptArgs(_Args):
    call_id: str = Field(min_length=1, max_length=64)


class StatsArgs(_Args):
    query: str | None = Field(default=None, max_length=100)
    split: Split | None = None
    source: Source | None = None


class ReportArgs(_Args):
    pass


ARGS = {"search_calls": SearchArgs, "get_transcript": TranscriptArgs, "compute_stats": StatsArgs,
        "redaction_report": ReportArgs}
TOOL_NAMES = tuple(ARGS)


@dataclass
class ToolResult:
    text: str                       # what the model reads
    summary: str                    # one line for people and logs: counts and ids, never transcript text
    chunks: dict[str, dict] = field(default_factory=dict)  # turn id -> turn, for citations
    calls: list[str] = field(default_factory=list)          # call ids this result mentions
    ok: bool = True


def _clock(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def _matches(query_terms: list[str], text: str) -> bool:
    have = set(terms(text))
    return all(t in have for t in query_terms)


def stats(calls: list[dict], query: str | None = None) -> dict:
    """Counts over calls; with a query, turns and calls containing every query term."""
    turns_by, words_by, redactions = Counter(), Counter(), Counter()
    durations = []
    q = [t for t in terms(query)] if query else []
    match_turns, match_calls = Counter(), []
    for c in calls:
        hit = False
        for t in c["turns"]:
            turns_by[t["speaker"]] += 1
            words_by[t["speaker"]] += len(t["text"].split())
            for p in PLACEHOLDER_TYPES:
                redactions[p] += t["text"].count(f"[{p}]")
            if q and _matches(q, t["text"]):
                match_turns[t["speaker"]] += 1
                hit = True
        if hit:
            match_calls.append(c["call_id"])
        if c["turns"]:
            durations.append(float(c["turns"][-1]["start_sec"]))
    total_words = sum(words_by.values())
    out = {
        "calls": len(calls),
        "turns": sum(turns_by.values()),
        "turns_by_speaker": {s: turns_by[s] for s in ("agent", "customer")},
        "words_by_speaker": {s: words_by[s] for s in ("agent", "customer")},
        "agent_talk_share": round(words_by["agent"] / total_words, 3) if total_words else 0.0,
        "avg_call_minutes": round(sum(durations) / len(durations) / 60, 2) if durations else 0.0,
        "redactions": {p: redactions[p] for p in PLACEHOLDER_TYPES if redactions[p]},
    }
    if query is not None:
        out["matching"] = {"query": query, "query_terms": q, "calls": len(match_calls),
                           "turns_by_speaker": {s: match_turns[s] for s in ("agent", "customer")},
                           "example_calls": match_calls[:5]}
    return out


class Tools:
    def __init__(self, calls, index, results):
        """calls: CallStore, index: CallIndex (or a fake), results: ResultsReader."""
        directory = getattr(calls, "directory", None)
        if directory is not None and Path(directory).resolve().is_relative_to(RAW_DIR.resolve()):
            raise ValueError("the agent's call store must read redacted data, not data/raw/")
        self.calls = calls
        self.index = index
        self.results = results

    def _filtered(self, split: str | None, source: str | None) -> list[dict]:
        return [c for c in self.calls.all()
                if (split is None or c["split"] == split) and (source is None or c["source"] == source)]

    def run(self, name: str, args: dict) -> ToolResult:
        model = ARGS.get(name)
        if model is None:
            return ToolResult(f"error: unknown tool {name!r}. Tools: {', '.join(TOOL_NAMES)}.",
                              f"unknown tool {name!r}", ok=False)
        try:
            parsed = model.model_validate(args if isinstance(args, dict) else {})
        except ValidationError as e:
            first = e.errors()[0]
            where = ".".join(str(p) for p in first.get("loc", ())) or "args"
            return ToolResult(f"error: {name} {where}: {first['msg']}. Fix the arguments and try again.",
                              f"invalid arguments for {name}", ok=False)
        return getattr(self, name)(parsed)

    def search_calls(self, a: SearchArgs) -> ToolResult:
        hits = self.index.search(a.query, k=a.k, split=a.split, mode="bm25")
        if not hits:
            return ToolResult("No matching turns.", "0 turns")
        lines, chunks = [], {}
        for h in hits:
            chunks[h.chunk_id] = {"chunk_id": h.chunk_id, "call_id": h.call_id, "turn_id": h.turn_id,
                                  "speaker": h.speaker, "start_sec": h.start_sec, "text": h.text}
            lines.append(f"{h.chunk_id} ({h.speaker}, {h.split} split, at {_clock(h.start_sec)}): "
                         f"{h.text[:SNIPPET_CHARS]}")
        call_ids = list(dict.fromkeys(h.call_id for h in hits))
        return ToolResult("\n".join(lines), f"{len(hits)} turns from {len(call_ids)} calls", chunks, call_ids)

    def get_transcript(self, a: TranscriptArgs) -> ToolResult:
        call = next((c for c in self.calls.all() if c["call_id"] == a.call_id.strip()), None)
        if call is None:
            return ToolResult(f"error: no call with id {a.call_id!r}. Use an id from a search result.",
                              "call not found", ok=False)
        turns = call["turns"][:MAX_TRANSCRIPT_TURNS]
        head = (f"{call['call_id']}: {call['source']} call, {call['split']} split, {len(call['turns'])} turns, "
                f"last turn at {_clock(call['turns'][-1]['start_sec']) if call['turns'] else '0:00'}")
        lines, chunks = [head], {}
        for t in turns:
            cid = f"{call['call_id']}:{t['turn_id']}"
            chunks[cid] = {"chunk_id": cid, "call_id": call["call_id"], "turn_id": t["turn_id"],
                           "speaker": t["speaker"], "start_sec": t["start_sec"], "text": t["text"]}
            lines.append(f"{cid} {t['speaker']} {_clock(t['start_sec'])}: {t['text']}")
        if len(call["turns"]) > len(turns):
            lines.append(f"(first {len(turns)} of {len(call['turns'])} turns shown)")
        return ToolResult("\n".join(lines), f"{len(call['turns'])} turns", chunks, [call["call_id"]])

    def compute_stats(self, a: StatsArgs) -> ToolResult:
        calls = self._filtered(a.split, a.source)
        out = stats(calls, a.query)
        ids = out.get("matching", {}).get("example_calls", [])
        scope = ", ".join(x for x in (a.split and f"{a.split} split", a.source) if x) or "all calls"
        summary = f"{out['calls']} calls ({scope})"
        if a.query is not None:
            summary += f", {out['matching']['calls']} matching the query"
        return ToolResult(json.dumps(out), summary, calls=ids)

    def redaction_report(self, a: ReportArgs) -> ToolResult:
        r = self.results
        methods = [{"method": d.label, "name_precision": round(d.person_precision, 3),
                    "name_recall": round(d.person_recall, 3), "name_f1": round(d.person_f1, 3),
                    "leak_rate_all_types": round(d.leak_rate, 4), "other_types_recall": round(d.structured_recall, 3)}
                   for d in r.redaction()]
        g = r.guard()
        out = {"split": "test", "methods": methods,
               "index_check": None if g is None else {"passed": g.passed, "indexed_turns": g.n_chunks,
                                                       "structured_values_found": g.structured_leaks,
                                                       "names_missed": g.person_leaks,
                                                       "names_planted": g.person_support}}
        if not methods and g is None:
            return ToolResult("No saved redaction results. Run the redaction eval first.", "no results")
        return ToolResult(json.dumps(out), f"{len(methods)} methods")
