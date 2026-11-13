"""Agent loop: the model picks a tool, reads the result, and repeats until it answers.

Each turn the model gets the question and every result so far, numbered [r1], [r2], ...,
and replies with one JSON object: {"tool": ..., "args": {...}} or {"answer": ..., "citations": [...]}.
This works with any provider that returns text (Ollama locally, Claude through the same
interface), and the whole run is one readable list of steps.

Rules enforced here, not by the prompt:
- at most max_steps tool calls (default 6); a malformed reply or a bad tool call uses a step
- an identical repeated call is not re-run; the model is pointed at the earlier result
- citations must be ids that appeared in a tool result this run: turn ids (call_0002:4),
  call ids (call_0002) or result ids (r3). Anything else discards the answer as not found
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from callsearch.agent.tools import TOOL_NAMES, ToolResult, Tools
from callsearch.rag.answer import NOT_FOUND, Citation
from callsearch.reply_cache import ReplyCache, reply_key

CACHE_DIR = Path(".cache/agent")
MAX_STEPS = 6
PROMPT_VERSION = "v1"

PROMPTS = {"v1": """You answer questions about an archive of call-centre transcripts by using tools.
The transcripts are redacted: placeholders like [PERSON], [CARD], [PHONE], [EMAIL], [SIN] and [ACCOUNT]
replace personal data. Never guess what a placeholder hides.

Calls come from two sources: "template" (banking and account service calls) and "multiwoz"
(travel and booking calls: trains, hotels, restaurants, taxis, attractions). Each call is in one
split: train, dev or test. Speakers are "agent" and "customer".

Tools:
- search_calls {"query": str, "k": 1-10, "split": str or null, "source": str or null}
  Keyword search over transcript turns. Returns turn ids like call_0002:4 with the turn text.
- get_transcript {"call_id": str}
  The whole redacted call, turn by turn. Use it when you need more than the matching turn.
- compute_stats {"query": str or null, "split": str or null, "source": str or null}
  Counts of calls, turns and words by speaker, agent talk share and average call length.
  With a query: how many calls and how many turns per speaker contain all of its words,
  plus example call ids. Use it for "how many", "how often", "which speaker" questions.
- redaction_report {}
  Precision, recall and F1 for names, and leak rates, of each redaction method on the test
  split, plus the search index check.

Reply with exactly one JSON object and nothing else:
{"tool": "<tool name>", "args": {...}}                to use a tool, or
{"answer": "<answer>", "citations": ["<id>", ...]}    when the results answer the question.

Citations are ids shown in the results that support the answer: turn ids (call_0002:4),
call ids (call_0002), or result ids for statistics and reports (r2). Cite at least one.
Use digits for numbers. Answer in one to three sentences.
If the question asks for several things, get all of them before answering.
If the results cannot answer the question, reply {"answer": "NOT_FOUND", "citations": []}."""}

@dataclass
class Step:
    n: int
    tool: str | None
    args: dict
    summary: str
    ok: bool
    ms: int
    result: ToolResult | None = None


@dataclass
class AgentAnswer:
    question: str
    answer: str
    found: bool
    citations: list[Citation]          # cited turns
    cited_calls: list[str]             # every call the answer cites, directly or through a turn
    cited_results: list[int]           # step numbers of cited statistics or reports
    steps: list[Step]
    rejected: str | None = None
    provenance: dict = field(default_factory=dict)
    llm_calls: int = 0

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "found": self.found,
                "citations": [c.__dict__ for c in self.citations], "cited_calls": self.cited_calls,
                "cited_results": self.cited_results, "rejected": self.rejected, "provenance": self.provenance,
                "llm_calls": self.llm_calls,
                "steps": [{"n": s.n, "tool": s.tool, "args": s.args, "summary": s.summary, "ok": s.ok, "ms": s.ms}
                          for s in self.steps]}


def first_json(raw: str) -> dict | None:
    s = re.sub(r"```(?:json)?", "", raw)
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def build_message(question: str, steps: list[Step], left: int) -> str:
    lines = [f"Question: {question}", ""]
    if steps:
        lines.append("Results so far:")
        for s in steps:
            call = f"{s.tool} {json.dumps(s.args, sort_keys=True)}" if s.tool else "(no tool)"
            lines += [f"[r{s.n}] {call}", s.result.text if s.result else s.summary, ""]
    else:
        lines += ["No results yet.", ""]
    lines.append(f"Tool calls left: {left}." + (" You must answer now." if left == 0 else ""))
    return "\n".join(lines)


class Agent:
    def __init__(self, tools: Tools, provider=None, max_steps: int = MAX_STEPS, prompt_version: str = PROMPT_VERSION,
                 cache_dir: Path | None = CACHE_DIR):
        self.tools = tools
        self._provider = provider
        self.max_steps = max_steps
        self.prompt_version = prompt_version
        self.cache_dir = cache_dir
        self.requests = self.cache_hits = 0

    @property
    def provider(self):
        if self._provider is None:
            from callsearch.redact.llm import default_provider

            self._provider = default_provider()
        return self._provider

    def provenance(self) -> dict:
        return {"provider": self.provider.name, "model": self.provider.model, "prompt_version": self.prompt_version,
                "max_steps": self.max_steps, "retrieval": "bm25"}

    def _complete(self, user: str) -> str:
        p = self.provider
        key = reply_key(p.name, p.model, f"agent-{self.prompt_version}", user)
        cache = ReplyCache(self.cache_dir)
        cached = cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.requests += 1
        raw = p.complete(PROMPTS[self.prompt_version], user)
        cache.put(key, raw)
        return raw

    def _finish(self, question, obj, steps, llm_calls) -> AgentAnswer:
        prov = self.provenance()
        answer = obj.get("answer")
        answer = answer.strip() if isinstance(answer, str) else ""
        if not answer or answer.upper().startswith(NOT_FOUND):
            return AgentAnswer(question, NOT_FOUND, False, [], [], [], steps, None, prov, llm_calls)
        refs = obj.get("citations")
        if not isinstance(refs, list) or not refs:
            return AgentAnswer(question, NOT_FOUND, False, [], [], [], steps, "answer had no citations", prov, llm_calls)
        chunks = {cid: c for s in steps if s.result for cid, c in s.result.chunks.items()}
        calls_seen = {cid for s in steps if s.result for cid in s.result.calls} | {c["call_id"] for c in chunks.values()}
        result_steps = {s.n for s in steps if s.result is not None and s.ok}
        cited_chunks, cited_calls, cited_results = [], [], []
        for r in refs:
            ref = str(r).strip().strip("[]") if isinstance(r, (str, int)) and not isinstance(r, bool) else ""
            if ref in chunks:
                cited_chunks.append(ref)
            elif ref in calls_seen:
                cited_calls.append(ref)
            elif re.fullmatch(r"r\d+", ref) and int(ref[1:]) in result_steps:
                cited_results.append(int(ref[1:]))
            else:
                why = "was not in any tool result" if ref else "is not an id"
                return AgentAnswer(question, NOT_FOUND, False, [], [], [], steps,
                                   f"cited {r!r}, which {why}", prov, llm_calls)
        cited_chunks = list(dict.fromkeys(cited_chunks))
        citations = [Citation(c["chunk_id"], c["call_id"], c["turn_id"], c["speaker"], c["start_sec"], c["text"])
                     for c in (chunks[i] for i in cited_chunks)]
        all_calls = list(dict.fromkeys([c.call_id for c in citations] + cited_calls))
        return AgentAnswer(question, answer, True, citations, all_calls, list(dict.fromkeys(cited_results)), steps,
                           None, prov, llm_calls)

    def run(self, question: str) -> AgentAnswer:
        steps: list[Step] = []
        llm_calls = 0
        while True:
            left = self.max_steps - len(steps)
            raw = self._complete(build_message(question, steps, left))
            llm_calls += 1
            obj = first_json(raw)
            if obj is not None and "answer" in obj:
                return self._finish(question, obj, steps, llm_calls)
            if left <= 0:
                return AgentAnswer(question, NOT_FOUND, False, [], [], [], steps,
                                   f"step cap of {self.max_steps} tool calls reached", self.provenance(), llm_calls)
            n = len(steps) + 1
            if obj is None or not isinstance(obj.get("tool"), str):
                note = ('error: reply with one JSON object, {"tool": ..., "args": {...}} or '
                        '{"answer": ..., "citations": [...]}.')
                steps.append(Step(n, None, {}, "unreadable reply", False, 0, ToolResult(note, "unreadable reply", ok=False)))
                continue
            name, args = obj["tool"].strip(), obj.get("args") if isinstance(obj.get("args"), dict) else {}
            earlier = next((s for s in steps if s.tool == name and s.args == args and s.ok), None)
            if earlier is not None:
                note = f"You already ran this call; its result is [r{earlier.n}]. Use it, try something else, or answer."
                steps.append(Step(n, name, args, f"repeat of r{earlier.n}", False, 0, ToolResult(note, "repeat", ok=False)))
                continue
            t0 = time.perf_counter()
            result = self.tools.run(name, args)
            steps.append(Step(n, name if name in TOOL_NAMES else name[:40], args, result.summary, result.ok,
                              round((time.perf_counter() - t0) * 1000), result))
