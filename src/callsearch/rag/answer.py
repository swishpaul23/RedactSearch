"""Phase 3: Answer questions from retrieved chunks with cited call snippets.

python -m callsearch.rag.answer "Why did the customer want a refund?" -k 5 [--split test]

1. Retrieve the top k redacted chunks from the index.
2. Show them to the LLM as numbered excerpts. It must answer only from them, cite the
   excerpt numbers it used, and reply NOT_FOUND when they do not support an answer.
3. Server check: the reply must parse, cite at least one excerpt, and cite only excerpts
   that were retrieved. Anything else is discarded and returned as not found.

Only redacted text from the index ever goes into a prompt. The provider comes from
callsearch.redact.llm (CALLSEARCH_LLM_PROVIDER = anthropic | ollama). Replies are cached in
.cache/answers/ keyed by provider, model, prompt version, question and retrieved chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NOT_FOUND = "NOT_FOUND"
CACHE_DIR = Path(".cache/answers")

# Never edit a released version; add a new one. The version is part of the cache key.
PROMPTS: dict[str, str] = {}
PROMPTS["v1"] = """You answer questions about customer-service calls using only the numbered excerpts you are given.

Rules:
- Use only facts stated in the excerpts. No outside knowledge, no guessing.
- Cite the number of every excerpt you used.
- If the excerpts do not answer the question, reply NOT_FOUND.
- Placeholders such as [PERSON], [CARD], [PHONE], [EMAIL], [SIN] and [ACCOUNT] stand for personal data
  that was removed. Never guess what they hide, and do not count them as an answer.

Reply with JSON only, no other text:
{"answer": "<one or two sentences>", "citations": [<excerpt numbers>]}
If the excerpts do not answer the question, reply exactly:
{"answer": "NOT_FOUND", "citations": []}"""
PROMPT_VERSION = "v1"


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    call_id: str
    turn_id: int
    speaker: str
    start_sec: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str                          # NOT_FOUND unless found
    found: bool
    citations: list[Citation]
    retrieved: list[str]                 # chunk ids shown to the model, in rank order
    rejected: str | None = None          # why a model reply was discarded, if it was
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "found": self.found,
                "citations": [c.__dict__ for c in self.citations], "retrieved": self.retrieved,
                "rejected": self.rejected, "provenance": self.provenance}


def build_user_message(question: str, hits) -> str:
    lines = [f"Question: {question}", "", "Excerpts:"]
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] call {h.call_id}, turn {h.turn_id}, {h.speaker} at {h.start_sec:.0f}s: {h.text}")
    return "\n".join(lines)


def parse_reply(raw: str) -> dict:
    """First JSON object with an "answer" key, tolerating code fences and text around it."""
    s = re.sub(r"```(?:json)?", "", raw)
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            return obj
    raise ValueError("no {\"answer\": ...} object in the reply")


def _resolve(ref, hits) -> int | None:
    """Excerpt number (1-based int or digit string) or chunk id -> index into hits."""
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().lstrip("[").rstrip("]").isdigit()):
        n = int(str(ref).strip().lstrip("[").rstrip("]"))
        return n - 1 if 1 <= n <= len(hits) else None
    if isinstance(ref, str):
        ids = [h.chunk_id for h in hits]
        return ids.index(ref.strip()) if ref.strip() in ids else None
    return None


def check_reply(raw: str, hits) -> tuple[str, list[int], str | None]:
    """Server-side validation. Returns (answer, cited hit indexes, rejection reason)."""
    try:
        obj = parse_reply(raw)
    except ValueError:
        return NOT_FOUND, [], "unparseable reply"
    answer = obj["answer"].strip()
    if not answer or answer.upper().startswith(NOT_FOUND):
        return NOT_FOUND, [], None  # the model said it cannot answer: a valid outcome
    refs = obj.get("citations")
    if not isinstance(refs, list) or not refs:
        return NOT_FOUND, [], "answer had no citations"
    cited: list[int] = []
    for r in refs:
        i = _resolve(r, hits)
        if i is None:
            return NOT_FOUND, [], f"cited {r!r}, which was not retrieved"
        if i not in cited:
            cited.append(i)
    return answer, cited, None


class Answerer:
    def __init__(self, index=None, provider=None, k: int = 5, prompt_version: str = PROMPT_VERSION,
                 cache_dir: Path | None = CACHE_DIR):
        if prompt_version not in PROMPTS:
            raise KeyError(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
        self._index = index
        self._provider = provider
        self.k = k
        self.prompt_version = prompt_version
        self.cache_dir = cache_dir
        self.requests = 0
        self.cache_hits = 0

    @property
    def index(self):
        if self._index is None:
            from callsearch.rag.store import CallIndex

            self._index = CallIndex()
        return self._index

    @property
    def provider(self):
        if self._provider is None:
            from callsearch.redact.llm import default_provider

            self._provider = default_provider()
        return self._provider

    def provenance(self) -> dict:
        return {"provider": self.provider.name, "model": self.provider.model, "prompt_version": self.prompt_version,
                "k": self.k}

    def _complete(self, user: str) -> str:
        p = self.provider
        key = hashlib.sha256(f"{p.name}\0{p.model}\0{self.prompt_version}\0{user}".encode("utf-8")).hexdigest()
        path = None if self.cache_dir is None else self.cache_dir / key[:2] / f"{key}.json"
        if path is not None and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        self.requests += 1
        raw = p.complete(PROMPTS[self.prompt_version], user)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"response": raw}), encoding="utf-8")
        return raw

    def ask(self, question: str, split: str | None = None) -> Answer:
        hits = self.index.search(question, k=self.k, split=split)
        retrieved = [h.chunk_id for h in hits]
        if not hits:
            return Answer(question, NOT_FOUND, False, [], retrieved, "nothing retrieved", self.provenance())
        answer, cited, rejected = check_reply(self._complete(build_user_message(question, hits)), hits)
        citations = [Citation(hits[i].chunk_id, hits[i].call_id, hits[i].turn_id, hits[i].speaker,
                              hits[i].start_sec, hits[i].text) for i in cited]
        return Answer(question, answer, answer != NOT_FOUND, citations, retrieved, rejected, self.provenance())


def main() -> None:
    p = argparse.ArgumentParser(description="Answer a question from the call index, with citations.")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--split", choices=["train", "dev", "test"])
    p.add_argument("--prompt", default=PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--json", action="store_true", help="print the full answer record as JSON")
    args = p.parse_args()

    a = Answerer(k=args.k, prompt_version=args.prompt).ask(args.question, args.split)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
        return
    print(a.answer)
    if a.rejected:
        print(f"(model reply discarded: {a.rejected})")
    for c in a.citations:
        print(f"  [{c.chunk_id} {c.start_sec:.0f}s {c.speaker}] {c.snippet}")
    pv = a.provenance
    print(f"\n{pv['provider']}:{pv['model']} prompt {pv['prompt_version']}, retrieved {len(a.retrieved)}")


if __name__ == "__main__":
    main()
