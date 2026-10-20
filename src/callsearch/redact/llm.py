"""Phase 2, Layer 3: LLM catch-all for personal data the regex and NER layers missed.

Add-only (recall-first): the LLM sees each turn ALREADY MASKED by the base layers
("[CARD]", "[PERSON]", ...) and lists any remaining personal data. Its spans are added;
base spans are never removed. Offsets come from our own exact-substring search, not
from the model.

Providers: AnthropicProvider (ANTHROPIC_API_KEY, model CALLSEARCH_LLM_MODEL) and
OllamaProvider (local, CALLSEARCH_OLLAMA_MODEL). Pick with CALLSEARCH_LLM_PROVIDER.
Responses are cached in .cache/llm/ (gitignored) keyed by provider, model, prompt
version and masked text, so re-running an eval is free and reproducible.

python -m callsearch.redact.llm --split dev
    Splits LLM-added spans into correct vs false additions (uses the cache).
    Saves results/llm_additions_<split>.json (metrics only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

from callsearch.redact.base import Detector, Span
from callsearch.redact.merge import _rank, resolve_overlaps

ALLOWED_TYPES = ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]  # anything else is a bad type
# Post-filter: only these LLM types are kept. Regex is exact on the structured types, and on dev
# every structured span the LLM added was a false positive (results/llm_additions_dev.json).
KEEP_TYPES = ["PERSON"]
CACHE_DIR = Path(".cache/llm")
EST_OUTPUT_TOKENS = 40  # dry-run guess per request; most turns have nothing left to find

# Old versions stay here so their cached replies can still be re-scored (--llm-prompt v1).
# Never edit a released version; add a new one. The version is part of the cache key.
PROMPTS: dict[str, str] = {}

PROMPTS["v1"] = """You find personal data left in a call-centre transcript turn.

Some personal data has already been replaced with placeholders like [PERSON], [CARD], [SIN],
[PHONE], [EMAIL], [ACCOUNT]. Ignore placeholders. Report only personal data that is still
visible as plain text.

Allowed types:
- PERSON: a real individual's name (first name, last name, or full name), including the agent's name.
- EMAIL: an email address.
- PHONE: a phone number.
- CARD: a payment card number.
- SIN: a Canadian social insurance number.
- ACCOUNT: a bank or customer account number.

NOT personal data. Never report these:
- Business, restaurant, hotel, shop, bank and product names, even if they contain a person's name.
- Street names, place names, cities, landmarks, train stations.
- Order numbers, reference numbers, amounts, dates, times.

Reply with JSON only, no other text:
{"spans": [{"text": "<exact substring as it appears>", "type": "<one allowed type>"}]}
Copy "text" exactly as it appears in the turn, same spelling and case. If nothing remains,
reply {"spans": []}."""

PROMPTS["v2"] = """You find people's names left in a call-centre transcript turn.

Some personal data has already been replaced with placeholders like [PERSON], [CARD], [SIN],
[PHONE], [EMAIL], [ACCOUNT]. Ignore placeholders. Report only names still visible as plain text.

Report only one type:
- PERSON: a real individual's name (first name, last name, or full name), including the agent's name.

NOT a person. Never report these:
- Business, restaurant, hotel, shop, bank and product names, even if they contain a person's name.
- Street names, place names, cities, landmarks, train stations.
- Numbers of any kind, email addresses, amounts, dates, times.

Reply with JSON only, no other text:
{"spans": [{"text": "<exact name as it appears>", "type": "PERSON"}]}
Copy "text" exactly as it appears in the turn, same spelling and case.
If no names remain, reply exactly {"spans": []}."""

PROMPT_VERSION = "v2"
SYSTEM_PROMPT = PROMPTS[PROMPT_VERSION]


# --- providers ---------------------------------------------------------------------

class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


class _Usage:
    """Thread-safe token counters shared by providers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, inp: int, out: int) -> None:
        with self.lock:
            self.input_tokens += inp
            self.output_tokens += out


class AnthropicProvider:
    """Claude via the Anthropic SDK. The key is read from ANTHROPIC_API_KEY by the SDK;
    it is never logged or stored here."""

    name = "anthropic"
    default_workers = 8

    def __init__(self, model: str | None = None, max_tokens: int = 512, client=None):
        self.model = model or os.environ.get("CALLSEARCH_LLM_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens
        self.usage = _Usage()
        self._client = client
        # USD per million tokens, for the cost estimate only. Override with env vars if pricing differs.
        self.price_in = float(os.environ.get("CALLSEARCH_LLM_PRICE_IN", "1.0"))
        self.price_out = float(os.environ.get("CALLSEARCH_LLM_PRICE_OUT", "5.0"))

    @property
    def client(self):
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set (set it in the environment; never commit it)")
            import anthropic

            self._client = anthropic.Anthropic(max_retries=5)
        return self._client

    def complete(self, system: str, user: str) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, temperature=0, system=system,
            messages=[{"role": "user", "content": user}])
        self.usage.add(msg.usage.input_tokens, msg.usage.output_tokens)
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")


class OllamaProvider:
    """Local model through Ollama's HTTP API. Nothing leaves the machine."""

    name = "ollama"
    default_workers = 1  # a local model serves one request at a time; more threads only queue
    price_in = price_out = 0.0

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self.model = model or os.environ.get("CALLSEARCH_OLLAMA_MODEL", "")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.usage = _Usage()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, system: str, user: str) -> str:
        if not self.model:
            raise RuntimeError("set CALLSEARCH_OLLAMA_MODEL to a pulled model (see `ollama list`)")
        data = self._post({
            "model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        self.usage.add(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]


def default_provider() -> LLMProvider:
    kind = os.environ.get("CALLSEARCH_LLM_PROVIDER", "anthropic")
    if kind == "anthropic":
        return AnthropicProvider()
    if kind == "ollama":
        return OllamaProvider()
    raise ValueError(f"unknown CALLSEARCH_LLM_PROVIDER {kind!r} (anthropic | ollama)")


# --- masking, parsing, offsets ---------------------------------------------------------

def mask(text: str, spans: list[Span]) -> tuple[str, list[tuple[int, int, int, int]]]:
    """Same output as merge.redact(), plus where each placeholder sits.

    Returns (masked_text, [(masked_start, masked_end, orig_start, orig_end), ...]).
    """
    merged: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if merged and s.start < merged[-1].end:
            prev = merged[-1]
            winner = prev if _rank(prev) <= _rank(s) else s
            merged[-1] = Span(prev.start, max(prev.end, s.end), winner.type, winner.source)
        else:
            merged.append(s)
    out, places, pos = [], [], 0
    for s in merged:
        out.append(text[pos:s.start])
        m_start = sum(map(len, out))
        ph = f"[{s.type}]"
        out.append(ph)
        places.append((m_start, m_start + len(ph), s.start, s.end))
        pos = s.end
    out.append(text[pos:])
    return "".join(out), places


def to_original(m_pos: int, places: list[tuple[int, int, int, int]]) -> int:
    """Map a masked-text offset (outside any placeholder) back to the original text."""
    shift = 0
    for ms, me, os_, oe in places:
        if me <= m_pos:
            shift += (oe - os_) - (me - ms)
    return m_pos + shift


class ParseError(ValueError):
    pass


def parse_response(raw: str) -> list[dict]:
    """Pull {"spans": [...]} out of a reply, tolerating code fences and text around it.

    {} and {"spans": []} are valid empty answers (some local models reply {} for "nothing").
    Anything else without a spans list raises ParseError.
    """
    s = re.sub(r"```(?:json)?", "", raw)
    dec = json.JSONDecoder()
    saw_empty = False
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("spans"), list):
            return obj["spans"]
        saw_empty = saw_empty or obj == {}
    if saw_empty:
        return []
    raise ParseError("no {\"spans\": [...]} object found")


def _word_edge_ok(text: str, start: int, end: int) -> bool:
    """Reject matches that start or end in the middle of a word ("Ann" inside "Annual")."""
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (text[start].isalnum() and before.isalnum()) and not (text[end - 1].isalnum() and after.isalnum())


def locate(items: list, masked: str, places, stats: Counter | None = None) -> list[tuple[int, int, str]]:
    """Turn model items into masked-text offsets: every verbatim occurrence, allowed types
    only, never overlapping a placeholder. Discards are counted in `stats`."""
    stats = stats if stats is not None else Counter()
    found = []
    for it in items:
        if not (isinstance(it, dict) and isinstance(it.get("text"), str) and isinstance(it.get("type"), str)):
            stats["discard_bad_item"] += 1
            continue
        sub, typ = it["text"], it["type"].strip().upper()
        if typ not in ALLOWED_TYPES:
            stats["discard_bad_type"] += 1
            continue
        if not sub.strip():
            stats["discard_bad_item"] += 1
            continue
        hits = [m.start() for m in re.finditer(re.escape(sub), masked)]
        if not hits:
            stats["discard_not_found"] += 1
            continue
        kept = 0
        for st in hits:
            en = st + len(sub)
            if any(st < me and ms < en for ms, me, _, _ in places):
                stats["discard_in_placeholder"] += 1
            elif not _word_edge_ok(masked, st, en):
                stats["discard_partial_word"] += 1
            else:
                found.append((st, en, typ))
                kept += 1
        stats["kept"] += kept
    return found


# --- detectors -------------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


class LLMDetector:
    """LLM layer. With base=None (the "llm" ablation) the model sees the turn unmasked."""

    def __init__(self, base: Detector | None = None, provider: LLMProvider | None = None,
                 name: str = "llm", cache_dir: Path | None = CACHE_DIR, workers: int | None = None,
                 max_requests: int | None = None, prompt_version: str = PROMPT_VERSION,
                 keep_types: list[str] | None = None, progress_every: int = 100):
        if prompt_version not in PROMPTS:
            raise KeyError(f"unknown prompt version {prompt_version!r}; known: {sorted(PROMPTS)}")
        self.name = name
        self.base = base
        self._provider = provider
        self.cache_dir = cache_dir
        self._workers = workers  # None -> the provider's default (ollama 1, anthropic 8)
        self.max_requests = max_requests
        self.prompt_version = prompt_version
        self.system_prompt = PROMPTS[prompt_version]
        self.keep_types = KEEP_TYPES if keep_types is None else keep_types
        self.progress_every = progress_every
        # requests, cache_hits, empty_answers, parse_failures, skipped_empty, discards...
        self.stats = Counter()
        self._lock = threading.Lock()
        self._base_memo: dict[str, list[Span]] = {}
        self._t_last = None

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = default_provider()
        return self._provider

    @property
    def workers(self) -> int:
        return self._workers or getattr(self.provider, "default_workers", 1)

    @workers.setter
    def workers(self, n: int | None) -> None:
        self._workers = n

    def provenance(self) -> dict:
        """Which model and prompt produced these results. Saved with every eval."""
        return {"provider": self.provider.name, "model": self.provider.model, "prompt_version": self.prompt_version}

    # --- base layers and masking
    def base_spans(self, text: str) -> list[Span]:
        if self.base is None:
            return []
        if text not in self._base_memo:
            self._base_memo[text] = self.base.detect(text)
        return self._base_memo[text]

    def masked(self, text: str):
        return mask(text, self.base_spans(text))

    @staticmethod
    def _needs_llm(masked: str) -> bool:
        return bool(re.search(r"[A-Za-z0-9]", re.sub(r"\[[A-Z]+\]", "", masked)))

    # --- cache
    def _key(self, masked: str) -> str:
        p = self.provider
        key = f"{p.name}\0{p.model}\0{self.prompt_version}\0{masked}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, masked: str) -> str | None:
        path = self._cache_path(self._key(masked))
        if path is not None and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        return None

    def _call(self, masked: str) -> str:
        with self._lock:
            if self.max_requests is not None and self.stats["requests"] >= self.max_requests:
                raise BudgetExceeded(f"--max-requests {self.max_requests} reached")
            self.stats["requests"] += 1
        raw = self.provider.complete(self.system_prompt, masked)
        path = self._cache_path(self._key(masked))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"response": raw}), encoding="utf-8")
        with self._lock:
            self.stats["completed"] += 1
            n = self.stats["completed"]
            if self.progress_every and n % self.progress_every == 0:
                now = time.perf_counter()
                print(f"  {n} requests done, last {self.progress_every} took {now - self._t_last:.1f}s", flush=True)
                self._t_last = now
        return raw

    def response(self, masked: str) -> str:
        raw = self._cached(masked)
        if raw is not None:
            with self._lock:
                self.stats["cache_hits"] += 1
            return raw
        return self._call(masked)

    # --- batch helpers for the eval CLI
    def plan(self, texts: list[str]) -> dict:
        """What a run would cost, without calling the model."""
        todo, hits, skipped, chars = set(), 0, 0, 0
        for t in texts:
            m, _ = self.masked(t)
            if not self._needs_llm(m):
                skipped += 1
            elif self._cached(m) is not None:
                hits += 1
            elif m not in todo:
                todo.add(m)
                chars += len(self.system_prompt) + len(m)
        p = self.provider
        est_in, est_out = chars // 4, EST_OUTPUT_TOKENS * len(todo)
        return {"turns": len(texts), "requests": len(todo), "cache_hits": hits, "skipped_empty": skipped,
                "est_input_tokens": est_in, "est_output_tokens": est_out,
                "est_cost_usd": (est_in * p.price_in + est_out * p.price_out) / 1e6,
                "workers": self.workers, **self.provenance()}

    def prefetch(self, texts: list[str]) -> None:
        """Fill the cache with a thread pool, so detect() runs from cache."""
        todo = []
        for t in texts:
            m, _ = self.masked(t)
            if self._needs_llm(m) and self._cached(m) is None and m not in todo:
                todo.append(m)
        if self.max_requests is not None and len(todo) > self.max_requests:
            raise BudgetExceeded(f"{len(todo)} requests needed, --max-requests is {self.max_requests}")
        self._t_last = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self._call, todo))

    # --- Detector interface
    def llm_spans(self, text: str) -> list[Span]:
        m, places = self.masked(text)
        if not self._needs_llm(m):
            with self._lock:
                self.stats["skipped_empty"] += 1
            return []
        raw = self.response(m)
        try:
            items = parse_response(raw)
        except ParseError:
            with self._lock:
                self.stats["parse_failures"] += 1
            return []
        local = Counter(empty_answers=int(not items))
        found = locate(items, m, places, local)
        kept = [f for f in found if f[2] in self.keep_types]
        local["discard_filtered_type"] += len(found) - len(kept)
        with self._lock:
            self.stats.update(local)
        return [Span(to_original(s, places), to_original(e, places), t, "llm") for s, e, t in kept]

    def detect(self, text: str) -> list[Span]:
        base = self.base_spans(text)
        # LLM spans never overlap a placeholder, so they cannot displace a base span;
        # the filter makes that explicit. Overlaps among LLM spans go to resolve_overlaps.
        added = [s for s in self.llm_spans(text) if not any(s.overlaps(b) for b in base)]
        return sorted(base + resolve_overlaps(added), key=lambda s: s.start)

    def usage_report(self) -> dict:
        p = self.provider
        u = getattr(p, "usage", None)
        inp, out = (u.input_tokens, u.output_tokens) if u else (0, 0)
        return {**self.provenance(), "requests": self.stats["requests"],
                "cache_hits": self.stats["cache_hits"], "empty_answers": self.stats["empty_answers"],
                "parse_failures": self.stats["parse_failures"],
                "discard_filtered_type": self.stats["discard_filtered_type"],
                "input_tokens": inp, "output_tokens": out,
                "est_cost_usd": (inp * getattr(p, "price_in", 0) + out * getattr(p, "price_out", 0)) / 1e6}


class PipelineLLMDetector(LLMDetector):
    """regex + ner, then the LLM on the masked text (add-only)."""

    def __init__(self, base: Detector | None = None, name: str = "pipeline_llm", **kwargs):
        if base is None:
            from callsearch.redact.pipeline import PipelineDetector

            base = PipelineDetector()
        super().__init__(base=base, name=name, **kwargs)


# --- addition analysis -------------------------------------------------------------------

def addition_stats(calls: list[dict], det: LLMDetector) -> dict:
    """Classify each LLM-added span:
    correct         matches a same-type label the base layers missed (a new true positive)
    overlaps_label  touches a label but adds no TP: extends a label the base layers already
                    matched (e.g. the other half of a name), or a label of another type
    distractor      overlaps a distractor
    false           overlaps nothing labeled"""
    from callsearch.eval.redaction_eval import match_spans

    out = Counter()
    by_type = {t: Counter() for t in ALLOWED_TYPES}
    false_samples = []  # (where, snippet) for false/wrong-type PERSON additions; printed, never saved
    for call in calls:
        for turn in call["turns"]:
            tid = turn["turn_id"]
            golds = [l for l in call["labels"] if l["turn_id"] == tid]
            dists = [d for d in call.get("distractors", []) if d["turn_id"] == tid]
            spans = det.detect(turn["text"])
            base_matched, _, _ = match_spans([s for s in spans if s.source != "llm"], golds)
            used = [g for _, g in base_matched]
            free = [g for g in golds if g not in used]
            added = [s for s in spans if s.source == "llm"]
            matched, rest, _ = match_spans(added, free)
            for s, _ in matched:
                out["correct"] += 1
                by_type[s.type]["correct"] += 1
            for s in rest:
                ov = lambda x: s.start < x["end"] and x["start"] < s.end
                kind = ("overlaps_label" if any(ov(g) for g in golds)
                        else "distractor" if any(ov(d) for d in dists) else "false")
                out[kind] += 1
                by_type[s.type][kind] += 1
                if s.type == "PERSON":
                    t = turn["text"]
                    snip = f"{t[max(0, s.start - 30):s.start]}[[{t[s.start:s.end]}]]{t[s.end:s.end + 30]}"
                    false_samples.append((f"{call['call_id']} t{tid} ({call['source']}) {kind}", snip))
    kinds = ["correct", "overlaps_label", "distractor", "false"]
    return {"detector": det.name, "added": sum(out.values()), **{k: out[k] for k in kinds},
            "by_type": {t: {k: c[k] for k in kinds} for t, c in by_type.items()},
            "false_samples": false_samples}


def main() -> None:
    from callsearch.data.generator import read_jsonl

    p = argparse.ArgumentParser(description="Split LLM-added spans into correct vs false additions.")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--max-requests", type=int, default=0,
                   help="API calls allowed (default 0: cache only; run the eval first)")
    p.add_argument("--llm-prompt", default=PROMPT_VERSION, choices=sorted(PROMPTS))
    p.add_argument("--show-false", type=int, default=0, metavar="N",
                   help="also print N false PERSON additions with context (synthetic text; not saved)")
    args = p.parse_args()

    calls = [c for c in read_jsonl(args.data) if args.split == "all" or c["split"] == args.split]
    det = PipelineLLMDetector(max_requests=args.max_requests, prompt_version=args.llm_prompt)
    stats = addition_stats(calls, det)
    samples = stats.pop("false_samples")
    res = {"split": args.split, **stats, **{
        k: v for k, v in det.usage_report().items() if k in ("provider", "model", "prompt_version", "requests",
                                                               "cache_hits", "parse_failures")}}
    print(json.dumps(res, indent=2))
    if args.show_false:
        print(f"\nFALSE PERSON ADDITIONS: {len(samples)} total, showing {min(len(samples), args.show_false)}")
        for where, snip in samples[:args.show_false]:
            print(f"  {where:<34} {snip}")
    out = Path("results") / f"llm_additions_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
