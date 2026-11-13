"""What the API reads: redacted calls, the vector index, the answerer and saved eval results.

Everything is loaded lazily and only from redacted sources: data/redacted/ (through its
manifest check), the index and results/. Nothing here can open data/raw/. Tests swap in
fakes by passing their own objects to Services.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

from callsearch.api.schemas import (CallDetail, CallSummary, DetectorScore, GuardScores, ModelCard, RagScores,
                                    TurnOut)

PLACEHOLDER = re.compile(r"\[([A-Z_]+)\]")
DETECTOR_LABELS = {  # shown on the Quality page, in this order
    "pipeline_hf": "Regex + DistilBERT",
    "pipeline": "Regex + spaCy",
    "presidio": "Microsoft Presidio",
    "regex": "Regex only",
}


class Unavailable(RuntimeError):
    """A dependency (index, answer model, data) is missing; the message says how to fix it."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class CallStore:
    """Redacted calls in memory, read once through the manifest check."""

    def __init__(self, calls: list[dict] | None = None, redacted_dir: Path = Path("data/redacted")):
        self._calls = calls
        self._dir = redacted_dir
        self._lock = threading.Lock()

    def _load(self) -> list[dict]:
        if self._calls is None:
            with self._lock:
                if self._calls is None:
                    from callsearch.redact.export import ManifestError, load_redacted

                    try:
                        calls, _ = load_redacted(self._dir)
                    except ManifestError as e:
                        raise Unavailable("calls_unavailable", f"Redacted calls are not ready: {e}") from e
                    self._calls = sorted(calls, key=lambda c: c["call_id"])
                    self._by_id = {c["call_id"]: c for c in self._calls}
        if not hasattr(self, "_by_id"):
            self._by_id = {c["call_id"]: c for c in self._calls}
        return self._calls

    def append(self, new_calls: list[dict], detector_name: str) -> None:
        """Add already-redacted calls and rewrite the redacted file and its manifest."""
        from callsearch.redact.export import ManifestError, load_redacted, write_redacted

        with self._lock:
            try:
                existing, manifest = load_redacted(self._dir)
                detector_name = manifest.get("detector", detector_name)
            except ManifestError:
                if (self._dir / "manifest.json").exists():
                    raise Unavailable("calls_unavailable", "The redacted calls file failed its check, so "
                                      "nothing was added. Re-run the redaction export.") from None
                existing = []
            calls = existing + new_calls
            write_redacted(calls, self._dir, detector_name)
            self._calls = sorted(calls, key=lambda c: c["call_id"])
            self._by_id = {c["call_id"]: c for c in self._calls}

    def all(self) -> list[dict]:
        """Every redacted call, sorted by id (read through the manifest check)."""
        return self._load()

    @property
    def directory(self) -> Path:
        return self._dir

    def count(self) -> int:
        try:
            return len(self._load())
        except Unavailable:
            return 0

    @staticmethod
    def _redactions(call: dict) -> Counter:
        return Counter(m for t in call["turns"] for m in PLACEHOLDER.findall(t["text"]))

    @staticmethod
    def _duration(call: dict) -> float:
        return float(call["turns"][-1]["start_sec"]) if call["turns"] else 0.0

    def summary(self, call: dict) -> CallSummary:
        first = next((t["text"] for t in call["turns"] if t["speaker"] == "customer"),
                     call["turns"][0]["text"] if call["turns"] else "")
        return CallSummary(call_id=call["call_id"], split=call["split"], source=call["source"],
                           n_turns=len(call["turns"]), duration_sec=self._duration(call),
                           redactions=sum(self._redactions(call).values()), preview=first[:160])

    def page(self, q: str | None, split: str | None, source: str | None, sort: str, order: str,
             page: int, page_size: int) -> tuple[list[CallSummary], int]:
        calls = self._load()
        if split:
            calls = [c for c in calls if c["split"] == split]
        if source:
            calls = [c for c in calls if c["source"] == source]
        if q:
            needle = q.lower().strip()
            calls = [c for c in calls if needle in c["call_id"].lower()
                     or any(needle in t["text"].lower() for t in c["turns"])]
        rows = [self.summary(c) for c in calls]
        key = {"call_id": lambda r: r.call_id, "n_turns": lambda r: (r.n_turns, r.call_id),
               "duration_sec": lambda r: (r.duration_sec, r.call_id),
               "redactions": lambda r: (r.redactions, r.call_id)}[sort]
        rows.sort(key=key, reverse=order == "desc")
        start = (page - 1) * page_size
        return rows[start:start + page_size], len(rows)

    def detail(self, call_id: str) -> CallDetail | None:
        self._load()
        call = self._by_id.get(call_id)
        if call is None:
            return None
        return CallDetail(call_id=call["call_id"], split=call["split"], source=call["source"],
                          duration_sec=self._duration(call),
                          turns=[TurnOut(**{k: t[k] for k in ("turn_id", "speaker", "start_sec", "text")})
                                 for t in call["turns"]],
                          redactions=dict(sorted(self._redactions(call).items())))


class ResultsReader:
    """Test-split numbers from results/, for the Quality page. Missing files are skipped."""

    def __init__(self, results_dir: Path = Path("results")):
        self.dir = results_dir

    def _read(self, name: str) -> dict | None:
        p = self.dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def redaction(self) -> list[DetectorScore]:
        out = []
        for det, label in DETECTOR_LABELS.items():
            r = self._read(f"eval_{det}_test.json")
            if r is None:
                continue
            per = r["per_type"]["PERSON"]
            out.append(DetectorScore(detector=det, label=label, person_precision=per["precision"],
                                     person_recall=per["recall"], person_f1=per["f1"],
                                     person_leak_rate=per["leak_rate"], leak_rate=r["micro"]["leak_rate"],
                                     structured_recall=r["micro_excl_person"]["recall"]))
        return out

    def rag(self) -> RagScores | None:
        r = self._read("rag_eval_test_bm25_n1.json")
        if r is None:
            return None
        answered = [q for q in r["questions"] if q["answerable"] and q.get("found")]
        right = sum(bool(q.get("citation_ok")) for q in answered)
        return RagScores(retrieval=r.get("retrieval", "dense"), neighbors=r.get("neighbors", 0),
                         model=(r.get("llm") or {}).get("model"), n_answerable=r["n_answerable"],
                         n_unanswerable=r["n_unanswerable"], recall_at_5=r["recall@5"],
                         citation_accuracy=r["citation_accuracy"], answer_term_match=r["answer_term_match"],
                         abstention_rate=r["abstention_rate"],
                         answered_correct_call=right / len(answered) if answered else 0.0)

    def guard(self) -> GuardScores | None:
        g = self._read("index_guard.json")
        if g is None:
            return None
        leaks = g["planted_leaks"]
        return GuardScores(passed=g["passed"], n_chunks=g["n_chunks"],
                           structured_leaks=sum(v for k, v in leaks.items() if k != "PERSON"),
                           person_leaks=leaks["PERSON"], person_support=g["planted_support"]["PERSON"])

    def models(self) -> list[ModelCard]:
        out = []
        for name, file in (("DistilBERT", "train_ner_distilbert.json"), ("spaCy", "train_ner_spacy.json")):
            r = self._read(file)
            if r:
                out.append(ModelCard(name=name, dev_f1=r["best_dev_f1"], size_mb=r["model_size_mb"],
                                     train_seconds=r["train_seconds"], parameters=r.get("n_parameters")))
        return out


class Services:
    def __init__(self, calls: CallStore | None = None, index=None, answerer=None,
                 results: ResultsReader | None = None, detector=None, agent=None):
        self.calls = calls or CallStore()
        self.results = results or ResultsReader()
        self._index = index
        self._answerer = answerer
        self._detector = detector
        self._agent = agent  # factory: tools -> object with .run(question), for tests
        self._ask_lock = threading.Lock()  # one answer at a time: the local model is the bottleneck anyway

    @property
    def index(self):
        if self._index is None:
            from callsearch.rag.store import CallIndex

            self._index = CallIndex()
        return self._index

    def index_info(self) -> dict | None:
        try:
            info = self.index.info
            return {"embed_model": info.get("embed_model"), "detector": info.get("detector"),
                    "n_chunks": info.get("n_chunks")}
        except FileNotFoundError:
            return None

    def require_index(self):
        if self.index_info() is None:
            raise Unavailable("index_unavailable",
                              "The search index is not built. Run: python scripts/build_index.py")
        return self.index

    def answerer(self, k: int, mode: str):
        if self._answerer is not None:
            return self._answerer(k=k, mode=mode)
        from callsearch.rag.answer import Answerer

        return Answerer(index=self.require_index(), k=k, mode=mode)

    def llm_info(self) -> dict | None:
        try:
            from callsearch.redact.llm import default_provider

            p = default_provider()
            return {"provider": p.name, "model": p.model}
        except Exception:
            return None

    def ask(self, question: str, k: int, mode: str, split: str | None):
        self.require_index()
        with self._ask_lock:
            try:
                return self.answerer(k, mode).ask(question, split)
            except Unavailable:
                raise
            except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
                raise Unavailable("answer_model_unavailable",
                                  f"The answer model could not be reached: {e}. Check that Ollama is running "
                                  "or that CALLSEARCH_LLM_PROVIDER is set correctly.") from e

    @property
    def detector(self):
        """Redaction for uploads: the same regex + DistilBERT pipeline that redacted the dataset."""
        if self._detector is None:
            from callsearch.redact.pipeline import PipelineHFDetector

            self._detector = PipelineHFDetector()
        return self._detector

    def ingest(self, calls: list[dict], stage) -> dict:
        """Redact raw calls in memory, save them with the redacted data and index them.

        calls carry call_id, split, source and turns (turn_id, speaker, start_sec, text).
        stage(name) reports progress. The raw text is never written anywhere.
        """
        from callsearch.rag.chunk import chunk_calls
        from callsearch.rag.guard import pattern_scan
        from callsearch.redact.export import redact_calls

        stage("redacting")
        try:
            redacted = redact_calls(calls, self.detector)
        except (OSError, RuntimeError, ImportError) as e:
            raise Unavailable("redaction_unavailable", "The redaction model could not be loaded. Train it "
                              "first: python scripts/train_ner_hf.py") from e
        left = pattern_scan([t["text"] for c in redacted for t in c["turns"]])
        if any(left.values()):
            raise Unavailable("redaction_check_failed", f"{sum(left.values())} card, SIN, account, email or "
                              "phone values were still found after redaction, so nothing was saved.")
        self.calls.append(redacted, getattr(self.detector, "name", "pipeline_hf"))
        counts = Counter(m for c in redacted for t in c["turns"] for m in PLACEHOLDER.findall(t["text"]))
        info = self.index_info()
        indexed = False
        if info is not None:
            stage("indexing")
            context = int(self.index.info.get("context_turns", 1))
            with self._ask_lock:  # no answer reads the index while it grows
                self.index.add(chunk_calls(redacted, context))
            indexed = True
        return {"call_ids": [c["call_id"] for c in redacted], "redactions": dict(sorted(counts.items())),
                "indexed": indexed}

    def run_agent(self, question: str):
        """The tool-using agent (callsearch.agent) over redacted calls, the index and saved results."""
        from callsearch.agent.loop import Agent
        from callsearch.agent.tools import Tools

        tools = Tools(self.calls, self.require_index(), self.results)
        agent = self._agent(tools) if self._agent is not None else Agent(tools)
        with self._ask_lock:
            try:
                return agent.run(question)
            except Unavailable:
                raise
            except (RuntimeError, OSError, ConnectionError, TimeoutError) as e:
                raise Unavailable("answer_model_unavailable",
                                  f"The answer model could not be reached: {e}. Check that Ollama is running "
                                  "or that CALLSEARCH_LLM_PROVIDER is set correctly.") from e
