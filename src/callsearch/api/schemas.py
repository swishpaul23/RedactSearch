"""Request and response models for the HTTP API.

These models are the contract with the web app: scripts/export_openapi.py writes their
OpenAPI schema to web/openapi.json, the frontend generates its TypeScript types from it,
and a test fails if the committed schema drifts from the code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Split = Literal["train", "dev", "test", "upload"]
Source = Literal["template", "multiwoz", "upload"]
UploadStatus = Literal["queued", "redacting", "indexing", "done", "failed"]
Mode = Literal["dense", "bm25", "hybrid"]
Speaker = Literal["agent", "customer"]


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable code, e.g. rate_limited, index_unavailable")
    message: str = Field(description="What went wrong and how to fix it, for people")


class ErrorOut(BaseModel):
    error: ErrorBody


class IndexInfo(BaseModel):
    available: bool
    embed_model: str | None = None
    detector: str | None = None
    n_chunks: int | None = None


class LLMInfo(BaseModel):
    provider: str
    model: str


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    calls: int = Field(description="Redacted calls available to browse")
    index: IndexInfo
    llm: LLMInfo | None = Field(description="Answer model, or null when none is configured")


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=10, description="Excerpts given to the answer model")
    mode: Mode = "bm25"
    split: Split | None = Field(default=None, description="Restrict retrieval to one data split")


class CitationOut(BaseModel):
    chunk_id: str
    call_id: str
    turn_id: int
    speaker: Speaker
    start_sec: float
    snippet: str = Field(description="The cited turn, redacted")


class Provenance(BaseModel):
    provider: str
    model: str
    prompt_version: str
    k: int
    retrieval: Mode
    neighbors: int
    grounding: bool


class AnswerOut(BaseModel):
    question: str
    answer: str = Field(description="The answer, or NOT_FOUND")
    found: bool
    rejected: str | None = Field(description="Why a model reply was discarded, if it was")
    citations: list[CitationOut]
    retrieved: list[str] = Field(description="Matched chunk ids in rank order")
    provenance: Provenance
    elapsed_ms: int


class HitOut(BaseModel):
    chunk_id: str
    call_id: str
    turn_id: int
    speaker: Speaker
    start_sec: float
    split: Split
    text: str
    score: float


class SearchOut(BaseModel):
    query: str
    mode: Mode
    hits: list[HitOut]


class TurnOut(BaseModel):
    turn_id: int
    speaker: Speaker
    start_sec: float
    text: str


class CallSummary(BaseModel):
    call_id: str
    split: Split
    source: Source
    n_turns: int
    duration_sec: float = Field(description="Start time of the last turn")
    redactions: int = Field(description="Placeholders in the transcript")
    preview: str = Field(description="Opening customer turn, redacted")


class CallPage(BaseModel):
    items: list[CallSummary]
    total: int
    page: int
    page_size: int


class CallDetail(BaseModel):
    call_id: str
    split: Split
    source: Source
    duration_sec: float
    turns: list[TurnOut]
    redactions: dict[str, int] = Field(description="Placeholder counts by type")


class DetectorScore(BaseModel):
    detector: str
    label: str
    person_precision: float
    person_recall: float
    person_f1: float
    person_leak_rate: float
    leak_rate: float = Field(description="Share of all labeled values left verbatim")
    structured_recall: float = Field(description="Recall over email, phone, card, SIN and account")


class RagScores(BaseModel):
    retrieval: Mode
    neighbors: int
    model: str | None
    n_answerable: int
    n_unanswerable: int
    recall_at_5: float
    citation_accuracy: float
    answer_term_match: float
    abstention_rate: float
    answered_correct_call: float = Field(description="Of the answers given, share citing the right call")


class GuardScores(BaseModel):
    passed: bool
    n_chunks: int
    structured_leaks: int
    person_leaks: int
    person_support: int


class ModelCard(BaseModel):
    name: str
    dev_f1: float
    size_mb: float
    train_seconds: float
    parameters: int | None


class MetricsOut(BaseModel):
    split: Literal["test"]
    redaction: list[DetectorScore]
    rag: RagScores | None
    guard: GuardScores | None
    models: list[ModelCard]


class UploadIssue(BaseModel):
    where: str | None = Field(description='Location only, e.g. "Line 4". Never quotes file text.')
    message: str


class UploadFileReport(BaseModel):
    name: str
    calls: int = Field(description="Calls accepted from this file")
    turns: int
    issues: list[UploadIssue] = Field(description="Calls with an issue are not imported")


class UploadJob(BaseModel):
    id: str
    status: UploadStatus
    created_at: str = Field(description="ISO 8601, UTC")
    files: list[UploadFileReport]
    n_calls: int = Field(description="Calls accepted for redaction")
    call_ids: list[str] = Field(description="Ids of the imported calls, once saved")
    redactions: dict[str, int] = Field(description="Placeholders added, by type")
    indexed: bool = Field(description="Whether the calls were added to the search index")
    message: str | None = None


class UploadList(BaseModel):
    items: list[UploadJob]


class AgentIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AgentStepOut(BaseModel):
    n: int = Field(description="Step number; the model cites statistics results as r<n>")
    tool: str | None = Field(description="Tool the model called, or null for an unreadable reply")
    args: dict[str, Any]
    summary: str = Field(description="Counts and ids only, never transcript text")
    ok: bool
    ms: int


class AgentProvenance(BaseModel):
    provider: str
    model: str
    prompt_version: str
    max_steps: int
    retrieval: Mode


class AgentAnswerOut(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[CitationOut] = Field(description="Cited turns")
    cited_calls: list[str] = Field(description="Every call the answer cites, directly or through a turn")
    cited_results: list[int] = Field(description="Steps whose statistics or report the answer cites")
    steps: list[AgentStepOut]
    rejected: str | None = Field(description="Why the answer was discarded, if it was")
    provenance: AgentProvenance
    llm_calls: int
    elapsed_ms: int
