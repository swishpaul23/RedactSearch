"""HTTP API for the web app: cited answers, search, redacted calls and quality metrics.

python -m callsearch.api.main            # http://127.0.0.1:8000, docs at /api/docs
uvicorn callsearch.api.main:app --reload

Serves the built web app (web/dist) at / when it exists. Reads redacted data only.
Every response carries an X-Request-ID; logs hold method, path, status, timing and that id,
never question or transcript text. /api/ask is rate limited per client, since every call
runs the answer model.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from callsearch import __version__ as VERSION
from callsearch.api.schemas import (AgentAnswerOut, AgentIn, AgentProvenance, AgentStepOut, AnswerOut, AskIn, CallDetail, CallPage, CitationOut, ErrorOut, HealthOut,
                                    HitOut, IndexInfo, LLMInfo, MetricsOut, Mode, Provenance, SearchOut, Source,
                                    Split, UploadJob, UploadList)
from callsearch.api.services import Services, Unavailable
from callsearch.api.uploads import UploadJobs
from callsearch.ingest import MAX_FILE_BYTES, MAX_FILES

WEB_DIST = Path(os.environ.get("CALLSEARCH_WEB_DIST", "web/dist"))
ASK_PER_MINUTE = int(os.environ.get("CALLSEARCH_ASK_PER_MINUTE", "20"))
UPLOADS_PER_MINUTE = int(os.environ.get("CALLSEARCH_UPLOADS_PER_MINUTE", "10"))
# "off" in the hosted demo: Lambda freezes the process after each response, so a background job cannot run.
UPLOADS_ENABLED = os.environ.get("CALLSEARCH_UPLOADS", "on") != "off"
MAX_UPLOAD_BODY = MAX_FILES * MAX_FILE_BYTES + 1_000_000
ERRORS = {r: {"model": ErrorOut} for r in (400, 404, 413, 422, 429, 503)}

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                                "base-uri 'self'; form-action 'self'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
DOCS_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: https://fastapi.tiangolo.com; "
            "frame-ancestors 'none'")


class RateLimiter:
    """Sliding one-minute window per client key."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> int:
        """0 when allowed, otherwise seconds until the next request is allowed."""
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            return 0


def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def create_app(services: Services | None = None, ask_per_minute: int = ASK_PER_MINUTE,
               web_dist: Path = WEB_DIST, uploads: UploadJobs | None = None,
               uploads_per_minute: int = UPLOADS_PER_MINUTE, uploads_enabled: bool = UPLOADS_ENABLED) -> FastAPI:
    svc = services or Services()
    limiter = RateLimiter(ask_per_minute)
    upload_limiter = RateLimiter(uploads_per_minute)
    jobs = uploads or UploadJobs(svc)
    app = FastAPI(title="RedactSearch API", version=VERSION, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json",
                  description="Search and cited answers over redacted call transcripts. Redacted data only.")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        too_big = request.url.path == "/api/uploads" and int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BODY
        response = (error(413, "upload_too_large", f"Uploads are limited to {MAX_FILES} files of "
                          f"{MAX_FILE_BYTES // 1_000_000} MB each.") if too_big else await call_next(request))
        ms = round((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-ID"] = rid
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path == "/api/docs":
            response.headers["Content-Security-Policy"] = DOCS_CSP
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
            print(json.dumps({"rid": rid, "method": request.method, "path": request.url.path,
                              "status": response.status_code, "ms": ms}), flush=True)
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
        return error(422, "invalid_request", f"{field}: {first.get('msg', 'invalid value')}")

    @app.exception_handler(Unavailable)
    async def unavailable(_: Request, exc: Unavailable):
        return error(503, exc.code, str(exc))

    @app.get("/api/health", response_model=HealthOut, tags=["system"])
    def health() -> HealthOut:
        info = svc.index_info()
        llm = svc.llm_info()
        return HealthOut(status="ok", version=VERSION, calls=svc.calls.count(),
                         index=IndexInfo(available=info is not None, **(info or {})),
                         llm=LLMInfo(**llm) if llm else None)

    @app.post("/api/ask", response_model=AnswerOut, responses=ERRORS, tags=["answers"])
    def ask(body: AskIn, request: Request) -> AnswerOut | JSONResponse:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait:
            return error(429, "rate_limited", f"Too many questions. Try again in {wait} seconds.",
                         {"Retry-After": str(wait)})
        t0 = time.perf_counter()
        a = svc.ask(body.question.strip(), body.k, body.mode, body.split)
        return AnswerOut(question=a.question, answer=a.answer, found=a.found, rejected=a.rejected,
                         citations=[CitationOut(**c.__dict__) for c in a.citations], retrieved=a.retrieved,
                         provenance=Provenance(**a.provenance), elapsed_ms=round((time.perf_counter() - t0) * 1000))

    @app.post("/api/agent", response_model=AgentAnswerOut, responses=ERRORS, tags=["answers"])
    def agent(body: AgentIn, request: Request) -> AgentAnswerOut | JSONResponse:
        client = request.client.host if request.client else "unknown"
        wait = limiter.check(client)
        if wait:
            return error(429, "rate_limited", f"Too many questions. Try again in {wait} seconds.",
                         {"Retry-After": str(wait)})
        t0 = time.perf_counter()
        a = svc.run_agent(body.question.strip())
        return AgentAnswerOut(
            question=a.question, answer=a.answer, found=a.found,
            citations=[CitationOut(**c.__dict__) for c in a.citations], cited_calls=a.cited_calls,
            cited_results=a.cited_results,
            steps=[AgentStepOut(n=s.n, tool=s.tool, args=s.args, summary=s.summary, ok=s.ok, ms=s.ms) for s in a.steps],
            rejected=a.rejected, provenance=AgentProvenance(**a.provenance), llm_calls=a.llm_calls,
            elapsed_ms=round((time.perf_counter() - t0) * 1000))

    @app.get("/api/search", response_model=SearchOut, responses=ERRORS, tags=["search"])
    def search(q: str = Query(min_length=2, max_length=500), k: int = Query(10, ge=1, le=50),
               mode: Mode = "bm25", split: Split | None = None) -> SearchOut:
        hits = svc.require_index().search(q, k=k, split=split, mode=mode)
        return SearchOut(query=q, mode=mode, hits=[HitOut(**h.__dict__) for h in hits])

    @app.get("/api/calls", response_model=CallPage, responses=ERRORS, tags=["calls"])
    def calls(q: str | None = Query(None, max_length=200), split: Split | None = None,
              source: Source | None = None,
              sort: Literal["call_id", "n_turns", "duration_sec", "redactions"] = "call_id",
              order: Literal["asc", "desc"] = "asc", page: int = Query(1, ge=1),
              page_size: int = Query(25, ge=5, le=100)) -> CallPage:
        items, total = svc.calls.page(q, split, source, sort, order, page, page_size)
        return CallPage(items=items, total=total, page=page, page_size=page_size)

    @app.get("/api/calls/{call_id}", response_model=CallDetail, responses=ERRORS, tags=["calls"])
    def call(call_id: str) -> CallDetail:
        detail = svc.calls.detail(call_id)
        if detail is None:
            raise HTTPException(404, f"No call with id {call_id}.")
        return detail

    @app.get("/api/metrics", response_model=MetricsOut, tags=["quality"])
    def metrics() -> MetricsOut:
        r = svc.results
        return MetricsOut(split="test", redaction=r.redaction(), rag=r.rag(), guard=r.guard(), models=r.models())

    @app.post("/api/uploads", response_model=UploadJob, status_code=202, responses=ERRORS, tags=["uploads"])
    async def upload(request: Request, files: list[UploadFile] = File(description=".jsonl, .json or .txt")):
        if not uploads_enabled:
            return error(503, "uploads_disabled", "Uploads are turned off in the hosted demo. "
                         "Run the app locally to redact and search your own transcripts.")
        client = request.client.host if request.client else "unknown"
        wait = upload_limiter.check(client)
        if wait:
            return error(429, "rate_limited", f"Too many uploads. Try again in {wait} seconds.",
                         {"Retry-After": str(wait)})
        read = []
        for f in files[:MAX_FILES + 1]:
            read.append((f.filename, await f.read(MAX_FILE_BYTES + 1)))
            await f.close()
        return jobs.submit(read)

    @app.get("/api/uploads", response_model=UploadList, tags=["uploads"])
    def uploads_list() -> UploadList:
        return UploadList(items=jobs.recent())

    @app.get("/api/uploads/{job_id}", response_model=UploadJob, responses=ERRORS, tags=["uploads"])
    def upload_status(job_id: str) -> UploadJob:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "No upload with that id. Upload history resets when the server restarts.")
        return job

    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
    def api_not_found(rest: str):
        return error(404, "not_found", f"No API route /api/{rest}.")

    if (web_dist / "index.html").exists():
        index_html = web_dist / "index.html"
        root = web_dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            target = (web_dist / path).resolve()
            if path and target.is_file() and root in target.parents:
                cache = "public, max-age=31536000, immutable" if path.startswith("assets/") else "no-cache"
                return FileResponse(target, headers={"Cache-Control": cache})
            return FileResponse(index_html, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("callsearch.api.main:app", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
