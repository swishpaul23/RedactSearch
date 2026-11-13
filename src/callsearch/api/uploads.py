"""Upload jobs: parse now, redact and index in the background, report progress.

Jobs live in memory (the last MAX_JOBS), so a restart forgets the list but never the calls:
imported calls are already saved with the redacted data. One worker runs jobs in order,
because the redaction model and the index are shared.
"""

from __future__ import annotations

import json
import secrets
import sys
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime, timezone

from callsearch.api.schemas import UploadFileReport, UploadIssue, UploadJob
from callsearch.api.services import Services, Unavailable
from callsearch.ingest import MAX_CALLS, MAX_FILES, ParsedFile, parse_file

MAX_JOBS = 50


class InlineExecutor(Executor):
    """Runs a job inside submit(); tests use it to avoid waiting on a thread."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


class UploadJobs:
    def __init__(self, services: Services, executor: Executor | None = None):
        self.svc = services
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload")
        self.jobs: OrderedDict[str, UploadJob] = OrderedDict()
        self.lock = threading.Lock()

    def _save(self, job: UploadJob) -> None:
        with self.lock:
            self.jobs[job.id] = job
            self.jobs.move_to_end(job.id)
            while len(self.jobs) > MAX_JOBS:
                self.jobs.popitem(last=False)

    def _update(self, job_id: str, **fields) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is not None:
                self.jobs[job_id] = job.model_copy(update=fields)

    def get(self, job_id: str) -> UploadJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def recent(self) -> list[UploadJob]:
        with self.lock:
            return list(reversed(self.jobs.values()))

    def submit(self, files: list[tuple[str | None, bytes]]) -> UploadJob:
        job_id = secrets.token_hex(6)
        parsed: list[ParsedFile] = [parse_file(name, data) for name, data in files[:MAX_FILES]]
        reports = [UploadFileReport(name=p.name, calls=len(p.calls), turns=p.n_turns,
                                    issues=[UploadIssue(where=i.where, message=i.message) for i in p.issues])
                   for p in parsed]
        raw = [turns for p in parsed for turns in p.calls]
        message = None
        if len(files) > MAX_FILES:
            message = f"Only the first {MAX_FILES} files were read. Upload the rest separately."
        if len(raw) > MAX_CALLS:
            raw, message = [], f"This upload has {len(raw)} calls; the limit is {MAX_CALLS}. Nothing was imported."
        elif not raw:
            message = message or "No calls could be imported. Fix the issues listed and upload again."
        job = UploadJob(id=job_id, status="queued" if raw else "failed",
                        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), files=reports,
                        n_calls=len(raw), call_ids=[], redactions={}, indexed=False, message=message)
        self._save(job)
        if raw:
            calls = [{"call_id": f"up-{job_id}-{n:03d}", "split": "upload", "source": "upload",
                      "turns": [{"turn_id": i, **t} for i, t in enumerate(turns)]}
                     for n, turns in enumerate(raw, 1)]
            self.executor.submit(self._run, job_id, calls)
        return self.get(job_id)

    def _run(self, job_id: str, calls: list[dict]) -> None:
        try:
            result = self.svc.ingest(calls, lambda stage: self._update(job_id, status=stage))
            note = None if result["indexed"] else "Saved, but not searchable yet: build the search index first."
            self._update(job_id, status="done", message=note, **result)
        except Unavailable as e:
            self._update(job_id, status="failed", message=str(e))
        except Exception as e:  # the message may quote raw text, so log the type and location only
            frame = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(json.dumps({"upload_job": job_id, "error": type(e).__name__,
                              "at": f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"}), flush=True)
            self._update(job_id, status="failed", message=f"Import stopped by an unexpected error. Check the "
                         f"server log for job {job_id}, then upload again.")
        finally:
            del calls
