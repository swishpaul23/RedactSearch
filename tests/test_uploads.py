"""Uploads: file parsing, the redact-save-index job, and the HTTP routes. Offline."""

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callsearch.api.main import create_app
from callsearch.api.services import CallStore, ResultsReader, Services
from callsearch.api.uploads import InlineExecutor, UploadJobs
from callsearch.ingest import MAX_TURN_CHARS, parse_file
from callsearch.redact.base import Span
from callsearch.redact.export import load_redacted, write_redacted
from callsearch.redact.pipeline import PipelineDetector
from callsearch.redact.regex import RegexDetector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_rag import FakeEmbedder, call  # noqa: E402

CARD = "4111 1111 1111 1111"


class NameDetector:
    """Stands in for the DistilBERT layer: every capitalised word after 'is' is a name."""
    name = "names"

    def detect(self, text):
        return [Span(m.start(1), m.end(1), "PERSON", "names") for m in re.finditer(r"\bis ([A-Z][a-z]+)", text)]


def detector():
    d = PipelineDetector([RegexDetector(), NameDetector()], name="pipeline_hf")
    return d


# --- parsing ------------------------------------------------------------------------------

def test_parse_jsonl_with_aliases_and_estimated_times():
    data = "\n".join([
        json.dumps({"turns": [{"speaker": "Agent", "text": "Hello there"}, {"role": "caller", "text": "Hi, refund"}]}),
        "",
        json.dumps({"turns": [{"speaker": "rep", "text": "One", "start_sec": 3}, {"speaker": "client", "text": "Two"}]}),
    ]).encode()
    p = parse_file("batch.jsonl", data)
    assert p.issues == [] and len(p.calls) == 2 and p.n_turns == 4
    assert [t["speaker"] for t in p.calls[0]] == ["agent", "customer"]
    assert p.calls[0][0]["start_sec"] == 0 and p.calls[0][1]["start_sec"] > 0
    assert p.calls[1][1]["start_sec"] > 3


def test_parse_json_list_object_and_wrapper():
    one = {"turns": [{"speaker": "agent", "text": "a"}]}
    assert len(parse_file("x.json", json.dumps(one).encode()).calls) == 1
    assert len(parse_file("x.json", json.dumps([one, one]).encode()).calls) == 2
    assert len(parse_file("x.json", json.dumps({"calls": [one, one, one]}).encode()).calls) == 3


def test_parse_txt_timestamps_and_continuation_lines():
    text = "[00:05] Agent: Thanks for calling.\nCustomer: I need help\nwith my order.\n\n[1:02:03] Agent: Sure."
    p = parse_file("call.txt", text.encode())
    assert p.issues == [] and len(p.calls) == 1
    turns = p.calls[0]
    assert [t["start_sec"] for t in (turns[0], turns[2])] == [5.0, 3723.0]
    assert turns[1]["text"] == "I need help with my order."


def test_issues_name_the_location_and_never_quote_text():
    secret = "Zebediah"
    bad = "\n".join([json.dumps({"turns": [{"speaker": secret, "text": f"hi {secret}"}]}), "{not json " + secret])
    p = parse_file("bad.jsonl", bad.encode())
    assert p.calls == [] and [i.where for i in p.issues] == ["Line 1, turn 1", "Line 2"]
    assert all(secret not in i.message for i in p.issues)
    txt = parse_file("t.txt", f"{secret}: hello\nAgent: hi".encode())
    assert txt.calls == [] and txt.issues[0].where == "Line 1" and secret not in txt.issues[0].message


@pytest.mark.parametrize("name,data,msg", [
    ("notes.docx", b"x", "Use a .jsonl"),
    ("a.txt", b"\xff\xfe\x00", "UTF-8"),
    ("a.txt", b"   \n", "empty"),
    ("a.txt", f"Agent: {'x' * (MAX_TURN_CHARS + 1)}".encode(), "longer than"),
])
def test_rejected_files(name, data, msg):
    p = parse_file(name, data)
    assert p.calls == [] and msg in p.issues[0].message


def test_one_bad_call_does_not_block_the_others():
    good = json.dumps({"turns": [{"speaker": "agent", "text": "ok"}]})
    p = parse_file("b.jsonl", f"{good}\n{{}}\n{good}".encode())
    assert len(p.calls) == 2 and p.issues[0].where == "Line 2"


def test_file_name_is_reduced_to_its_base_name():
    assert parse_file("../../etc/x.txt", b"Agent: hi").name == "x.txt"
    assert parse_file("C:\\Users\\me\\calls.json", b"[]").name == "calls.json"


# --- the job ------------------------------------------------------------------------------

@pytest.fixture
def stack(tmp_path):
    """Real redacted store and Chroma index on disk, fake embedder and redactor."""
    from build_index import build
    from callsearch.rag.store import CallIndex

    write_redacted([call("call_9001"), call("call_9002", split="dev", turns=("Booking", "A table for two"))],
                   tmp_path / "red", "pipeline_hf")
    build(tmp_path / "red", tmp_path / "idx", FakeEmbedder(), context_turns=1)
    svc = Services(calls=CallStore(redacted_dir=tmp_path / "red"), index=CallIndex(tmp_path / "idx", FakeEmbedder()),
                   results=ResultsReader(tmp_path / "none"), detector=detector())
    jobs = UploadJobs(svc, executor=InlineExecutor())
    return tmp_path, svc, TestClient(create_app(svc, web_dist=Path("/nonexistent"), uploads=jobs,
                                                uploads_per_minute=100))


def upload(client, *files):
    return client.post("/api/uploads", files=[("files", (n, d, "application/octet-stream")) for n, d in files])


def test_upload_redacts_saves_and_indexes(stack, capsys):
    tmp_path, svc, c = stack
    txt = f"Agent: Thanks, my name is Alice.\nCustomer: Charge card {CARD} for the parrot food please."
    r = upload(c, ("call.txt", txt.encode()))
    job = r.json()
    assert r.status_code == 202 and job["status"] == "done" and job["indexed"] is True
    assert job["n_calls"] == 1 and job["redactions"] == {"CARD": 1, "PERSON": 1}
    cid = job["call_ids"][0]
    assert re.fullmatch(r"up-[0-9a-f]{12}-001", cid)

    saved, manifest = load_redacted(tmp_path / "red")
    stored = next(x for x in saved if x["call_id"] == cid)
    assert manifest["n_calls"] == 3 and stored["split"] == stored["source"] == "upload"
    on_disk = (tmp_path / "red" / "calls.jsonl").read_text() + "".join(d for _, d in svc.index.documents())
    assert "Alice" not in on_disk and CARD not in on_disk and "[PERSON]" in on_disk

    hits = svc.index.search("parrot food", k=1, mode="bm25")
    assert hits[0].call_id == cid and hits[0].split == "upload"
    assert svc.index.info["n_chunks"] == 6
    detail = c.get(f"/api/calls/{cid}").json()
    assert detail["redactions"] == {"CARD": 1, "PERSON": 1}
    assert c.get("/api/calls", params={"source": "upload"}).json()["total"] == 1
    assert c.get(f"/api/uploads/{job['id']}").json() == job
    assert c.get("/api/uploads").json()["items"][0]["id"] == job["id"]
    assert "Alice" not in capsys.readouterr().out


def test_upload_reports_issues_and_imports_nothing_when_all_bad(stack):
    _, _, c = stack
    job = upload(c, ("a.txt", b"Bob: hello"), ("b.csv", b"x,y")).json()
    assert job["status"] == "failed" and job["n_calls"] == 0 and job["call_ids"] == []
    assert [f["name"] for f in job["files"]] == ["a.txt", "b.csv"]
    assert job["files"][0]["issues"] == [{"where": "Line 1", "message": "Start the line with Agent: or Customer:."}]
    assert "Bob" not in json.dumps(job)


def test_upload_without_index_saves_but_says_so(tmp_path):
    from callsearch.rag.store import CallIndex

    svc = Services(calls=CallStore(redacted_dir=tmp_path / "red"), index=CallIndex(tmp_path / "noidx"),
                   results=ResultsReader(tmp_path), detector=detector())
    c = TestClient(create_app(svc, web_dist=Path("/nonexistent"), uploads=UploadJobs(svc, InlineExecutor())))
    job = upload(c, ("a.txt", b"Agent: hello")).json()
    assert job["status"] == "done" and job["indexed"] is False and "not searchable" in job["message"]
    assert not (tmp_path / "noidx").exists()
    assert len(load_redacted(tmp_path / "red")[0]) == 1


def test_upload_unexpected_error_logs_no_text(stack, capsys):
    _, svc, c = stack

    class Boom:
        name = "pipeline_hf"

        def detect(self, text):
            raise ValueError(f"cannot parse {text}")

    svc._detector = Boom()
    job = upload(c, ("a.txt", b"Agent: my secret is Quixote")).json()
    assert job["status"] == "failed" and job["id"] in job["message"]
    out = capsys.readouterr().out
    assert "ValueError" in out and "Quixote" not in out and "Quixote" not in json.dumps(job)


def test_upload_limits(stack):
    _, _, c = stack
    r = c.post("/api/uploads", content=b"x", headers={"content-length": str(10**9), "content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "upload_too_large"
    assert c.post("/api/uploads").status_code == 422
    assert c.get("/api/uploads/nope").status_code == 404


def test_upload_rate_limit(tmp_path):
    svc = Services(calls=CallStore([]), index=None, results=ResultsReader(tmp_path), detector=detector())
    c = TestClient(create_app(svc, web_dist=Path("/nonexistent"), uploads=UploadJobs(svc, InlineExecutor()),
                              uploads_per_minute=1))
    assert upload(c, ("a.csv", b"x")).status_code == 202
    r = upload(c, ("a.csv", b"x"))
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_uploads_can_be_turned_off(tmp_path):
    svc = Services(calls=CallStore([]), index=None, results=ResultsReader(tmp_path), detector=detector())
    c = TestClient(create_app(svc, web_dist=Path("/nonexistent"), uploads=UploadJobs(svc, InlineExecutor()),
                              uploads_enabled=False))
    r = upload(c, ("a.jsonl", b"{}"))
    assert r.status_code == 503 and r.json()["error"]["code"] == "uploads_disabled"
    assert c.get("/api/uploads").json() == {"items": []}
