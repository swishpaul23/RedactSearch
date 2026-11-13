"""Tests for the HTTP API, offline: fake index and answerer, small in-memory call set."""

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callsearch.api.main import RateLimiter, create_app
from callsearch.api.services import CallStore, ResultsReader, Services, Unavailable
from callsearch.rag.answer import NOT_FOUND, Answer, Citation
from callsearch.rag.store import Hit

ROOT = Path(__file__).resolve().parents[1]

CALLS = [
    {"call_id": "call_0001", "split": "test", "source": "template", "turns": [
        {"turn_id": 0, "speaker": "agent", "start_sec": 0.0, "text": "Thanks for calling, this is [PERSON]."},
        {"turn_id": 1, "speaker": "customer", "start_sec": 4.5, "text": "I want a refund on card [CARD]."},
        {"turn_id": 2, "speaker": "agent", "start_sec": 9.0, "text": "Refund issued to [PERSON]."}]},
    {"call_id": "call_0002", "split": "dev", "source": "multiwoz", "turns": [
        {"turn_id": 0, "speaker": "customer", "start_sec": 0.0, "text": "Book a table for two."},
        {"turn_id": 1, "speaker": "agent", "start_sec": 3.0, "text": "Booked at the Galleria."}]},
]


class FakeIndex:
    info = {"embed_model": "bge-small", "detector": "pipeline_hf", "n_chunks": 5}

    def search(self, q, k=5, split=None, mode="bm25"):
        return [Hit("call_0001:1", "call_0001", 1, "customer", 4.5, "test", "I want a refund on card [CARD].", 3.2)][:k]


class FakeAnswerer:
    def __init__(self, k, mode):
        self.k, self.mode = k, mode

    def ask(self, question, split=None):
        prov = {"provider": "fake", "model": "fake-1", "prompt_version": "v1", "k": self.k, "retrieval": self.mode,
                "neighbors": 1, "grounding": True}
        if "boat" in question:
            return Answer(question, NOT_FOUND, False, [], ["call_0001:1"], "question term 'boat' appears nowhere", prov)
        if "crash" in question:
            raise RuntimeError("Ollama is not running")
        return Answer(question, "They want a refund.", True,
                      [Citation("call_0001:1", "call_0001", 1, "customer", 4.5, "I want a refund on card [CARD].")],
                      ["call_0001:1"], None, prov)


def client(results_dir=None, per_minute=100, index=FakeIndex(), web_dist=Path("/nonexistent")):
    svc = Services(calls=CallStore(CALLS), index=index, answerer=FakeAnswerer,
                   results=ResultsReader(results_dir or ROOT / "results"))
    return TestClient(create_app(svc, ask_per_minute=per_minute, web_dist=web_dist))


def test_health():
    r = client().get("/api/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "ok" and body["calls"] == 2
    assert body["index"] == {"available": True, "embed_model": "bge-small", "detector": "pipeline_hf", "n_chunks": 5}


def test_ask_returns_cited_answer():
    r = client().post("/api/ask", json={"question": "  why a refund?  ", "k": 3, "mode": "hybrid"})
    body = r.json()
    assert r.status_code == 200 and body["found"] and body["question"] == "why a refund?"
    assert body["citations"][0] == {"chunk_id": "call_0001:1", "call_id": "call_0001", "turn_id": 1,
                                    "speaker": "customer", "start_sec": 4.5,
                                    "snippet": "I want a refund on card [CARD]."}
    assert body["provenance"]["k"] == 3 and body["provenance"]["retrieval"] == "hybrid"
    assert isinstance(body["elapsed_ms"], int)


def test_ask_not_found_reports_reason():
    body = client().post("/api/ask", json={"question": "which boat?"}).json()
    assert not body["found"] and body["answer"] == NOT_FOUND and "boat" in body["rejected"]


@pytest.mark.parametrize("payload,field", [({"question": "hi"}, "question"), ({"question": "fine q", "k": 0}, "k"),
                                           ({"question": "fine q", "mode": "magic"}, "mode"), ({}, "question")])
def test_ask_validation_errors(payload, field):
    r = client().post("/api/ask", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    assert r.json()["error"]["message"].startswith(field)


def test_ask_rate_limited_with_retry_after():
    c = client(per_minute=2)
    assert [c.post("/api/ask", json={"question": "refund?"}).status_code for _ in range(3)] == [200, 200, 429]
    r = c.post("/api/ask", json={"question": "refund?"})
    assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["retry-after"]) >= 1


def test_rate_limiter_window():
    rl = RateLimiter(2)
    assert rl.check("a", 0) == 0 and rl.check("a", 1) == 0 and rl.check("a", 2) > 0
    assert rl.check("b", 2) == 0 and rl.check("a", 61) == 0


def test_ask_model_down_is_503_with_fix():
    r = client().post("/api/ask", json={"question": "crash please"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "answer_model_unavailable"
    assert "Ollama" in r.json()["error"]["message"]


class NoIndex:
    @property
    def info(self):
        raise FileNotFoundError("no index")


def test_missing_index_is_503_with_fix():
    c = client(index=NoIndex())
    assert c.get("/api/health").json()["index"]["available"] is False
    r = c.post("/api/ask", json={"question": "refund?"})
    assert r.status_code == 503 and "build_index.py" in r.json()["error"]["message"]
    assert c.get("/api/search?q=refund").status_code == 503


def test_search():
    body = client().get("/api/search?q=refund&k=3&mode=bm25").json()
    assert body["mode"] == "bm25" and body["hits"][0]["chunk_id"] == "call_0001:1" and body["hits"][0]["split"] == "test"


def test_calls_list_filter_search_sort_paginate():
    c = client()
    page = c.get("/api/calls").json()
    assert page["total"] == 2 and [i["call_id"] for i in page["items"]] == ["call_0001", "call_0002"]
    first = page["items"][0]
    assert first == {"call_id": "call_0001", "split": "test", "source": "template", "n_turns": 3,
                     "duration_sec": 9.0, "redactions": 3, "preview": "I want a refund on card [CARD]."}
    assert c.get("/api/calls?split=dev").json()["total"] == 1
    assert c.get("/api/calls?source=template").json()["items"][0]["call_id"] == "call_0001"
    assert c.get("/api/calls?q=galleria").json()["items"][0]["call_id"] == "call_0002"  # searches text
    assert c.get("/api/calls?q=CALL_0002").json()["total"] == 1                        # and ids
    assert [i["call_id"] for i in c.get("/api/calls?sort=redactions&order=desc").json()["items"]] == [
        "call_0001", "call_0002"]
    p2 = c.get("/api/calls?page=2&page_size=5").json()
    assert p2["items"] == [] and p2["total"] == 2
    assert c.get("/api/calls?sort=text").status_code == 422


def test_call_detail_and_404():
    c = client()
    d = c.get("/api/calls/call_0001").json()
    assert d["redactions"] == {"CARD": 1, "PERSON": 2} and d["duration_sec"] == 9.0 and len(d["turns"]) == 3
    r = c.get("/api/calls/call_9999")
    assert r.status_code == 404 and r.json() == {"error": {"code": "not_found", "message": "No call with id call_9999."}}


def test_calls_unavailable_without_manifest(tmp_path):
    svc = Services(calls=CallStore(redacted_dir=tmp_path), index=FakeIndex(), answerer=FakeAnswerer,
                   results=ResultsReader(tmp_path))
    c = TestClient(create_app(svc, web_dist=Path("/nonexistent")))
    assert c.get("/api/health").json()["calls"] == 0
    r = c.get("/api/calls")
    assert r.status_code == 503 and r.json()["error"]["code"] == "calls_unavailable"


def test_metrics_reads_results_and_skips_missing(tmp_path):
    body = client(results_dir=tmp_path).get("/api/metrics").json()
    assert body == {"split": "test", "redaction": [], "rag": None, "guard": None, "models": []}
    (tmp_path / "index_guard.json").write_text(json.dumps({
        "passed": True, "n_chunks": 10, "planted_leaks": {"PERSON": 2, "CARD": 0, "EMAIL": 1},
        "planted_support": {"PERSON": 50}}))
    g = client(results_dir=tmp_path).get("/api/metrics").json()["guard"]
    assert g == {"passed": True, "n_chunks": 10, "structured_leaks": 1, "person_leaks": 2, "person_support": 50}


@pytest.mark.skipif(not (ROOT / "results" / "eval_pipeline_hf_test.json").exists(), reason="no results")
def test_metrics_on_committed_results():
    body = client().get("/api/metrics").json()
    hf = next(d for d in body["redaction"] if d["detector"] == "pipeline_hf")
    assert round(hf["person_f1"], 3) == 0.970 and round(hf["leak_rate"], 3) == 0.023
    assert body["redaction"][0]["detector"] == "pipeline_hf"  # display order


def test_headers_request_id_and_security():
    c = client()
    r = c.get("/api/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["x-request-id"] == "abc123" and r.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert len(c.get("/api/health").headers["x-request-id"]) == 16


def test_logs_never_contain_question_text(capsys):
    client().post("/api/ask", json={"question": "why a refund for the secret plan?"})
    out = capsys.readouterr().out
    assert "secret plan" not in out and '"path": "/api/ask"' in out


def test_unknown_api_route_is_json_404():
    r = client().get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_serves_spa_with_fallback_and_no_traversal(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html>app</html>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path.parent / "secret.txt").write_text("nope")
    c = client(web_dist=tmp_path)
    assert c.get("/").text == "<html>app</html>"
    assert c.get("/calls/call_0001").text == "<html>app</html>"  # client-side route
    r = c.get("/assets/app.js")
    assert r.text == "console.log(1)" and "immutable" in r.headers["cache-control"]
    assert c.get("/../secret.txt").text == "<html>app</html>"
    assert c.get("/api/calls").status_code == 200  # API still wins over the SPA


def test_openapi_file_matches_code():
    """web/openapi.json is generated (scripts/export_openapi.py); the TS types come from it."""
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    committed = ROOT / "web" / "openapi.json"
    assert committed.exists(), "run python scripts/export_openapi.py"
    assert committed.read_text(encoding="utf-8") == mod.render(), (
        "API changed: run python scripts/export_openapi.py, then (cd web && npm run gen:api)")
