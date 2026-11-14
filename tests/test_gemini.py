"""GeminiProvider without the network: a fake post function stands in for the REST call."""

import io
import json
import urllib.error

import pytest

from callsearch.redact.llm import GeminiProvider, MissingApiKey, ModelBusy, default_provider


def reply(text='{"answer": "ok"}'):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}}


def http_error(code):
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"{}"))


@pytest.fixture
def no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CALLSEARCH_GEMINI_MODEL", raising=False)
    return tmp_path / ".env"  # does not exist


def test_request_shape_key_in_header_and_usage_counted(no_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-123")
    seen = {}

    def post(payload, key):
        seen.update(payload=payload, key=key)
        return reply()

    p = GeminiProvider(env_file=no_env, post=post)
    assert p.complete("SYSTEM", "USER") == '{"answer": "ok"}'
    assert seen["key"] == "k-123" and "k-123" not in json.dumps(seen["payload"])
    assert seen["payload"]["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    assert seen["payload"]["contents"][0]["parts"][0]["text"] == "USER"
    cfg = seen["payload"]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["thinkingConfig"]["thinkingBudget"] == 0
    assert (p.usage.input_tokens, p.usage.output_tokens) == (12, 3)
    assert p.model == "gemini-2.5-flash"


def test_key_and_model_from_dotenv(tmp_path, no_env):
    env = tmp_path / "with.env"
    env.write_text("GEMINI_API_KEY=from-file\nCALLSEARCH_GEMINI_MODEL=gemini-2.5-flash-lite\n")
    keys = []
    p = GeminiProvider(env_file=env, post=lambda payload, key: keys.append(key) or reply())
    p.complete("s", "u")
    assert keys == ["from-file"] and p.model == "gemini-2.5-flash-lite"


def test_missing_key(no_env):
    with pytest.raises(MissingApiKey, match="GEMINI_API_KEY"):
        GeminiProvider(env_file=no_env, post=lambda *a: reply()).complete("s", "u")


def test_retries_then_busy_on_429(no_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("callsearch.redact.llm.time.sleep", lambda s: None)
    calls = []

    def post(payload, key):
        calls.append(1)
        raise http_error(429)

    with pytest.raises(ModelBusy):
        GeminiProvider(env_file=no_env, post=post).complete("s", "u")
    assert len(calls) == 2


def test_transient_error_then_success(no_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr("callsearch.redact.llm.time.sleep", lambda s: None)
    answers = iter([http_error(503), reply("{}")])

    def post(payload, key):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert GeminiProvider(env_file=no_env, post=post).complete("s", "u") == "{}"


def test_bad_request_is_not_retried_and_blocked_reply_is_an_error(no_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = []

    def post(payload, key):
        calls.append(1)
        raise http_error(400)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        GeminiProvider(env_file=no_env, post=post).complete("s", "u")
    assert len(calls) == 1
    blocked = GeminiProvider(env_file=no_env, post=lambda *a: {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(RuntimeError, match="SAFETY"):
        blocked.complete("s", "u")


def test_default_provider_selects_gemini(monkeypatch):
    monkeypatch.setenv("CALLSEARCH_LLM_PROVIDER", "gemini")
    assert isinstance(default_provider(), GeminiProvider)
