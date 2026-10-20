"""Tests for the LLM layer with mocked providers. No network."""

import json

import pytest

from callsearch.redact.base import Span, get_detector
from callsearch.redact.llm import (SYSTEM_PROMPT, BudgetExceeded, LLMDetector, OllamaProvider, locate, mask,
                                   parse_response)
from callsearch.redact.merge import redact


class FakeProvider:
    name, model = "fake", "fake-1"
    price_in = price_out = 0.0

    def __init__(self, reply):
        self.reply = reply  # str or callable(user) -> str
        self.calls = []

    def complete(self, system, user):
        self.calls.append(user)
        return self.reply(user) if callable(self.reply) else self.reply


class FixedBase:
    name = "base"

    def __init__(self, spans):
        self.spans = spans

    def detect(self, text):
        return list(self.spans)


def reply(*items):
    return json.dumps({"spans": [{"text": t, "type": ty} for t, ty in items]})


def det(base_spans, rep, tmp_path=None, **kw):
    return LLMDetector(base=FixedBase(base_spans), provider=FakeProvider(rep),
                       cache_dir=tmp_path, **kw)


# --- offsets ---------------------------------------------------------------------

def test_mask_matches_redact():
    text = "card 4539 1488 0343 6467 and Ann Lee, call Ann"
    spans = [Span(5, 24, "CARD", "regex"), Span(29, 36, "PERSON", "ner"), Span(30, 33, "PERSON", "ner")]
    masked, places = mask(text, spans)
    assert masked == redact(text, spans)
    assert [(masked[a:b], text[c:d]) for a, b, c, d in places] == [
        ("[CARD]", "4539 1488 0343 6467"), ("[PERSON]", "Ann Lee")]


def test_all_occurrences_mapped_to_original_offsets():
    text = "I'm Smith, card 4539 1488 0343 6467, Smith again, mr Smith."
    c = text.index("4539")
    d = det([Span(c, c + 19, "CARD", "regex")], reply(("Smith", "PERSON")))
    llm = [s for s in d.detect(text) if s.source == "llm"]
    assert [text[s.start:s.end] for s in llm] == ["Smith"] * 3
    assert [s.start for s in llm] == [i for i in range(len(text)) if text.startswith("Smith", i)]


def test_partial_word_matches_rejected():
    found = locate([{"text": "Ann", "type": "PERSON"}], "Annual fee for Ann.", [])
    assert found == [(15, 18, "PERSON")]


# --- discards --------------------------------------------------------------------

def test_hallucinated_bad_type_and_placeholder_spans_discarded():
    text = "this is Ann Lee speaking, hi"
    i = text.index("Ann Lee")
    d = det([Span(i, i + 7, "PERSON", "ner")],
            reply(("Bob Jones", "PERSON"),     # not in the text
                  ("speaking", "ADDRESS"),     # type not allowed
                  ("[PERSON]", "PERSON"),      # a placeholder
                  ("PERSON", "PERSON"),        # inside a placeholder
                  ("is [PERSON] sp", "PERSON"),  # overlaps a placeholder
                  ("hi", "PERSON")))           # valid (wrong, but verbatim and allowed)
    spans = d.detect(text)
    assert [(text[s.start:s.end], s.source) for s in spans] == [("Ann Lee", "ner"), ("hi", "llm")]
    assert d.stats["discard_not_found"] == 1 and d.stats["discard_bad_type"] == 1
    assert d.stats["discard_in_placeholder"] == 3


def test_type_case_normalised_and_bad_items_discarded():
    found = locate([{"text": "Ann", "type": "person"}, {"text": 5, "type": "PERSON"}, "Ann",
                    {"text": " ", "type": "PERSON"}], "hi Ann", [])
    assert found == [(3, 6, "PERSON")]


# --- parsing ---------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    '{"spans": [{"text": "Ann", "type": "PERSON"}]}',
    '```json\n{"spans": [{"text": "Ann", "type": "PERSON"}]}\n```',
    'Sure! Here you go:\n{"spans": [{"text": "Ann", "type": "PERSON"}]}\nHope that helps {x}',
])
def test_parse_tolerates_fences_and_extra_text(raw):
    assert parse_response(raw) == [{"text": "Ann", "type": "PERSON"}]


def test_malformed_json_returns_no_spans_and_is_counted():
    d = det([], "I think the name is Ann {spans: [oops")
    assert d.detect("hi Ann") == []
    d2 = det([], '{"names": ["Ann"]}')
    assert d2.detect("hi Ann") == []
    assert d.stats["parse_failures"] == 1 and d2.stats["parse_failures"] == 1


@pytest.mark.parametrize("raw", ["{}", '{"spans": []}', "```json\n{}\n```", "Nothing found. {}"])
def test_empty_answers_are_valid_not_failures(raw):
    d = det([], raw)
    assert d.detect("hi Ann") == []
    assert d.stats["parse_failures"] == 0 and d.stats["empty_answers"] == 1


def test_spans_object_preferred_over_earlier_empty_object():
    assert parse_response('{} then {"spans": [{"text": "Ann", "type": "PERSON"}]}') == [
        {"text": "Ann", "type": "PERSON"}]


# --- PERSON-only post-filter and prompt versions -------------------------------------

def test_post_filter_keeps_person_only():
    text = "Ann at ann@x.com, 604-555-0182"
    d = det([], reply(("ann@x.com", "EMAIL"), ("604-555-0182", "PHONE"), ("Ann", "PERSON")))
    assert [(text[s.start:s.end], s.type) for s in d.detect(text)] == [("Ann", "PERSON")]
    assert d.stats["discard_filtered_type"] == 2


def test_filtered_type_does_not_suppress_overlapping_person():
    # A longer CARD claim over the name must not win the overlap and then get filtered away.
    text = "hi Ann Lee there"
    d = det([], reply(("Ann Lee there", "CARD"), ("Ann Lee", "PERSON")))
    assert [(text[s.start:s.end], s.type) for s in d.detect(text)] == [("Ann Lee", "PERSON")]


def test_prompt_versions_kept_and_part_of_cache_key(tmp_path):
    from callsearch.redact.llm import PROMPT_VERSION, PROMPTS

    assert {"v1", "v2"} <= set(PROMPTS) and SYSTEM_PROMPT == PROMPTS[PROMPT_VERSION]
    assert '{"spans": []}' in PROMPTS["v2"] and "EMAIL:" not in PROMPTS["v2"]  # PERSON only
    v1 = det([], reply(), tmp_path, prompt_version="v1")
    v2 = det([], reply(), tmp_path, prompt_version="v2")
    assert v1._key("a") != v2._key("a")
    v1.detect("hi Ann")
    assert v1._provider.calls and v1.system_prompt == PROMPTS["v1"]
    with pytest.raises(KeyError):
        det([], reply(), prompt_version="v99")


def test_provenance_and_worker_defaults():
    from callsearch.redact.llm import AnthropicProvider

    d = det([], reply())
    assert d.provenance() == {"provider": "fake", "model": "fake-1", "prompt_version": "v2"}
    assert d.workers == 1  # FakeProvider has no default -> 1
    assert LLMDetector(provider=OllamaProvider(model="m")).workers == 1
    assert LLMDetector(provider=AnthropicProvider(client=object())).workers == 8
    assert LLMDetector(provider=OllamaProvider(model="m"), workers=3).workers == 3


def test_progress_printed_every_n_requests(tmp_path, capsys):
    d = det([], reply(), tmp_path, progress_every=2)
    d.prefetch([f"hi {n}" for n in "ABCDE"])
    assert capsys.readouterr().out.count("requests done") == 2


# --- cache, budget, plan -----------------------------------------------------------

def test_cache_hit_avoids_second_call(tmp_path):
    d = det([], reply(("Ann", "PERSON")), tmp_path)
    first = d.detect("hi Ann")
    d2 = det([], reply(("Ann", "PERSON")), tmp_path)  # fresh detector, same cache dir
    assert d2.detect("hi Ann") == first
    assert len(d._provider.calls) == 1 and d2._provider.calls == []
    assert d2.stats["cache_hits"] == 1 and d2.stats["requests"] == 0


def test_cache_key_depends_on_masked_text_model_and_prompt(tmp_path):
    d = det([], reply(), tmp_path)
    assert d._key("a") != d._key("b")
    other = det([], reply(), tmp_path)
    other._provider.model = "fake-2"
    assert d._key("a") != other._key("a")


def test_prefetch_respects_max_requests_and_plan_counts(tmp_path):
    texts = ["hi Ann", "hi Bob", "hi Ann", "[CARD]."]  # duplicates and placeholder-only turns cost nothing
    d = det([], reply(), tmp_path, max_requests=1)
    plan = d.plan(texts)
    assert plan["requests"] == 2 and plan["cache_hits"] == 0 and plan["est_input_tokens"] > len(SYSTEM_PROMPT) // 4
    assert d._provider.calls == []  # planning never calls the model
    with pytest.raises(BudgetExceeded):
        d.prefetch(texts)
    assert d._provider.calls == []  # aborted before any request
    d.max_requests = 2
    d.prefetch(texts)
    assert sorted(d._provider.calls) == ["hi Ann", "hi Bob"]
    assert d.plan(texts)["requests"] == 0


def test_nothing_left_after_masking_skips_the_model():
    d = det([Span(0, 7, "PERSON", "ner")], reply(("x", "PERSON")))
    assert [s.source for s in d.detect("Ann Lee.")] == ["ner"]
    assert d._provider.calls == []


def test_only_masked_text_is_sent():
    text = "card 4539 1488 0343 6467 for Ann"
    d = det([Span(5, 24, "CARD", "regex")], reply())
    d.detect(text)
    assert d._provider.calls == ["card [CARD] for Ann"]


# --- merge: add-only -----------------------------------------------------------------

def test_llm_spans_never_remove_base_spans():
    text = "Ann Lee, 604-555-0182, account 123456789012, and Grace"
    base = [Span(0, 7, "PERSON", "ner"), Span(9, 21, "PHONE", "regex"), Span(31, 43, "ACCOUNT", "regex")]
    # The model claims bigger, higher-priority spans around the base ones, plus one real addition.
    d = det(base, reply(("[PERSON], [PHONE]", "CARD"), ("account [ACCOUNT]", "SIN"), ("Grace", "PERSON"),
                        ("Gra", "CARD")))
    spans = d.detect(text)
    assert set(base) <= set(spans)
    assert [(text[s.start:s.end], s.type) for s in spans if s.source == "llm"] == [("Grace", "PERSON")]


def test_llm_spans_overlapping_each_other_resolved():
    d = det([], reply(("Ann Lee", "PERSON"), ("Lee", "PERSON")))
    assert [(s.start, s.end) for s in d.detect("hi Ann Lee")] == [(3, 10)]


# --- providers and registry --------------------------------------------------------

def test_ollama_provider_with_mocked_http(monkeypatch):
    sent = {}

    def fake_post(self, payload):
        sent.update(payload)
        return {"message": {"content": reply(("Ann", "PERSON"))}, "prompt_eval_count": 50, "eval_count": 9}

    monkeypatch.setattr(OllamaProvider, "_post", fake_post)
    p = OllamaProvider(model="qwen2:7b")
    out = p.complete("SYS", "hi Ann")
    assert parse_response(out) == [{"text": "Ann", "type": "PERSON"}]
    assert sent["model"] == "qwen2:7b" and sent["options"]["temperature"] == 0 and sent["stream"] is False
    assert sent["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi Ann"}]
    assert (p.usage.input_tokens, p.usage.output_tokens) == (50, 9)


def test_ollama_requires_model(monkeypatch):
    monkeypatch.delenv("CALLSEARCH_OLLAMA_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="CALLSEARCH_OLLAMA_MODEL"):
        OllamaProvider().complete("s", "u")


def test_anthropic_provider_with_mocked_client(monkeypatch):
    from types import SimpleNamespace as NS

    from callsearch.redact.llm import AnthropicProvider

    monkeypatch.delenv("CALLSEARCH_LLM_MODEL", raising=False)
    seen = {}

    class Msgs:
        def create(self, **kw):
            seen.update(kw)
            return NS(content=[NS(type="text", text='{"spans": []}')], usage=NS(input_tokens=10, output_tokens=3))

    p = AnthropicProvider(client=NS(messages=Msgs()))
    assert p.model == "claude-haiku-5-5"
    assert p.complete("SYS", "hi") == '{"spans": []}'
    assert seen["temperature"] == 0 and seen["system"] == "SYS" and seen["model"] == "claude-haiku-5-5"
    assert (p.usage.input_tokens, p.usage.output_tokens) == (10, 3)


def test_anthropic_missing_key_is_clear(monkeypatch):
    from callsearch.redact.llm import AnthropicProvider

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider().complete("s", "u")


def test_registry_is_lazy():
    for name in ("llm", "pipeline_llm"):
        d = get_detector(name)
        assert d.name == name and d._provider is None and callable(d.detect)
    assert get_detector("llm").base is None
