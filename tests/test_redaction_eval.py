"""Tests for redaction_eval metrics on a tiny hand-made fixture."""

import pytest

from callsearch.eval.redaction_eval import evaluate
from callsearch.redact.base import Span

TEXT = "Card 4539 1488 0343 6467, call 604-555-0182, order 1111 2222 3333 4444, I'm Ann Lee."


def span(sub, typ):
    i = TEXT.index(sub)
    return {"turn_id": 0, "start": i, "end": i + len(sub), "type": typ, "value": sub}


CALLS = [{
    "call_id": "c1", "split": "test",
    "turns": [{"turn_id": 0, "speaker": "customer", "start_sec": 0.0, "text": TEXT}],
    "labels": [span("4539 1488 0343 6467", "CARD"), span("604-555-0182", "PHONE"), span("Ann Lee", "PERSON")],
    "distractors": [span("1111 2222 3333 4444", "ORDER_NUMBER")],
}]


class FakeDetector:
    """CARD exact, PHONE partial (overlap but not exact), one FP on the order number, PERSON missed."""
    name = "fake"

    def detect(self, text):
        c, p, o = (TEXT.index(s) for s in ("4539", "604", "1111"))
        return [Span(c, c + 19, "CARD", "fake"), Span(p, p + 7, "PHONE", "fake"), Span(o, o + 19, "CARD", "fake")]


def test_metrics_on_fixture():
    res = evaluate(CALLS, FakeDetector())
    card, phone, person = (res["per_type"][t] for t in ("CARD", "PHONE", "PERSON"))

    assert (card["tp"], card["fp"], card["fn"]) == (1, 1, 0)
    assert card["precision"] == 0.5 and card["recall"] == 1.0 and card["exact_recall"] == 1.0
    assert (phone["tp"], phone["fp"], phone["fn"]) == (1, 0, 0)
    assert phone["recall"] == 1.0 and phone["exact_recall"] == 0.0  # overlap counts, exact does not
    assert person["recall"] == 0.0 and person["support"] == 1

    m = res["micro"]
    assert (m["tp"], m["fp"], m["fn"], m["support"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)


def test_leaks_and_distractors():
    res = evaluate(CALLS, FakeDetector())
    assert res["per_type"]["CARD"]["leaked"] == 0
    assert res["per_type"]["PHONE"]["leaked"] == 0  # partial redaction still breaks the verbatim value
    assert res["per_type"]["PERSON"]["leaked"] == 1
    assert res["distractor_fp"] == {"ORDER_NUMBER": {"flagged": 1, "total": 1}}


def test_one_to_one_matching():
    calls = [{**CALLS[0], "labels": [span("4539 1488 0343 6467", "CARD")], "distractors": []}]

    class Twice:
        name = "twice"

        def detect(self, text):
            c = TEXT.index("4539")
            return [Span(c, c + 9, "CARD", "t"), Span(c + 10, c + 19, "CARD", "t")]

    card = evaluate(calls, Twice())["per_type"]["CARD"]
    assert (card["tp"], card["fp"], card["fn"]) == (1, 1, 0)
