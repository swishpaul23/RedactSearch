"""Tests for RegexDetector, overlap resolution and redact()."""

import pytest

from callsearch.redact.base import Span, get_detector
from callsearch.redact.merge import redact, resolve_overlaps
from callsearch.redact.regex import RegexDetector

det = RegexDetector()


def found(text):
    return [(text[s.start:s.end], s.type) for s in det.detect(text)]


@pytest.mark.parametrize("card", ["4539 1488 0343 6467", "4539-1488-0343-6467", "4539148803436467"])
def test_luhn_card_detected(card):
    assert found(f"my card is {card}.") == [(card, "CARD")]


def test_non_luhn_16_digits_not_card():
    assert found("order 4539 1488 0343 6468 shipped") == []
    assert found("order 4539148803436468") == []


def test_card_inside_longer_run_not_detected():
    assert found("id 94539148803436467") == []  # 17-digit run, not Luhn


@pytest.mark.parametrize("sin", ["130-692-544", "130 692 544", "130692544"])
def test_valid_sin_detected(sin):
    assert found(f"My SIN is {sin}") == [(sin, "SIN")]


@pytest.mark.parametrize("bad", ["130-692-545", "046-454-286", "830 692 544"])
def test_invalid_sin_not_detected(bad):
    assert found(f"ref {bad}") == []


def test_account_needs_context():
    assert found("the account number is 4467065627.") == [("4467065627", "ACCOUNT")]
    assert found("Account no. 99812734501") == [("99812734501", "ACCOUNT")]
    assert found("acct #12345678") == [("12345678", "ACCOUNT")]
    assert found("it's account 123456789012") == [("123456789012", "ACCOUNT")]
    assert found("ticket 12345678 is open") == []


@pytest.mark.parametrize("phone", ["604-555-0182", "(604) 555-0182", "604.555.0182", "604 555 0182",
                                   "6045550182", "+1 604 555 0182", "+1-604-555-0182"])
def test_phone_formats(phone):
    assert found(f"call me at {phone} thanks") == [(phone, "PHONE")]


def test_email():
    assert found("send it to jane.doe+x@mail.example.ca.") == [("jane.doe+x@mail.example.ca", "EMAIL")]


def test_card_not_also_sin_or_phone():
    spans = det.detect("card 4539148803436467 and 4539 1488 0343 6467")
    assert [s.type for s in spans] == ["CARD", "CARD"]


def test_resolve_overlaps_priority_and_length():
    spans = [Span(0, 10, "PHONE", "x"), Span(2, 12, "CARD", "x"), Span(20, 25, "PHONE", "x"),
             Span(20, 28, "PHONE", "x")]
    assert resolve_overlaps(spans) == [Span(2, 12, "CARD", "x"), Span(20, 28, "PHONE", "x")]


def test_redact_multiple_spans():
    text = "Card 4539 1488 0343 6467, email a@b.com, phone 604-555-0182."
    assert redact(text, det.detect(text)) == "Card [CARD], email [EMAIL], phone [PHONE]."


def test_redact_unions_leftover_overlaps():
    assert redact("abcdefghij", [Span(2, 6, "PHONE", "x"), Span(4, 8, "CARD", "x")]) == "ab[CARD]ij"


def test_registry():
    assert get_detector("regex").name == "regex"
    with pytest.raises(KeyError):
        get_detector("nope")
