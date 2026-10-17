"""Tests for the synthetic data layer: label offsets, checksums, determinism, splits, leakage."""

import json
import random
import re

import pytest

from callsearch.data.generator import generate_calls
from callsearch.data.templates import TEST_TEMPLATES, TRAIN_TEMPLATES
from callsearch.validators import luhn_ok, make_luhn_number, make_non_luhn_number, make_sin, sin_ok

N, SEED = 120, 7
DIGITS = re.compile(r"\D")


@pytest.fixture(scope="module")
def calls():
    try:
        return generate_calls(n=N, seed=SEED)
    except OSError as e:  # MultiWOZ missing and no network
        pytest.skip(f"MultiWOZ data unavailable: {e}")


# --- validators ---------------------------------------------------------------

def test_luhn_known_values():
    assert luhn_ok("4539 1488 0343 6467")
    assert luhn_ok("4539-1488-0343-6467")
    assert not luhn_ok("4539 1488 0343 6468")
    assert not luhn_ok("abc")


def test_sin_known_values():
    assert sin_ok("130-692-544")
    assert sin_ok("130 692 544")
    assert not sin_ok("046454286")  # Luhn-valid but starts with 0
    assert not sin_ok("130-692-545")


def test_generators_respect_checksums():
    rng = random.Random(0)
    for _ in range(500):
        assert luhn_ok(make_luhn_number("4", 16, rng))
        assert sin_ok(make_sin(rng))
        assert not luhn_ok(make_non_luhn_number(16, rng))
        assert not sin_ok(make_non_luhn_number(9, rng))


# --- generated calls ----------------------------------------------------------

def test_label_offsets_match_values(calls):
    for c in calls:
        for span in c["labels"] + c["distractors"]:
            text = c["turns"][span["turn_id"]]["text"]
            assert text[span["start"]:span["end"]] == span["value"], (c["call_id"], span)


def test_turn_ids_sequential(calls):
    for c in calls:
        assert [t["turn_id"] for t in c["turns"]] == list(range(len(c["turns"])))
        assert all(t["speaker"] in ("agent", "customer") for t in c["turns"])


def test_checksums(calls):
    for c in calls:
        for l in c["labels"]:
            digits = DIGITS.sub("", l["value"])
            if l["type"] == "CARD":
                assert len(digits) == 16 and luhn_ok(l["value"]) and digits[0] in "45"
            elif l["type"] == "SIN":
                assert sin_ok(l["value"])
            elif l["type"] == "ACCOUNT":
                assert 10 <= len(digits) <= 12
            elif l["type"] == "PHONE":
                assert digits[-10:-7] in {"604", "778", "236", "250", "416", "647", "403"}
        for d in c["distractors"]:
            if d["type"] == "ORDER_NUMBER":
                assert len(DIGITS.sub("", d["value"])) == 16 and not luhn_ok(d["value"])
            elif d["type"] == "REFERENCE":
                assert len(DIGITS.sub("", d["value"])) == 9 and not sin_ok(d["value"])


def test_account_has_context_word(calls):
    for c in calls:
        for l in c["labels"]:
            if l["type"] == "ACCOUNT":
                before = c["turns"][l["turn_id"]]["text"][:l["start"]].lower()
                assert "account" in before[-30:]


def test_all_types_present(calls):
    for split in ("train", "test"):
        types = {l["type"] for c in calls if c["split"] == split for l in c["labels"]}
        assert types == {"PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"}, split


def test_same_seed_identical(calls):
    again = generate_calls(n=N, seed=SEED)
    assert json.dumps(again) == json.dumps(calls)
    assert json.dumps(generate_calls(n=N, seed=SEED + 1)) != json.dumps(calls)


def test_splits_disjoint(calls):
    by_split = {s: {(c["source"], c["source_id"]) for c in calls if c["split"] == s} for s in ("train", "test")}
    assert not by_split["train"] & by_split["test"]
    assert {c["template"] for c in calls if c["split"] == "train"} - {None} <= {t.id for t in TRAIN_TEMPLATES}
    assert {c["template"] for c in calls if c["split"] == "test"} - {None} <= {t.id for t in TEST_TEMPLATES}
    mw_ids = [c["source_id"] for c in calls if c["source"] == "multiwoz"]
    assert len(mw_ids) == len(set(mw_ids))  # each dialogue used at most once


def test_split_and_source_mix(calls):
    n_test = sum(c["split"] == "test" for c in calls)
    n_mw = sum(c["source"] == "multiwoz" for c in calls)
    assert abs(n_test / N - 0.3) < 0.02
    assert abs(n_mw / N - 0.6) < 0.02


UNLABELED_PII = re.compile(
    r"\S+@\S+"                          # email
    r"|\d(?:[\s\-.()]*\d){6,}"          # 7+ digits: phone, card, SIN, account
)


def test_no_unlabeled_pii_like_text(calls):
    for c in calls:
        masked = [list(t["text"]) for t in c["turns"]]
        for span in c["labels"] + c["distractors"]:
            masked[span["turn_id"]][span["start"]:span["end"]] = "#" * (span["end"] - span["start"])
        for t, chars in zip(c["turns"], masked):
            m = UNLABELED_PII.search("".join(chars))
            assert m is None, (c["call_id"], t["turn_id"], m and m.group(0), t["text"])


def test_repeated_values_always_labeled(calls):
    """A labeled non-name value must be labeled everywhere it appears in the call."""
    for c in calls:
        for l in c["labels"]:
            if l["type"] == "PERSON":
                continue
            covered = {(x["turn_id"], x["start"]) for x in c["labels"] if x["value"] == l["value"]}
            for t in c["turns"]:
                for m in re.finditer(re.escape(l["value"]), t["text"]):
                    assert (t["turn_id"], m.start()) in covered, (c["call_id"], l["value"])
