"""Checksum validators and generators for card numbers (Luhn) and Canadian SINs."""

from __future__ import annotations

import random
import re

_SEPARATORS = re.compile(r"[\s\-]")


def _digits(number: str) -> str | None:
    """Strip spaces/dashes; return None if anything other than digits remains."""
    stripped = _SEPARATORS.sub("", number)
    return stripped if stripped.isdigit() else None


def _luhn_check_digit(payload: str) -> int:
    """Check digit that makes payload + digit Luhn-valid."""
    total = 0
    # Walk right-to-left; the check digit will sit to the right, so double payload[-1].
    for i, ch in enumerate(reversed(payload)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def luhn_ok(number: str) -> bool:
    """True if number (spaces/dashes allowed) passes the Luhn checksum."""
    digits = _digits(number)
    if not digits or len(digits) < 2:
        return False
    return _luhn_check_digit(digits[:-1]) == int(digits[-1])


def make_luhn_number(prefix: str, length: int, rng: random.Random) -> str:
    """Random Luhn-valid digit string of the given length starting with prefix."""
    if len(prefix) >= length:
        raise ValueError("prefix must be shorter than length")
    body = prefix + "".join(str(rng.randrange(10)) for _ in range(length - len(prefix) - 1))
    return body + str(_luhn_check_digit(body))


def make_non_luhn_number(length: int, rng: random.Random, prefix: str = "") -> str:
    """Random digit string of the given length that is guaranteed to FAIL Luhn."""
    if len(prefix) >= length:
        raise ValueError("prefix must be shorter than length")
    first = prefix or str(rng.randrange(1, 10))
    body = first + "".join(str(rng.randrange(10)) for _ in range(length - len(first) - 1))
    good = _luhn_check_digit(body)
    bad = (good + rng.randrange(1, 10)) % 10
    return body + str(bad)


def sin_ok(sin: str) -> bool:
    """True if sin is a structurally valid Canadian SIN: 9 digits, first not 0/8, Luhn-valid."""
    digits = _digits(sin)
    if not digits or len(digits) != 9 or digits[0] in "08":
        return False
    return luhn_ok(digits)


def make_sin(rng: random.Random) -> str:
    """Random valid SIN as 9 bare digits."""
    return make_luhn_number(rng.choice("1234567" "9"), 9, rng)
