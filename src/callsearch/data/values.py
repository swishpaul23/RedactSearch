"""Fake PII and distractor value makers driven by a seeded RNG (all values synthetic)."""

from __future__ import annotations

import random
import unicodedata

from faker import Faker

from callsearch.validators import make_luhn_number, make_non_luhn_number, make_sin

AREA_CODES = ["604", "778", "236", "250", "416", "647", "403"]
CARD_PREFIXES = ["4", "51", "52", "53", "54", "55"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _group(digits: str, sizes: list[int], sep: str) -> str:
    parts, i = [], 0
    for n in sizes:
        parts.append(digits[i:i + n])
        i += n
    return sep.join(parts)


class ValueMaker:
    """Makes fake values. All randomness flows from one seed, so output is reproducible."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.faker = Faker("en_CA")
        self.faker.seed_instance(seed)

    # --- PII -----------------------------------------------------------------

    def person(self) -> tuple[str, str]:
        return self.faker.first_name(), self.faker.last_name()

    def first_name(self) -> str:
        return self.faker.first_name()

    def email(self, first: str, last: str) -> str:
        local = self.rng.choice([
            f"{_ascii(first)}.{_ascii(last)}",
            f"{_ascii(first)[0]}{_ascii(last)}",
            f"{_ascii(first)}{self.rng.randrange(10, 99)}",
            f"{_ascii(last)}.{_ascii(first)[0]}",
        ])
        local = "".join(c for c in local if c.isalnum() or c == ".")
        return f"{local}@{self.faker.free_email_domain()}"

    def phone(self) -> str:
        area = self.rng.choice(AREA_CODES)
        exch = f"{self.rng.randrange(2, 10)}{self.rng.randrange(10)}{self.rng.randrange(10)}"
        line = f"{self.rng.randrange(10000):04d}"
        fmt = self.rng.choice([
            "({a}) {e}-{l}", "{a}-{e}-{l}", "{a}.{e}.{l}", "{a} {e} {l}", "{a}{e}{l}", "+1 {a} {e} {l}",
        ])
        return fmt.format(a=area, e=exch, l=line)

    def card(self) -> str:
        digits = make_luhn_number(self.rng.choice(CARD_PREFIXES), 16, self.rng)
        return _group(digits, [4, 4, 4, 4], self.rng.choice([" ", "-", ""]))

    def sin(self) -> str:
        return _group(make_sin(self.rng), [3, 3, 3], self.rng.choice(["-", " ", ""]))

    def account(self) -> str:
        n = self.rng.randrange(10, 13)
        return str(self.rng.randrange(1, 10)) + "".join(str(self.rng.randrange(10)) for _ in range(n - 1))

    # --- Distractors (look number-like, must NOT be flagged as PII) -----------

    def order_number(self) -> str:
        """16 digits that fail Luhn; sometimes starts with 4/5 to look card-like."""
        prefix = self.rng.choice(["", "", "4", "5"])
        digits = make_non_luhn_number(16, self.rng, prefix=prefix)
        return _group(digits, [4, 4, 4, 4], self.rng.choice([" ", "-", ""]))

    def reference_number(self) -> str:
        """9 digits that fail Luhn (and therefore fail the SIN check)."""
        digits = make_non_luhn_number(9, self.rng)
        return _group(digits, [3, 3, 3], self.rng.choice(["-", " ", ""]))

    def amount(self) -> str:
        dollars = self.rng.choice([self.rng.randrange(5, 100), self.rng.randrange(100, 5000)])
        cents = self.rng.choice([0, self.rng.randrange(1, 100)])
        if self.rng.random() < 0.2:
            return f"{dollars} dollars"
        return f"${dollars:,}" if cents == 0 else f"${dollars:,}.{cents:02d}"

    def date(self) -> str:
        m, d = self.rng.randrange(1, 13), self.rng.randrange(1, 29)
        return self.rng.choice([
            f"{MONTHS[m - 1]} {d}", f"{MONTHS[m - 1]} {d}, 2026", f"2026-{m:02d}-{d:02d}", f"the {d}{_ordinal(d)}",
        ])

    def time(self) -> str:
        h, mins = self.rng.randrange(8, 20), self.rng.choice([0, 15, 30, 45])
        if self.rng.random() < 0.5:
            return f"{h:02d}:{mins:02d}"
        h12 = h - 12 if h > 12 else h
        return f"{h12}:{mins:02d} {'pm' if h >= 12 else 'am'}"


def _ordinal(d: int) -> str:
    if 11 <= d <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
