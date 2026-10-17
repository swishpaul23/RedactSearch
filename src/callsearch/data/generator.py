"""Generate synthetic call transcripts with ground-truth PII span labels into data/raw/.

Two sources: MultiWOZ 2.2 dialogues reshaped into calls with PII injected into customer
turns, and finance templates. Every PII value is inserted via a {TOKEN} slot, so its exact
character offsets are known. Distractors (number-like non-PII) are recorded separately.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from callsearch.data import multiwoz
from callsearch.data.templates import INJECTIONS, TEST_TEMPLATES, TRAIN_TEMPLATES, Template
from callsearch.data.values import ValueMaker

SLOT = re.compile(r"\{([A-Z]+)(\d*)\}")
PII_KINDS = {"PERSON": "PERSON", "FIRST": "PERSON", "AGENT": "PERSON", "CARD": "CARD",
             "SIN": "SIN", "PHONE": "PHONE", "EMAIL": "EMAIL", "ACCOUNT": "ACCOUNT"}
DISTRACTOR_KINDS = {"ORDER": "ORDER_NUMBER", "REF": "REFERENCE", "AMOUNT": "AMOUNT",
                    "DATE": "DATE", "TIME": "TIME"}
TEMPLATES = {"train": TRAIN_TEMPLATES, "test": TEST_TEMPLATES}

# Agent turns that ask about a booking are natural points for the customer to give details.
_BOOKING = re.compile(r"\b(book|reserv|name|confirm)", re.I)
# Injection probability per PII type for MultiWOZ calls.
_INJECT_P = (("PERSON", 0.9), ("PHONE", 0.6), ("EMAIL", 0.5), ("CARD", 0.5))


class CallBuilder:
    """Accumulates turns, labels and distractors; caches slot values per call."""

    def __init__(self, vm: ValueMaker):
        self.vm = vm
        self.turns: list[dict] = []
        self.labels: list[dict] = []
        self.distractors: list[dict] = []
        self._cache: dict[str, str] = {}
        self._first, self._last = vm.person()

    def _make(self, kind: str) -> str:
        vm = self.vm
        return {
            "PERSON": lambda: f"{self._first} {self._last}",
            "FIRST": lambda: self._first,
            "AGENT": vm.first_name,
            "EMAIL": lambda: vm.email(self._first, self._last),
            "PHONE": vm.phone, "CARD": vm.card, "SIN": vm.sin, "ACCOUNT": vm.account,
            "ORDER": vm.order_number, "REF": vm.reference_number,
            "AMOUNT": vm.amount, "DATE": vm.date, "TIME": vm.time,
        }[kind]()

    def _fill(self, template: str, turn_id: int, offset: int) -> str:
        out, pos = [], 0
        for m in SLOT.finditer(template):
            out.append(template[pos:m.start()])
            kind, key = m.group(1), m.group(0)
            if key not in self._cache:
                self._cache[key] = self._make(kind)
            value = self._cache[key]
            start = offset + sum(map(len, out))
            span = {"turn_id": turn_id, "start": start, "end": start + len(value)}
            if kind in PII_KINDS:
                self.labels.append({**span, "type": PII_KINDS[kind], "value": value})
            else:
                self.distractors.append({**span, "type": DISTRACTOR_KINDS[kind], "value": value})
            out.append(value)
            pos = m.end()
        out.append(template[pos:])
        return "".join(out)

    def add_turn(self, speaker: str, template: str) -> None:
        turn_id = len(self.turns)
        self.turns.append({"turn_id": turn_id, "speaker": speaker, "text": self._fill(template, turn_id, 0)})

    def add_raw_turn(self, speaker: str, text: str) -> None:
        """Add source text verbatim (no slot filling)."""
        self.turns.append({"turn_id": len(self.turns), "speaker": speaker, "text": text})

    def append(self, turn_id: int, template: str) -> None:
        """Append a filled sentence to the end of an existing turn (earlier offsets unchanged)."""
        base = self.turns[turn_id]["text"]
        if base and base[-1] not in ".!?":
            base += "."
        base += " "
        self.turns[turn_id]["text"] = base + self._fill(template, turn_id, len(base))

    def finish(self) -> dict:
        t = 0.0
        for turn in self.turns:
            turn["start_sec"] = round(t, 1)
            t += 0.35 * len(turn["text"].split()) + self.vm.rng.uniform(0.3, 1.2)
        key = lambda s: (s["turn_id"], s["start"])
        return {
            "turns": [{k: tr[k] for k in ("turn_id", "speaker", "start_sec", "text")} for tr in self.turns],
            "labels": sorted(self.labels, key=key),
            "distractors": sorted(self.distractors, key=key),
        }


def build_template_call(template: Template, vm: ValueMaker) -> dict:
    b = CallBuilder(vm)
    for speaker, line in template.lines:
        b.add_turn(speaker, line)
    return b.finish()


def build_multiwoz_call(dialogue: dict, split: str, vm: ValueMaker) -> dict:
    rng, phrases = vm.rng, INJECTIONS[split]
    b = CallBuilder(vm)
    b.add_turn("agent", rng.choice(phrases["GREETING"]))
    for speaker, text in dialogue["turns"]:
        b.add_raw_turn(speaker, text)

    turns = b.turns
    customer = [t["turn_id"] for t in turns if t["speaker"] == "customer"]
    agent = [t["turn_id"] for t in turns if t["speaker"] == "agent" and t["turn_id"] > 0]
    after_booking = [i for i in customer if _BOOKING.search(turns[i - 1]["text"])]
    preferred = {"PERSON": after_booking, "CARD": after_booking, "PHONE": customer[-3:], "EMAIL": customer[-3:]}

    kinds = [k for k, p in _INJECT_P if rng.random() < p] or ["PERSON"]
    used: set[int] = set()
    for kind in kinds:
        pool = [i for i in preferred[kind] if i not in used] or [i for i in customer if i not in used] or customer
        i = rng.choice(pool)
        used.add(i)
        b.append(i, rng.choice(phrases[kind]))

    if rng.random() < 0.4:
        b.append(rng.choice(customer), rng.choice(phrases["DISTRACTOR_CUSTOMER"]))
    if agent and rng.random() < 0.5:
        booked = [i for i in agent if re.search(r"\bbook", turns[i]["text"], re.I)
                  and "reference" not in turns[i]["text"].lower()]
        b.append(rng.choice(booked or agent), rng.choice(phrases["DISTRACTOR_AGENT"]))
    return b.finish()


def generate_calls(n: int = 200, seed: int = 42, multiwoz_frac: float = 0.6, test_frac: float = 0.3,
                   multiwoz_root: Path = multiwoz.DEFAULT_DIR, download: bool = True) -> list[dict]:
    """Deterministic for a given seed. Train and test draw from disjoint dialogue/template pools."""
    vm = ValueMaker(seed)
    rng = vm.rng
    n_test = round(n * test_frac)
    plan: list[tuple[str, str, object]] = []
    for split, k in (("train", n - n_test), ("test", n_test)):
        n_mw = round(k * multiwoz_frac)
        if n_mw:
            pool = multiwoz.load_split(split, multiwoz_root, download=download)
            if len(pool) < n_mw:
                raise ValueError(f"only {len(pool)} clean MultiWOZ {split} dialogues, need {n_mw}")
            plan += [(split, "multiwoz", d) for d in rng.sample(pool, n_mw)]
        templates = TEMPLATES[split]
        plan += [(split, "template", templates[i % len(templates)]) for i in range(k - n_mw)]
    rng.shuffle(plan)

    calls = []
    for idx, (split, source, item) in enumerate(plan, start=1):
        if source == "multiwoz":
            body, template_id, source_id = build_multiwoz_call(item, split, vm), None, item["dialogue_id"]
        else:
            body, template_id, source_id = build_template_call(item, vm), item.id, item.id
        calls.append({"call_id": f"call_{idx:04d}", "split": split, "source": source,
                      "template": template_id, "source_id": source_id, **body})
    return calls


def write_jsonl(calls: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for call in calls:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
