"""Generate synthetic call transcripts with ground-truth PII span labels into data/raw/.

Two sources: MultiWOZ 2.2 dialogues reshaped into calls with PII injected into customer
turns, and finance templates. Splits are train / dev / test, disjoint in dialogues,
templates, injection phrases and name lists.

Each turn is built from pieces (literal text or slot values) and only rendered at the
end, so exact label offsets survive the optional transcript-noise transform.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from callsearch.data import multiwoz
from callsearch.data.templates import DEV_TEMPLATES, INJECTIONS, TEST_TEMPLATES, TRAIN_TEMPLATES, Template
from callsearch.data.values import ValueMaker

SPLITS = ("train", "dev", "test")
SLOT = re.compile(r"\{([A-Z]+)(\d*)\}")
PII_KINDS = {"PERSON": "PERSON", "FIRST": "PERSON", "LAST": "PERSON", "AGENT": "PERSON", "CARD": "CARD",
             "SIN": "SIN", "PHONE": "PHONE", "EMAIL": "EMAIL", "ACCOUNT": "ACCOUNT"}
DISTRACTOR_KINDS = {"ORDER": "ORDER_NUMBER", "REF": "REFERENCE", "AMOUNT": "AMOUNT",
                    "DATE": "DATE", "TIME": "TIME"}
TEMPLATES = {"train": TRAIN_TEMPLATES, "dev": DEV_TEMPLATES, "test": TEST_TEMPLATES}

# Agent turns that ask about a booking are natural points for the customer to give details.
_BOOKING = re.compile(r"\b(book|reserv|name|confirm)", re.I)
# Injection probability per PII type for MultiWOZ calls.
_INJECT_P = (("PERSON", 0.9), ("PHONE", 0.6), ("EMAIL", 0.5), ("CARD", 0.5))

# Transcript noise: drop punctuation (but not inside numbers like 14:15), add fillers.
_NOISE_PUNCT = re.compile(r'[.,!?;:"](?!\d)')
_WORD_START = re.compile(r"(?<= )(?=\w)")
FILLERS = ["um ", "uh "]
FILLER_P = 0.05


class CallBuilder:
    """Accumulates turns as pieces; caches slot values per call."""

    def __init__(self, vm: ValueMaker, split: str):
        self.vm = vm
        # turn = {"speaker", "pieces": [(text, tag)]}; tag is None for literal text,
        # ("label", TYPE) for PII or ("distractor", TYPE) for distractors.
        self.turns: list[dict] = []
        self._cache: dict[str, str] = {}
        self._first, self._last, self._title = vm.person(split)

    def _make(self, kind: str) -> str:
        vm = self.vm
        return {
            "PERSON": lambda: f"{self._first} {self._last}",
            "FIRST": lambda: self._first,
            "LAST": lambda: self._last,
            "TITLE": lambda: self._title,
            "AGENT": vm.first_name,
            "EMAIL": lambda: vm.email(self._first, self._last),
            "PHONE": vm.phone, "CARD": vm.card, "SIN": vm.sin, "ACCOUNT": vm.account,
            "ORDER": vm.order_number, "REF": vm.reference_number,
            "AMOUNT": vm.amount, "DATE": vm.date, "TIME": vm.time,
        }[kind]()

    def _pieces(self, template: str) -> list[tuple[str, tuple | None]]:
        out, pos = [], 0
        for m in SLOT.finditer(template):
            if m.start() > pos:
                out.append((template[pos:m.start()], None))
            kind, key = m.group(1), m.group(0)
            if key not in self._cache:
                self._cache[key] = self._make(kind)
            if kind in PII_KINDS:
                tag = ("label", PII_KINDS[kind])
            elif kind in DISTRACTOR_KINDS:
                tag = ("distractor", DISTRACTOR_KINDS[kind])
            else:
                tag = None  # e.g. TITLE: plain text
            out.append((self._cache[key], tag))
            pos = m.end()
        if pos < len(template):
            out.append((template[pos:], None))
        return out

    def text(self, turn_id: int) -> str:
        return "".join(t for t, _ in self.turns[turn_id]["pieces"])

    def add_turn(self, speaker: str, template: str) -> None:
        self.turns.append({"speaker": speaker, "pieces": self._pieces(template)})

    def add_raw_turn(self, speaker: str, text: str) -> None:
        """Add source text verbatim (no slot filling)."""
        self.turns.append({"speaker": speaker, "pieces": [(text, None)]})

    def append(self, turn_id: int, template: str) -> None:
        """Append a filled sentence to the end of an existing turn."""
        pieces = self.turns[turn_id]["pieces"]
        text = self.text(turn_id)
        if text and text[-1] not in ".!?":
            pieces.append((".", None))
        pieces.append((" ", None))
        pieces += self._pieces(template)

    def prepend(self, turn_id: int, template: str) -> None:
        """Prepend a filled sentence to the start of an existing turn."""
        self.turns[turn_id]["pieces"] = self._pieces(template) + [(" ", None)] + self.turns[turn_id]["pieces"]

    def _noisy_literal(self, s: str) -> str:
        s = _NOISE_PUNCT.sub("", s.lower())
        return _WORD_START.sub(lambda m: self.vm.rng.choice(FILLERS) if self.vm.rng.random() < FILLER_P else "", s)

    def finish(self, noise: bool = False) -> dict:
        """Render pieces to text. With noise: literals lowercased, unpunctuated, with fillers;
        slot values lowercased only. Offsets are measured on the final text."""
        turns, labels, distractors, t = [], [], [], 0.0
        for turn_id, turn in enumerate(self.turns):
            text = ""
            for piece, tag in turn["pieces"]:
                if noise:
                    piece = piece.lower() if tag else self._noisy_literal(piece)
                if tag:
                    span = {"turn_id": turn_id, "start": len(text), "end": len(text) + len(piece),
                            "type": tag[1], "value": piece}
                    (labels if tag[0] == "label" else distractors).append(span)
                text += piece
            turns.append({"turn_id": turn_id, "speaker": turn["speaker"], "start_sec": round(t, 1), "text": text})
            t += 0.35 * len(text.split()) + self.vm.rng.uniform(0.3, 1.2)
        return {"noise": noise, "turns": turns, "labels": labels, "distractors": distractors}


def build_template_call(template: Template, vm: ValueMaker, split: str, noise: bool = False) -> dict:
    b = CallBuilder(vm, split)
    for speaker, line in template.lines:
        b.add_turn(speaker, line)
    return b.finish(noise)


def build_multiwoz_call(dialogue: dict, split: str, vm: ValueMaker, noise: bool = False) -> dict:
    rng, phrases = vm.rng, INJECTIONS[split]
    b = CallBuilder(vm, split)
    b.add_turn("agent", rng.choice(phrases["GREETING"]))
    for speaker, text in dialogue["turns"]:
        b.add_raw_turn(speaker, text)

    n = len(b.turns)
    customer = [i for i in range(n) if b.turns[i]["speaker"] == "customer"]
    agent = [i for i in range(1, n) if b.turns[i]["speaker"] == "agent"]
    after_booking = [i for i in customer if _BOOKING.search(b.text(i - 1))]
    preferred = {"PERSON": after_booking, "CARD": after_booking, "PHONE": customer[-3:], "EMAIL": customer[-3:]}

    kinds = [k for k, p in _INJECT_P if rng.random() < p] or ["PERSON"]
    used: set[int] = set()
    for kind in kinds:
        pool = [i for i in preferred[kind] if i not in used] or [i for i in customer if i not in used] or customer
        i = rng.choice(pool)
        used.add(i)
        b.append(i, rng.choice(phrases[kind]))
        # Sometimes the agent answers by addressing the customer by name.
        if kind == "PERSON" and i + 1 < n and b.turns[i + 1]["speaker"] == "agent" and rng.random() < 0.5:
            b.prepend(i + 1, rng.choice(phrases["AGENT_NAME"]))

    if rng.random() < 0.4:
        b.append(rng.choice(customer), rng.choice(phrases["DISTRACTOR_CUSTOMER"]))
    if agent and rng.random() < 0.5:
        booked = [i for i in agent if re.search(r"\bbook", b.text(i), re.I) and "reference" not in b.text(i).lower()]
        b.append(rng.choice(booked or agent), rng.choice(phrases["DISTRACTOR_AGENT"]))
    return b.finish(noise)


def generate_calls(n: int = 400, seed: int = 42, multiwoz_frac: float = 0.6, test_frac: float = 0.25,
                   dev_frac: float = 0.15, noise_frac: float = 0.25,
                   multiwoz_root: Path = multiwoz.DEFAULT_DIR, download: bool = True) -> list[dict]:
    """Deterministic for a given seed. Each split draws from its own dialogues/templates/phrases."""
    vm = ValueMaker(seed)
    rng = vm.rng
    n_test, n_dev = round(n * test_frac), round(n * dev_frac)
    sizes = {"train": n - n_test - n_dev, "dev": n_dev, "test": n_test}
    plan: list[tuple[str, str, object]] = []
    for split in SPLITS:
        k = sizes[split]
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
        noise = rng.random() < noise_frac
        if source == "multiwoz":
            body, template_id, source_id = build_multiwoz_call(item, split, vm, noise), None, item["dialogue_id"]
        else:
            body, template_id, source_id = build_template_call(item, vm, split, noise), item.id, item.id
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
