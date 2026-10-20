"""Download, load, filter and reshape MultiWOZ 2.2 dialogues into agent/customer calls."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

# MultiWOZ (MIT License, (c) 2019 Pawel Budzianowski): https://github.com/budzianowski/multiwoz
BASE_URL = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2"
# Official MultiWOZ splits back our splits, so train/dev/test dialogues can never overlap.
SPLIT_FILES = {
    "train": "train/dialogues_001.json",
    "dev": "dev/dialogues_001.json",
    "test": "test/dialogues_001.json",
}
DEFAULT_DIR = Path("data/raw/multiwoz")

SPEAKERS = {"USER": "customer", "SYSTEM": "agent"}

# Existing venue phones/emails/long references are unlabeled and would count as false
# positives, so any dialogue containing them is dropped (v1).
_UNSAFE = re.compile(r"\d(?:[\s\-.()]*\d){6,}|@|\bwww\.|https?://")


def ensure_downloaded(root: Path = DEFAULT_DIR) -> None:
    for rel in SPLIT_FILES.values():
        dest = root / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MultiWOZ 2.2 {rel} -> {dest}")
        urllib.request.urlretrieve(f"{BASE_URL}/{rel}", dest)


def is_clean(turns: list[tuple[str, str]]) -> bool:
    return not any(_UNSAFE.search(text) for _, text in turns)


def load_split(split: str, root: Path = DEFAULT_DIR, download: bool = True) -> list[dict]:
    """Clean dialogues for one split as [{"dialogue_id", "turns": [(speaker, text), ...]}]."""
    if download:
        ensure_downloaded(root)
    raw = json.loads((root / SPLIT_FILES[split]).read_text(encoding="utf-8"))
    out = []
    for d in raw:
        turns = [(SPEAKERS[t["speaker"]], " ".join(t["utterance"].split())) for t in d["turns"]]
        turns = [(s, t) for s, t in turns if t]
        if turns and is_clean(turns):
            out.append({"dialogue_id": d["dialogue_id"], "turns": turns})
    out.sort(key=lambda d: d["dialogue_id"])
    return out
