"""CLI: measure how much phrasing dev/test share with train (metrics only, no text).

python scripts/split_overlap.py [--data data/raw/calls.jsonl]

- PERSON left context: the 3 words before each PERSON label in its turn (lowercased,
  punctuation and fillers dropped, padded with <s>). Reports the share of dev/test
  labels whose context also occurs anywhere in train.
- Templates and injection phrases: share of dev/test lines that also appear in train
  (should be 0).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from callsearch.data.generator import TEMPLATES, read_jsonl
from callsearch.data.templates import INJECTIONS

WORD = re.compile(r"[\w']+")
FILLERS = {"um", "uh"}


def words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in FILLERS]


def left_context(text: str, start: int, k: int = 3) -> tuple[str, ...]:
    ws = words(text[:start])[-k:]
    return tuple(["<s>"] * (k - len(ws)) + ws)


def norm_line(line: str) -> str:
    return " ".join(words(line))


def share(items: list, reference: set) -> float:
    return sum(i in reference for i in items) / len(items) if items else 0.0


def person_contexts(calls: list[dict], split: str) -> list[tuple[str, ...]]:
    return [left_context(c["turns"][l["turn_id"]]["text"], l["start"])
            for c in calls if c["split"] == split for l in c["labels"] if l["type"] == "PERSON"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    args = p.parse_args()
    calls = read_jsonl(args.data)

    train_ctx = set(person_contexts(calls, "train"))
    train_names = {l["value"].lower() for c in calls if c["split"] == "train" for l in c["labels"]
                   if l["type"] == "PERSON"}
    train_tpl = {norm_line(t) for tpl in TEMPLATES["train"] for _, t in tpl.lines}
    train_inj = {norm_line(s) for pool in INJECTIONS["train"].values() for s in pool}

    print(f"{'split':<6}{'PERSON labels':>14}{'ctx in train':>14}{'uniq ctx in train':>19}"
          f"{'name in train':>15}{'tmpl lines in train':>21}{'inj lines in train':>20}")
    for split in ("dev", "test"):
        ctx = person_contexts(calls, split)
        names = [l["value"].lower() for c in calls if c["split"] == split for l in c["labels"]
                 if l["type"] == "PERSON"]
        tpl = [norm_line(t) for tpl in TEMPLATES[split] for _, t in tpl.lines]
        inj = [norm_line(s) for pool in INJECTIONS[split].values() for s in pool]
        print(f"{split:<6}{len(ctx):>14}{share(ctx, train_ctx):>14.1%}{share(sorted(set(ctx)), train_ctx):>19.1%}"
              f"{share(names, train_names):>15.1%}{share(tpl, train_tpl):>21.1%}{share(inj, train_inj):>20.1%}")


if __name__ == "__main__":
    main()
