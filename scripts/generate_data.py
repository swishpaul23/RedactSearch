"""CLI: generate synthetic call transcripts into data/raw/."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from callsearch.data.generator import generate_calls, write_jsonl
from callsearch.data.multiwoz import DEFAULT_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--multiwoz-frac", type=float, default=0.6)
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--multiwoz-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    calls = generate_calls(args.n, args.seed, args.multiwoz_frac, args.test_frac, args.multiwoz_dir)
    write_jsonl(calls, args.out)

    print(f"Wrote {len(calls)} calls to {args.out}")
    for (split, source), c in sorted(Counter((c["split"], c["source"]) for c in calls).items()):
        print(f"  {split:5} {source:8} {c}")
    print("PII labels:", dict(Counter(l["type"] for c in calls for l in c["labels"]).most_common()))
    print("Distractors:", dict(Counter(d["type"] for c in calls for d in c["distractors"]).most_common()))


if __name__ == "__main__":
    main()
