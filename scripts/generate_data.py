"""CLI: generate synthetic call transcripts into data/raw/."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from callsearch.data.generator import SPLITS, generate_calls, write_jsonl
from callsearch.data.multiwoz import DEFAULT_DIR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--multiwoz-frac", type=float, default=0.6)
    p.add_argument("--dev-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--noise-frac", type=float, default=0.25)
    p.add_argument("--multiwoz-dir", type=Path, default=DEFAULT_DIR)
    args = p.parse_args()

    calls = generate_calls(args.n, args.seed, multiwoz_frac=args.multiwoz_frac, test_frac=args.test_frac,
                           dev_frac=args.dev_frac, noise_frac=args.noise_frac, multiwoz_root=args.multiwoz_dir)
    write_jsonl(calls, args.out)

    print(f"Wrote {len(calls)} calls to {args.out}\n")
    types = ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]
    print(f"{'split':<6}{'calls':>6}{'mwoz':>6}{'tmpl':>6}{'noisy':>6}" + "".join(f"{t:>8}" for t in types))
    for split in SPLITS:
        cs = [c for c in calls if c["split"] == split]
        src = Counter(c["source"] for c in cs)
        pii = Counter(l["type"] for c in cs for l in c["labels"])
        print(f"{split:<6}{len(cs):>6}{src['multiwoz']:>6}{src['template']:>6}{sum(c['noise'] for c in cs):>6}"
              + "".join(f"{pii[t]:>8}" for t in types))
    print("\nDistractors:", dict(Counter(d["type"] for c in calls for d in c["distractors"]).most_common()))


if __name__ == "__main__":
    main()
