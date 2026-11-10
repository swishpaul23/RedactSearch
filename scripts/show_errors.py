"""CLI: print false positives / false negatives with context, for tuning on the TRAIN split.

python scripts/show_errors.py --detector pipeline --split dev --type PERSON --n 10

Prints transcript text (synthetic data only). Kept out of redaction_eval.py, which
prints metrics only. Tune on dev; test is refused unless --allow-test is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from callsearch.data.generator import read_jsonl
from callsearch.eval.redaction_eval import match_spans
from callsearch.redact.base import available_detectors, get_detector

CONTEXT = 30


def snippet(text: str, start: int, end: int) -> str:
    left = text[max(0, start - CONTEXT):start]
    right = text[end:end + CONTEXT]
    return f"{'...' if start > CONTEXT else ''}{left}[[{text[start:end]}]]{right}{'...' if end + CONTEXT < len(text) else ''}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--detector", default="pipeline", choices=available_detectors())
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    p.add_argument("--type", help="only show errors of this PII type, e.g. PERSON")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--allow-test", action="store_true", help="permit inspecting test errors (don't tune on them)")
    args = p.parse_args()
    if args.split in ("test", "all"):
        if not args.allow_test:
            p.error(f"--split {args.split} includes test calls; tune on dev. Pass --allow-test to override.")
        print(f"WARNING: showing {args.split} errors, which include test calls. "
              "Do not tune on anything you see here.\n")

    det = get_detector(args.detector)
    fps, fns = [], []
    for call in read_jsonl(args.data):
        if args.split != "all" and call["split"] != args.split:
            continue
        for turn in call["turns"]:
            golds = [l for l in call["labels"] if l["turn_id"] == turn["turn_id"]]
            _, fp, fn = match_spans(det.detect(turn["text"]), golds)
            where = f"{call['call_id']} t{turn['turn_id']} ({call['source']})"
            fps += [(where, s.type, snippet(turn["text"], s.start, s.end)) for s in fp
                    if not args.type or s.type == args.type]
            fns += [(where, g["type"], snippet(turn["text"], g["start"], g["end"])) for g in fn
                    if not args.type or g["type"] == args.type]

    for title, errs in (("FALSE POSITIVES (predicted, not labeled)", fps), ("FALSE NEGATIVES (labeled, missed)", fns)):
        print(f"{title}: {len(errs)} total, showing {min(len(errs), args.n)}")
        for where, typ, snip in errs[:args.n]:
            print(f"  {where:<34} {typ:<8} {snip}")
        print()


if __name__ == "__main__":
    main()
