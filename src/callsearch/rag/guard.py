"""Index guard: prove that no personal data reached the vector index.

Two checks:
1. Pattern scan (no raw data needed): the regex layer runs over every stored document.
   Any card, SIN, account, email or phone it finds is a failure. build_index.py runs this
   before anything is written.
2. Planted-value scan (evaluation only): every labeled value from data/raw/ is looked up
   verbatim in the stored text of its own turn. This is the same leak test redaction_eval
   uses, applied to what is actually in the index.

python -m callsearch.rag.guard [--allow-person-leaks]
    Runs both checks on the built index, saves counts to results/index_guard.json and
    exits 1 on failure. Structured leaks always fail. PERSON leaks fail unless allowed,
    because no name model is perfect; the count is saved either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from callsearch.redact.regex import RegexDetector

PII_TYPES = ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]
STRUCTURED = [t for t in PII_TYPES if t != "PERSON"]


def pattern_scan(texts: list[str]) -> dict[str, int]:
    """Structured PII the regex layer still finds in (already redacted) text."""
    det, hits = RegexDetector(), Counter()
    for t in texts:
        for s in det.detect(t):
            hits[s.type] += 1
    return {t: hits[t] for t in STRUCTURED}


def planted_scan(docs: dict[str, str], raw_calls: list[dict]) -> dict:
    """Planted values still verbatim in their turn's stored text. docs: chunk id -> text."""
    by_split: dict[str, Counter] = {}
    totals, support, missing_turns = Counter(), Counter(), 0
    for call in raw_calls:
        split = by_split.setdefault(call["split"], Counter())
        for label in call["labels"]:
            text = docs.get(f"{call['call_id']}:{label['turn_id']}")
            support[label["type"]] += 1
            if text is None:
                missing_turns += 1
                continue
            if label["value"] in text:
                totals[label["type"]] += 1
                split[label["type"]] += 1
    return {"leaks": {t: totals[t] for t in PII_TYPES}, "support": {t: support[t] for t in PII_TYPES},
            "leaks_by_split": {s: {t: c[t] for t in PII_TYPES} for s, c in sorted(by_split.items())},
            "labels_without_chunk": missing_turns}


def verdict(pattern_hits: dict[str, int], planted: dict | None, allow_person_leaks: bool) -> list[str]:
    """Reasons the guard fails; empty means it passes."""
    reasons = [f"pattern scan found {n} {t}" for t, n in pattern_hits.items() if n]
    if planted:
        reasons += [f"{n} planted {t} values in the index" for t, n in planted["leaks"].items()
                    if n and (t != "PERSON" or not allow_person_leaks)]
        if planted["labels_without_chunk"]:
            reasons.append(f"{planted['labels_without_chunk']} labeled turns have no chunk (index incomplete?)")
    return reasons


def main() -> None:
    from callsearch.data.generator import read_jsonl
    from callsearch.rag.store import DEFAULT_INDEX_DIR, CallIndex

    p = argparse.ArgumentParser(description="Scan the built index for personal data.")
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"), help="raw calls, for planted values")
    p.add_argument("--allow-person-leaks", action="store_true", help="report leaked names without failing")
    p.add_argument("--results", type=Path, default=Path("results/index_guard.json"))
    args = p.parse_args()

    index = CallIndex(args.index_dir)
    docs = dict(index.documents())
    pattern = pattern_scan(list(docs.values()))
    planted = planted_scan(docs, read_jsonl(args.data))
    reasons = verdict(pattern, planted, args.allow_person_leaks)
    info = index.info

    res = {"detector": info.get("detector"), "embed_model": info.get("embed_model"), "n_chunks": len(docs),
           "pattern_hits": pattern, "planted_leaks": planted["leaks"], "planted_support": planted["support"],
           "planted_leaks_by_split": planted["leaks_by_split"], "labels_without_chunk": planted["labels_without_chunk"],
           "allow_person_leaks": args.allow_person_leaks, "passed": not reasons}
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")

    print(f"INDEX GUARD  {len(docs)} chunks  detector {res['detector']}  embeddings {res['embed_model']}")
    print(f"{'type':<10}{'pattern':>9}{'planted':>9}{'support':>9}")
    for t in PII_TYPES:
        print(f"{t:<10}{pattern.get(t, 0) if t != 'PERSON' else '-':>9}{planted['leaks'][t]:>9}{planted['support'][t]:>9}")
    print(f"saved {args.results}")
    if reasons:
        print("FAIL: " + "; ".join(reasons), file=sys.stderr)
        sys.exit(1)
    print("PASS" + (" (PERSON leaks allowed)" if args.allow_person_leaks and planted["leaks"]["PERSON"] else ""))


if __name__ == "__main__":
    main()
