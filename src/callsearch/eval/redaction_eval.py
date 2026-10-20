"""Score redaction against ground-truth PII spans (recall-weighted).

python -m callsearch.eval.redaction_eval --detector regex --split dev [--json PATH | --no-save]

Saves to results/eval_<detector>_<split>.json by default. Prints and saves metrics only,
never transcript text, so raw PII is not logged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from callsearch.data.generator import read_jsonl
from callsearch.redact.base import Detector, Span, available_detectors, get_detector
from callsearch.redact.merge import redact

PII_TYPES = ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]
DEFAULT_DATA = Path("data/raw/calls.jsonl")
RESULTS_DIR = Path("results")  # tracked in git; metrics only. data/redacted/ is for redacted transcripts only.


def match_spans(preds: list[Span], golds: list[dict]) -> tuple[list[tuple[Span, dict]], list[Span], list[dict]]:
    """Greedy one-to-one: each pred takes the unmatched same-type gold it overlaps most."""
    matched, fps, free = [], [], list(golds)
    for p in sorted(preds, key=lambda s: s.start):
        best, best_ov = None, 0
        for g in free:
            ov = min(p.end, g["end"]) - max(p.start, g["start"])
            if g["type"] == p.type and ov > best_ov:
                best, best_ov = g, ov
        if best is None:
            fps.append(p)
        else:
            free.remove(best)
            matched.append((p, best))
    return matched, fps, free


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def evaluate(calls: list[dict], detector: Detector) -> dict:
    tp, fp, fn, exact, support = Counter(), Counter(), Counter(), Counter(), Counter()
    leaks, distractor_fp, distractor_total = Counter(), Counter(), Counter()

    for call in calls:
        for turn in call["turns"]:
            text, tid = turn["text"], turn["turn_id"]
            golds = [l for l in call["labels"] if l["turn_id"] == tid]
            distractors = [d for d in call.get("distractors", []) if d["turn_id"] == tid]
            preds = detector.detect(text)

            matched, fps, fns = match_spans(preds, golds)
            for p, g in matched:
                tp[g["type"]] += 1
                exact[g["type"]] += (p.start, p.end) == (g["start"], g["end"])
            for p in fps:
                fp[p.type] += 1
            for g in fns:
                fn[g["type"]] += 1
            for g in golds:
                support[g["type"]] += 1

            for d in distractors:
                distractor_total[d["type"]] += 1
            for p in preds:
                hit = next((d for d in distractors if p.start < d["end"] and d["start"] < p.end), None)
                if hit:
                    distractor_fp[hit["type"]] += 1

            redacted = redact(text, preds)
            for g in golds:
                leaks[g["type"]] += g["value"] in redacted

    types = PII_TYPES + sorted((set(support) | set(fp)) - set(PII_TYPES))
    per_type = {}
    for t in types:
        per_type[t] = {**_prf(tp[t], fp[t], fn[t]), "tp": tp[t], "fp": fp[t], "fn": fn[t], "support": support[t],
                       "exact_recall": exact[t] / support[t] if support[t] else 0.0,
                       "leaked": leaks[t], "leak_rate": leaks[t] / support[t] if support[t] else 0.0}

    def micro(ts: list[str]) -> dict:
        s = {k: sum(per_type[t][k] for t in ts) for k in ("tp", "fp", "fn", "support", "leaked")}
        ex = sum(exact[t] for t in ts)
        return {**_prf(s["tp"], s["fp"], s["fn"]), **s,
                "exact_recall": ex / s["support"] if s["support"] else 0.0,
                "leak_rate": s["leaked"] / s["support"] if s["support"] else 0.0}

    return {
        "detector": detector.name,
        "n_calls": len(calls),
        "per_type": per_type,
        "micro": micro(types),
        "micro_excl_person": micro([t for t in types if t != "PERSON"]),
        "distractor_fp": {t: {"flagged": distractor_fp[t], "total": distractor_total[t]}
                          for t in sorted(distractor_total)},
    }


def format_table(res: dict, split: str) -> str:
    lines = [f"Detector: {res['detector']}   split: {split}   calls: {res['n_calls']}", ""]
    hdr = f"{'type':<16}{'prec':>7}{'recall':>8}{'f1':>7}{'exact_R':>9}{'support':>9}{'tp':>6}{'fp':>6}{'fn':>6}"
    lines += [hdr, "-" * len(hdr)]

    def row(name: str, m: dict) -> str:
        return (f"{name:<16}{m['precision']:>7.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['exact_recall']:>9.3f}"
                f"{m['support']:>9}{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")

    lines += [row(t, m) for t, m in res["per_type"].items()]
    lines += ["-" * len(hdr), row("micro (all)", res["micro"]), row("micro (no PER)", res["micro_excl_person"])]

    lines += ["", "LEAK CHECK (labeled values still verbatim after redaction)",
              f"{'type':<16}{'leaked':>8}{'support':>9}{'rate':>8}"]
    for t, m in res["per_type"].items():
        lines.append(f"{t:<16}{m['leaked']:>8}{m['support']:>9}{m['leak_rate']:>8.1%}")
    for name, key in (("total", "micro"), ("total (no PER)", "micro_excl_person")):
        m = res[key]
        lines.append(f"{name:<16}{m['leaked']:>8}{m['support']:>9}{m['leak_rate']:>8.1%}")

    lines += ["", "DISTRACTOR FALSE POSITIVES (predicted spans overlapping a distractor)",
              f"{'type':<16}{'flagged':>8}{'total':>9}"]
    for t, m in res["distractor_fp"].items():
        lines.append(f"{t:<16}{m['flagged']:>8}{m['total']:>9}")
    return "\n".join(lines)


def default_results_path(detector: str, split: str) -> Path:
    return RESULTS_DIR / f"eval_{detector}_{split}.json"


def save_results(res: dict, path: Path) -> None:
    """Results are metrics only (counts, rates, type names); never transcript text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a PII detector against labeled calls.")
    p.add_argument("--detector", default="regex", choices=available_detectors())
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--json", type=Path, help="results path (default: results/eval_<detector>_<split>.json)")
    p.add_argument("--no-save", action="store_true", help="print only, do not save results")
    args = p.parse_args()

    calls = [c for c in read_jsonl(args.data) if args.split == "all" or c["split"] == args.split]
    res = evaluate(calls, get_detector(args.detector))
    res["split"] = args.split
    print(format_table(res, args.split))
    if not args.no_save:
        path = args.json or default_results_path(args.detector, args.split)
        save_results(res, path)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
