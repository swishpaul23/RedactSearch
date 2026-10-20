"""CLI: compare detectors from results/eval_<detector>_<split>.json as a markdown table.

python scripts/compare_results.py --split dev      -> results/comparison_dev.md

Missing detector files are listed as missing, not fatal. Metrics only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DETECTORS = ["regex", "ner", "pipeline", "presidio", "presidio_casin", "pipeline_llm"]
LLM_DETECTORS = {"llm", "pipeline_llm"}
TYPES = ["PERSON", "EMAIL", "PHONE", "CARD", "SIN", "ACCOUNT"]
SUMMARY_TYPES = ["CARD", "SIN", "PERSON"]


def label(d: str, res: dict) -> str:
    """Detector name plus, for LLM detectors, the model and prompt that produced the numbers.
    Older files saved before provenance was recorded show [model unknown]."""
    llm = res.get("llm")
    if llm:
        return f"{d} [{llm['provider']}:{llm['model']} prompt {llm['prompt_version']}]"
    return f"{d} [model unknown]" if d in LLM_DETECTORS else d


def load(results_dir: Path, split: str) -> dict[str, dict | None]:
    out = {}
    for d in DETECTORS:
        path = results_dir / f"eval_{d}_{split}.json"
        out[d] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def _metrics(res: dict, t: str) -> dict:
    return res["micro"] if t == "overall" else res["per_type"][t]


def build_comparison(split: str, results_dir: Path = Path("results")) -> str:
    res = load(results_dir, split)
    present = [d for d in DETECTORS if res[d]]
    missing = [d for d in DETECTORS if not res[d]]

    lines = [f"# Detector comparison: `{split}` split", ""]
    if missing:
        lines += [f"Missing results (not run yet): {', '.join(missing)}", ""]
    lines += ["| detector | type | precision | recall | F1 | leak rate | support |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for d in present:
        for t in TYPES + ["overall"]:
            m = _metrics(res[d], t)
            lines.append(f"| {label(d, res[d])} | {t} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                         f"| {m['leak_rate']:.1%} | {m['support']} |")

    for other in ("presidio", "presidio_casin", "pipeline_llm"):
        lines += ["", f"## Pipeline vs {label(other, res[other]) if res[other] else other}", ""]
        if res["pipeline"] and res[other]:
            lines += [f"| type | pipeline recall | {other} recall | pipeline leak | {other} leak |",
                      "|---|---:|---:|---:|---:|"]
            for t in SUMMARY_TYPES + ["overall"]:
                a, b = _metrics(res["pipeline"], t), _metrics(res[other], t)
                lines.append(f"| {t} | {a['recall']:.3f} | {b['recall']:.3f} | {a['leak_rate']:.1%} "
                             f"| {b['leak_rate']:.1%} |")
        else:
            lines.append(f"n/a: needs both pipeline and {other} results.")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Write results/comparison_<split>.md")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    md = build_comparison(args.split, args.results)
    out = args.results / f"comparison_{args.split}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
