"""Phase 2: Microsoft Presidio baseline redactor, default configuration, for comparison.

Out-of-the-box on purpose: AnalyzerEngine() with its default spaCy model (en_core_web_lg)
and default recognizers. No custom recognizers, no threshold tuning.

python -m callsearch.redact.presidio_baseline --split dev
    Counts ALL Presidio entities (mapped or not) on distractors, and labels that only an
    unmapped entity touched. Saves results/presidio_extras_<split>.json (metrics only).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from callsearch.redact.base import Span

# Presidio entity -> our type. Everything else Presidio emits is ignored in scoring.
ENTITY_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    # Built-in Canadian SIN recognizer (regex + Luhn). It is DISABLED in Presidio's default
    # config, so under the default baseline it never fires; mapped so it counts if enabled.
    "CA_SIN": "SIN",
    # Presidio's only built-in bank-account recognizer (8-17 digits + context words).
    "US_BANK_NUMBER": "ACCOUNT",
}
# Notably NOT mapped: US_SSN (a US ID with a different format; mapping it to SIN would
# inflate the score), plus LOCATION, DATE_TIME, NRP, URL, IP_ADDRESS, IBAN_CODE, etc.


class PresidioDetector:
    """detect() returns mapped spans. Where mapped spans overlap, the higher Presidio score
    wins (as Presidio's anonymizer resolves conflicts); our type priority is not applied."""

    def __init__(self, analyzer=None, name: str = "presidio"):
        self.name = name
        self._analyzer = analyzer

    @property
    def analyzer(self):
        if self._analyzer is None:
            logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
            from presidio_analyzer import AnalyzerEngine

            self._analyzer = AnalyzerEngine()  # default config, default spaCy model
        return self._analyzer

    def model_name(self) -> str:
        meta = self.analyzer.nlp_engine.nlp["en"].meta
        return f"{meta['lang']}_{meta['name']}-{meta['version']}"

    def analyze_raw(self, text: str) -> list:
        """All Presidio results, mapped or not."""
        return self.analyzer.analyze(text=text, language="en")

    def detect(self, text: str) -> list[Span]:
        mapped = [r for r in self.analyze_raw(text) if r.entity_type in ENTITY_MAP]
        kept: list[Span] = []
        for r in sorted(mapped, key=lambda r: (-r.score, -(r.end - r.start), r.start)):
            s = Span(r.start, r.end, ENTITY_MAP[r.entity_type], self.name)
            if not any(s.overlaps(k) for k in kept):
                kept.append(s)
        return sorted(kept, key=lambda s: s.start)


def casin_analyzer():
    """Presidio's default config with one change: the built-in CaSinRecognizer is set to
    `enabled: true` in Presidio's own default_recognizers.yaml. Config only, no custom recognizer."""
    from pathlib import Path as _P

    import presidio_analyzer
    import yaml
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry import RecognizerRegistryProvider

    conf_path = _P(presidio_analyzer.__file__).parent / "conf" / "default_recognizers.yaml"
    conf = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
    for r in conf["recognizers"]:
        if r.get("name") == "CaSinRecognizer":
            r["enabled"] = True
    nlp_engine = NlpEngineProvider().create_engine()  # same default spaCy model
    registry = RecognizerRegistryProvider(registry_configuration=conf).create_recognizer_registry()
    registry.add_nlp_recognizer(nlp_engine=nlp_engine)
    return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)


class PresidioCaSinDetector(PresidioDetector):
    """Fairness variant: the Presidio baseline with the built-in CA_SIN recognizer enabled."""

    def __init__(self, analyzer=None, name: str = "presidio_casin"):
        super().__init__(analyzer=analyzer, name=name)

    @property
    def analyzer(self):
        if self._analyzer is None:
            logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
            self._analyzer = casin_analyzer()
        return self._analyzer


def _overlaps(a_start: int, a_end: int, b: dict) -> bool:
    return a_start < b["end"] and b["start"] < a_end


def extras(calls: list[dict], det: PresidioDetector) -> dict:
    """Distractor hits by ANY Presidio entity, and labels touched only by unmapped entities."""
    by_distractor, by_entity, totals, unmapped_only = Counter(), Counter(), Counter(), Counter()
    for call in calls:
        for turn in call["turns"]:
            tid = turn["turn_id"]
            raw = det.analyze_raw(turn["text"])
            for d in (d for d in call.get("distractors", []) if d["turn_id"] == tid):
                totals[d["type"]] += 1
                hits = [r for r in raw if _overlaps(r.start, r.end, d)]
                if hits:
                    by_distractor[d["type"]] += 1
                    for e in {r.entity_type for r in hits}:
                        by_entity[e] += 1
            for l in (l for l in call["labels"] if l["turn_id"] == tid):
                hits = [r for r in raw if _overlaps(r.start, r.end, l)]
                if hits and not any(r.entity_type in ENTITY_MAP for r in hits):
                    for e in {r.entity_type for r in hits}:
                        unmapped_only[f"{l['type']}:{e}"] += 1
    return {
        "detector": det.name,
        "model": det.model_name(),
        "distractors_flagged": {t: {"flagged": by_distractor[t], "total": totals[t]} for t in sorted(totals)},
        "distractor_hits_by_entity": dict(sorted(by_entity.items())),
        "labels_touched_only_by_unmapped": dict(sorted(unmapped_only.items())),
    }


def main() -> None:
    from callsearch.data.generator import read_jsonl

    p = argparse.ArgumentParser(description="Presidio extras: distractor hits by any entity.")
    p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    args = p.parse_args()

    calls = [c for c in read_jsonl(args.data) if args.split == "all" or c["split"] == args.split]
    det = PresidioDetector()
    res = {"split": args.split, **extras(calls, det)}

    print(f"Presidio model: {res['model']}   split: {args.split}   calls: {len(calls)}\n")
    print("DISTRACTORS FLAGGED BY ANY PRESIDIO ENTITY")
    print(f"{'type':<16}{'flagged':>8}{'total':>8}")
    for t, m in res["distractors_flagged"].items():
        print(f"{t:<16}{m['flagged']:>8}{m['total']:>8}")
    print("\nby Presidio entity:", res["distractor_hits_by_entity"])
    print("labels touched only by unmapped entities:", res["labels_touched_only_by_unmapped"] or "none")

    out = Path("results") / f"presidio_extras_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
