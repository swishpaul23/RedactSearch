"""Redaction boundary: write redacted transcripts plus a manifest that the index builder checks.

python -m callsearch.redact.export --detector pipeline_hf
    data/raw/calls.jsonl -> data/redacted/calls.jsonl + data/redacted/manifest.json

The output keeps only what search needs (call id, split, source, speaker, start time and
redacted text). Labels, distractors and template ids never cross this boundary. The manifest
records the sha256 of the redacted file, so the index builder can refuse a file that was not
written here or was changed afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from callsearch.redact.base import available_detectors, get_detector
from callsearch.redact.merge import redact

DEFAULT_RAW = Path("data/raw/calls.jsonl")
DEFAULT_OUT_DIR = Path("data/redacted")
REDACTED_FILE = "calls.jsonl"
MANIFEST_FILE = "manifest.json"
KEEP_CALL = ("call_id", "split", "source")
KEEP_TURN = ("turn_id", "speaker", "start_sec")
BATCH = 256


class ManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def redact_calls(calls: list[dict], detector) -> list[dict]:
    """Redacted copies of the calls, with every label-derived field dropped."""
    texts = [t["text"] for c in calls for t in c["turns"]]
    spans = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        spans += detector.detect_batch(chunk) if hasattr(detector, "detect_batch") else [detector.detect(t) for t in chunk]
    out, i = [], 0
    for c in calls:
        turns = []
        for t in c["turns"]:
            turns.append({**{k: t[k] for k in KEEP_TURN}, "text": redact(t["text"], spans[i])})
            i += 1
        out.append({**{k: c[k] for k in KEEP_CALL}, "turns": turns})
    return out


def write_redacted(calls: list[dict], out_dir: Path, detector_name: str) -> dict:
    """Write calls.jsonl, then the manifest that vouches for it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REDACTED_FILE
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in calls:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    manifest = {"redacted": True, "file": REDACTED_FILE, "sha256": sha256_file(path), "detector": detector_name,
                "n_calls": len(calls), "n_turns": sum(len(c["turns"]) for c in calls),
                "created_by": "callsearch.redact.export"}
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_redacted(out_dir: Path = DEFAULT_OUT_DIR) -> tuple[list[dict], dict]:
    """Read redacted calls only after the manifest checks out. Raises ManifestError otherwise."""
    mpath = out_dir / MANIFEST_FILE
    if not mpath.exists():
        raise ManifestError(f"no {MANIFEST_FILE} in {out_dir}; run: python -m callsearch.redact.export")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("redacted") is not True or manifest.get("created_by") != "callsearch.redact.export":
        raise ManifestError(f"{mpath} does not mark its file as redacted")
    path = out_dir / manifest["file"]
    if path.resolve().parent != out_dir.resolve():
        raise ManifestError("manifest points outside the redacted folder")
    if not path.exists() or sha256_file(path) != manifest["sha256"]:
        raise ManifestError(f"{path} is missing or changed since it was redacted; re-run the export")
    calls = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in calls:
        if set(c) != set(KEEP_CALL) | {"turns"}:
            raise ManifestError(f"{c.get('call_id')}: unexpected fields {sorted(set(c) - set(KEEP_CALL))}")
    return calls, manifest


def main() -> None:
    from callsearch.data.generator import read_jsonl

    p = argparse.ArgumentParser(description="Redact raw calls into data/redacted/ with a manifest.")
    p.add_argument("--detector", default="pipeline_hf", choices=available_detectors())
    p.add_argument("--data", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    if args.detector in {"llm", "pipeline_llm"}:
        p.error("LLM detectors are not supported here; run their eval first to fill the cache")

    t0 = time.perf_counter()
    calls = read_jsonl(args.data)
    manifest = write_redacted(redact_calls(calls, get_detector(args.detector)), args.out_dir, args.detector)
    print(f"redacted {manifest['n_calls']} calls / {manifest['n_turns']} turns with {args.detector} "
          f"in {time.perf_counter() - t0:.0f}s -> {args.out_dir / REDACTED_FILE}")


if __name__ == "__main__":
    main()
