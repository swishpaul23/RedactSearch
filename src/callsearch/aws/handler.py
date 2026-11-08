"""Lambda: redact a batch of raw calls when it lands in the raw bucket.

Trigger: s3:ObjectCreated on <raw bucket>/incoming/*.jsonl (one JSON call per line, the same
shape as data/raw/calls.jsonl without labels; see scripts/split_calls.py).

For each object it writes, to the redacted bucket:
    redacted/<name>.jsonl            redacted calls (same fields as callsearch.redact.export)
    redacted/<name>.manifest.json    sha256 of that file, detector, counts, source key

It uses the same redact_calls() as the local export, so local and cloud output match.
Logs carry keys and counts only, never transcript text. The detector (regex + DistilBERT) is
loaded once per container and reused across invocations.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse

RAW_PREFIX = os.environ.get("RAW_PREFIX", "incoming/")
OUT_PREFIX = "redacted/"
MAX_BYTES = 20 * 1024 * 1024  # a batch larger than this is a mistake, not a workload
REQUIRED = ("call_id", "split", "source", "turns")

_detector = None


def default_detector():
    """regex + DistilBERT (pipeline_hf), with the model baked into the image."""
    global _detector
    if _detector is None:
        from callsearch.redact.ner_hf import HFNerDetector
        from callsearch.redact.pipeline import PipelineHFDetector
        from callsearch.redact.regex import RegexDetector

        model_dir = os.environ.get("NER_MODEL_DIR", "models/ner_distilbert")
        _detector = PipelineHFDetector([RegexDetector(), HFNerDetector(model_dir=model_dir)])
    return _detector


def log(**fields) -> None:
    print(json.dumps(fields))  # one JSON line per event in CloudWatch


def parse_calls(body: bytes, key: str) -> list[dict]:
    calls = []
    for n, line in enumerate(body.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        c = json.loads(line)
        missing = [f for f in REQUIRED if f not in c]
        if missing:
            raise ValueError(f"{key} line {n}: missing fields {missing}")
        calls.append(c)
    if not calls:
        raise ValueError(f"{key}: no calls")
    return calls


def output_keys(key: str) -> tuple[str, str]:
    name = key[len(RAW_PREFIX):] if key.startswith(RAW_PREFIX) else key
    stem = name[: -len(".jsonl")] if name.endswith(".jsonl") else name
    return f"{OUT_PREFIX}{stem}.jsonl", f"{OUT_PREFIX}{stem}.manifest.json"


def process_object(s3, bucket: str, key: str, out_bucket: str, detector) -> dict:
    from callsearch.redact.export import redact_calls

    t0 = time.perf_counter()
    obj = s3.get_object(Bucket=bucket, Key=key)
    if obj.get("ContentLength", 0) > MAX_BYTES:
        raise ValueError(f"{key} is {obj['ContentLength']} bytes; split it (scripts/split_calls.py)")
    calls = parse_calls(obj["Body"].read(), key)
    redacted = redact_calls(calls, detector)

    body = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in redacted).encode("utf-8")
    out_key, manifest_key = output_keys(key)
    manifest = {"redacted": True, "file": out_key, "sha256": hashlib.sha256(body).hexdigest(),
                "detector": getattr(detector, "name", "unknown"), "n_calls": len(redacted),
                "n_turns": sum(len(c["turns"]) for c in redacted), "source_key": key,
                "created_by": "callsearch.aws.handler"}
    # file first, manifest last: a manifest only ever vouches for a file that already exists
    s3.put_object(Bucket=out_bucket, Key=out_key, Body=body, ContentType="application/x-ndjson")
    s3.put_object(Bucket=out_bucket, Key=manifest_key, Body=json.dumps(manifest, indent=2).encode("utf-8"),
                  ContentType="application/json")
    return {"source_key": key, "out_key": out_key, "n_calls": manifest["n_calls"], "n_turns": manifest["n_turns"],
            "seconds": round(time.perf_counter() - t0, 2)}


def handler(event, context=None, s3=None, detector=None):
    out_bucket = os.environ["REDACTED_BUCKET"]
    if s3 is None:
        import boto3

        s3 = boto3.client("s3")
    done, skipped = [], []
    for rec in event.get("Records", []):
        bucket = rec["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(rec["s3"]["object"]["key"])
        if not (key.startswith(RAW_PREFIX) and key.endswith(".jsonl")):
            skipped.append(key)
            log(event="skipped", key=key)
            continue
        result = process_object(s3, bucket, key, out_bucket, detector or default_detector())
        log(event="redacted", **result)
        done.append(result)
    return {"processed": done, "skipped": skipped}
