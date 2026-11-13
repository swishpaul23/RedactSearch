"""Download the Lambda's redacted batches and assemble data/redacted/ for the index builder.

python scripts/pull_redacted.py --bucket <redacted bucket>

Every redacted/<name>.jsonl must have a manifest whose sha256 matches what was downloaded,
or nothing is written. Call ids must be unique across batches. The result goes through
callsearch.redact.export.write_redacted, so build_index.py reads it exactly as it reads a
local export. The index is then built from the redacted bucket alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from callsearch.redact.export import DEFAULT_OUT_DIR, KEEP_CALL, ManifestError, write_redacted

PREFIX = "redacted/"


def list_keys(s3, bucket: str, prefix: str = PREFIX) -> list[str]:
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, **({"ContinuationToken": token} if token else {})}
        page = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return sorted(keys)
        token = page["NextContinuationToken"]


def fetch(s3, bucket: str) -> tuple[list[dict], set[str]]:
    """All redacted calls from the bucket, verified against their manifests. Returns (calls, detectors)."""
    keys = list_keys(s3, bucket)
    data = [k for k in keys if k.endswith(".jsonl")]
    if not data:
        raise ManifestError(f"no redacted batches under s3://{bucket}/{PREFIX}")
    calls, seen, detectors = [], set(), set()
    for key in data:
        mkey = key[: -len(".jsonl")] + ".manifest.json"
        if mkey not in keys:
            raise ManifestError(f"{key} has no manifest")
        manifest = json.loads(s3.get_object(Bucket=bucket, Key=mkey)["Body"].read())
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        if manifest.get("redacted") is not True or manifest.get("created_by") != "callsearch.aws.handler":
            raise ManifestError(f"{mkey} does not mark {key} as redacted by the Lambda")
        if hashlib.sha256(body).hexdigest() != manifest["sha256"]:
            raise ManifestError(f"{key} does not match its manifest's sha256")
        detectors.add(manifest["detector"])
        for line in body.decode("utf-8").splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            if set(c) != set(KEEP_CALL) | {"turns"}:
                raise ManifestError(f"{key}: unexpected fields in {c.get('call_id')}")
            if c["call_id"] in seen:
                raise ManifestError(f"{c['call_id']} appears in more than one batch")
            seen.add(c["call_id"])
            calls.append(c)
    return sorted(calls, key=lambda c: c["call_id"]), detectors


def main() -> None:
    import boto3

    p = argparse.ArgumentParser(description="Assemble data/redacted/ from the redacted S3 bucket.")
    p.add_argument("--bucket", required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    try:
        calls, detectors = fetch(boto3.client("s3"), args.bucket)
    except ManifestError as e:
        print(f"refusing to write: {e}", file=sys.stderr)
        sys.exit(1)
    if len(detectors) != 1:
        print(f"refusing to write: batches were redacted by different detectors {sorted(detectors)}", file=sys.stderr)
        sys.exit(1)
    m = write_redacted(calls, args.out_dir, detectors.pop())
    print(f"pulled {m['n_calls']} calls / {m['n_turns']} turns (detector {m['detector']}) -> {args.out_dir}")
    print("next: python scripts/build_index.py")


if __name__ == "__main__":
    main()
