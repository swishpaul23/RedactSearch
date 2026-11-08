"""Split raw calls into upload batches for the S3 raw bucket, with labels removed.

python scripts/split_calls.py --per-file 25
    data/raw/calls.jsonl -> data/upload/batch_000.jsonl, batch_001.jsonl, ...

Only what a real transcript would contain is kept: call id, split, source and the turns
(id, speaker, start time, text). Labels and distractors are evaluation data and never
leave this machine. Then: aws s3 cp data/upload s3://<raw bucket>/incoming/ --recursive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEEP_CALL = ("call_id", "split", "source")
KEEP_TURN = ("turn_id", "speaker", "start_sec", "text")


def strip_call(call: dict) -> dict:
    return {**{k: call[k] for k in KEEP_CALL}, "turns": [{k: t[k] for k in KEEP_TURN} for t in call["turns"]]}


def split(calls: list[dict], out_dir: Path, per_file: int) -> list[Path]:
    if per_file < 1:
        raise ValueError("per_file must be >= 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("batch_*.jsonl"):
        old.unlink()
    paths = []
    for i in range(0, len(calls), per_file):
        path = out_dir / f"batch_{i // per_file:03d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for c in calls[i:i + per_file]:
                f.write(json.dumps(strip_call(c), ensure_ascii=False) + "\n")
        paths.append(path)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Split raw calls into label-free upload batches.")
    p.add_argument("--data", type=Path, default=Path("data/raw/calls.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/upload"))
    p.add_argument("--per-file", type=int, default=25)
    args = p.parse_args()
    calls = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = split(calls, args.out_dir, args.per_file)
    print(f"{len(calls)} calls -> {len(paths)} files of up to {args.per_file} in {args.out_dir} (labels removed)")


if __name__ == "__main__":
    main()
