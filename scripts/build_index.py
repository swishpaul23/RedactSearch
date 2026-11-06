"""Phase 3: CLI: chunk and embed redacted transcripts and build the vector index.

python scripts/build_index.py --model bge-small --context-turns 1

Reads data/redacted/ only, and only if its manifest checks out (run
`python -m callsearch.redact.export` first). Never opens data/raw/. Runs the guard's pattern
scan before anything is written, and refuses to build if it finds structured PII.
Then run `python -m callsearch.rag.guard` for the planted-value check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from callsearch.rag.chunk import chunk_calls
from callsearch.rag.embed import DEFAULT_MODEL, MODELS, SentenceTransformerEmbedder
from callsearch.rag.guard import pattern_scan
from callsearch.rag.store import DEFAULT_INDEX_DIR, CallIndex
from callsearch.redact.export import DEFAULT_OUT_DIR, ManifestError, load_redacted


def build(redacted_dir: Path, index_dir: Path, embedder, context_turns: int = 1) -> dict:
    calls, manifest = load_redacted(redacted_dir)
    chunks = chunk_calls(calls, context_turns)
    hits = pattern_scan([c.text for c in chunks])
    if any(hits.values()):
        raise RuntimeError(f"pattern scan found structured PII in the redacted text: {hits}; nothing was written")
    t0 = time.perf_counter()
    n = CallIndex(index_dir).build(chunks, embedder, {
        "detector": manifest["detector"], "manifest_sha256": manifest["sha256"], "context_turns": context_turns})
    return {"n_calls": len(calls), "n_chunks": n, "detector": manifest["detector"], "embed_model": embedder.name,
            "embed_seconds": round(time.perf_counter() - t0, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description="Build the vector index from redacted calls.")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    p.add_argument("--context-turns", type=int, default=1, help="previous turns included in each embedded chunk")
    p.add_argument("--redacted-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = p.parse_args()
    try:
        s = build(args.redacted_dir, args.index_dir, SentenceTransformerEmbedder(args.model), args.context_turns)
    except (ManifestError, RuntimeError) as e:
        print(f"refusing to build: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"indexed {s['n_chunks']} chunks from {s['n_calls']} calls (redacted by {s['detector']}) "
          f"with {s['embed_model']} in {s['embed_seconds']}s -> {args.index_dir}")
    print("next: python -m callsearch.rag.guard")


if __name__ == "__main__":
    main()
