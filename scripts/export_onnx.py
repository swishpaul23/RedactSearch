"""Export a query embedder to ONNX, so the hosted API can embed questions without torch.

python scripts/export_onnx.py --out models/onnx/bge-small            # bge-small (the index's model)
python scripts/export_onnx.py --out DIR --model minilm

Writes model.onnx (the whole sentence-transformers pipeline: encoder, pooling and
normalisation), tokenizer.json and embedder.json. The export is checked against
sentence-transformers on a few sentences and fails if any vector differs by more than
--tolerance. Dockerfile.api runs this in a build stage; callsearch.rag.embed.OnnxEmbedder
loads the result when CALLSEARCH_ONNX_DIR points at it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from callsearch.rag.embed import MODELS, OnnxEmbedder  # noqa: E402

CHECK = ["I am looking for a hotel in Cambridge.", "When does the train to London leave?",
         "Represent this sentence for searching relevant passages: refund for a double charge", "ok"]


def export(name: str, out: Path, source: str | None = None, tolerance: float = 1e-4) -> float:
    import torch
    from sentence_transformers import SentenceTransformer

    hub_id, prefix = MODELS[name]
    st = SentenceTransformer(source or hub_id, device="cpu").eval()
    tok = st.tokenizer
    out.mkdir(parents=True, exist_ok=True)
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))

    sample = tok(CHECK, padding=True, truncation=True, return_tensors="pt")
    inputs = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in sample]

    class Pipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.st = st

        def forward(self, *tensors):
            return self.st(dict(zip(inputs, tensors)))["sentence_embedding"]

    axes = {k: {0: "batch", 1: "seq"} for k in inputs} | {"embedding": {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(Pipeline(), tuple(sample[k] for k in inputs), str(out / "model.onnx"),
                          input_names=inputs, output_names=["embedding"], dynamic_axes=axes,
                          opset_version=17, dynamo=False)
    (out / "embedder.json").write_text(json.dumps({
        "name": name, "hub_id": hub_id, "query_prefix": prefix, "inputs": inputs,
        "max_length": st.max_seq_length}, indent=2) + "\n", encoding="utf-8")

    expected = st.encode(CHECK, normalize_embeddings=True, convert_to_numpy=True)
    got = OnnxEmbedder(out).embed_passages(CHECK)
    diff = max(abs(a - b) for row_e, row_g in zip(expected, got) for a, b in zip(row_e, row_g))
    if diff > tolerance:
        sys.exit(f"ONNX export differs from sentence-transformers by {diff:.2e} (tolerance {tolerance})")
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="bge-small", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", help="local model folder instead of the Hub id (tests use a tiny model)")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()
    diff = export(args.model, args.out, args.source, args.tolerance)
    print(f"Exported {args.model} to {args.out} (max difference from sentence-transformers {diff:.1e})")


if __name__ == "__main__":
    main()
