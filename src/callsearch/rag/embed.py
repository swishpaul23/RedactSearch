"""Phase 3: Embed redacted text chunks with sentence-transformers.

Vectors are L2-normalised, so cosine similarity is a dot product. BGE models expect an
instruction prefix on queries (not on passages); MODELS records that per model.

The hosted API embeds questions with OnnxEmbedder instead (onnxruntime, no torch), loaded from
the folder scripts/export_onnx.py writes when CALLSEARCH_ONNX_DIR points at it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

# short name -> (Hub id, query prefix)
MODELS: dict[str, tuple[str, str]] = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
}
DEFAULT_MODEL = "bge-small"


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str = DEFAULT_MODEL, batch_size: int = 64, device: str | None = None):
        if name not in MODELS:
            raise KeyError(f"unknown embedding model {name!r}; known: {sorted(MODELS)}")
        self.name = name
        self.hub_id, self.query_prefix = MODELS[name]
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.hub_id, device=self.device)
        return self._model

    def _encode(self, texts: list[str], progress: bool = False) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=progress, convert_to_numpy=True)
        return vecs.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, progress=len(texts) > 500)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class OnnxEmbedder:
    """The same vectors as SentenceTransformerEmbedder, from an exported model.onnx."""

    def __init__(self, model_dir: Path | str, batch_size: int = 64):
        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "embedder.json").read_text(encoding="utf-8"))
        self.name = meta["name"]
        self.query_prefix = meta["query_prefix"]
        self.inputs = meta["inputs"]
        self.max_length = meta["max_length"]
        self.batch_size = batch_size
        self._session = self._tokenizer = None

    def _load(self):
        if self._session is None:
            import onnxruntime
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
            tok.enable_truncation(self.max_length)
            tok.enable_padding()
            self._tokenizer = tok
            self._session = onnxruntime.InferenceSession(str(self.dir / "model.onnx"),
                                                         providers=["CPUExecutionProvider"])
        return self._session, self._tokenizer

    def _encode(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        session, tok = self._load()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            enc = tok.encode_batch(texts[i:i + self.batch_size])
            feeds = {"input_ids": [e.ids for e in enc], "attention_mask": [e.attention_mask for e in enc],
                     "token_type_ids": [e.type_ids for e in enc]}
            vecs = session.run(None, {k: np.asarray(feeds[k], dtype=np.int64) for k in self.inputs})[0]
            vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
            out.extend(vecs.tolist())
        return out

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


def default_embedder(name: str) -> Embedder:
    """The ONNX export when CALLSEARCH_ONNX_DIR holds one for this model, else sentence-transformers."""
    onnx_dir = os.environ.get("CALLSEARCH_ONNX_DIR")
    if onnx_dir and (Path(onnx_dir) / "embedder.json").exists():
        embedder = OnnxEmbedder(onnx_dir)
        if embedder.name == name:
            return embedder
    return SentenceTransformerEmbedder(name)
