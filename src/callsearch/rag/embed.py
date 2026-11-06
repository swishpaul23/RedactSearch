"""Phase 3: Embed redacted text chunks with sentence-transformers.

Vectors are L2-normalised, so cosine similarity is a dot product. BGE models expect an
instruction prefix on queries (not on passages); MODELS records that per model.
"""

from __future__ import annotations

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
