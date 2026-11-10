"""Phase 3: Vector store for chunk embeddings in data/index/ (Chroma, cosine distance).

python -m callsearch.rag.store "customer wants a refund" -k 5 [--split test]
    Quick manual search. Prints call/turn ids, scores and redacted snippets.

The collection metadata records the embedding model, so a query is always embedded
with the model that built the index.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from callsearch.rag.chunk import Chunk
from callsearch.rag.embed import Embedder, SentenceTransformerEmbedder

DEFAULT_INDEX_DIR = Path("data/index/chroma")
COLLECTION = "calls"
ADD_BATCH = 1000


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    call_id: str
    turn_id: int
    speaker: str
    start_sec: float
    split: str
    text: str
    score: float  # cosine similarity, higher is closer


class CallIndex:
    def __init__(self, path: Path | str = DEFAULT_INDEX_DIR, embedder: Embedder | None = None):
        self.path = Path(path)
        self._embedder = embedder
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        return self._client

    def _collection(self):
        try:
            return self.client.get_collection(COLLECTION, embedding_function=None)
        except Exception as e:  # chromadb raises NotFoundError (version-dependent type)
            raise FileNotFoundError(f"no index at {self.path}; build one: python scripts/build_index.py") from e

    @property
    def info(self) -> dict:
        return dict(self._collection().metadata or {})

    def count(self) -> int:
        return self._collection().count()

    def build(self, chunks: list[Chunk], embedder: Embedder, info: dict | None = None) -> int:
        """Replace the collection with these chunks. Returns the number stored."""
        if any(c.name == COLLECTION for c in self.client.list_collections()):
            self.client.delete_collection(COLLECTION)
        col = self.client.create_collection(
            COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
            metadata={"embed_model": embedder.name, "n_chunks": len(chunks), **(info or {})})
        vectors = embedder.embed_passages([c.embed_text for c in chunks])
        for i in range(0, len(chunks), ADD_BATCH):
            part = chunks[i:i + ADD_BATCH]
            col.add(ids=[c.id for c in part], embeddings=vectors[i:i + ADD_BATCH],
                    documents=[c.text for c in part], metadatas=[c.metadata() for c in part])
        self._embedder = embedder
        return col.count()

    def documents(self, batch: int = 1000) -> Iterator[tuple[str, str]]:
        """Every stored (chunk id, document text), for the index guard."""
        col, offset = self._collection(), 0
        while True:
            got = col.get(include=["documents"], limit=batch, offset=offset)
            if not got["ids"]:
                return
            yield from zip(got["ids"], got["documents"])
            offset += len(got["ids"])

    def embedder(self) -> Embedder:
        model = self.info.get("embed_model")
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(model)
        if self._embedder.name != model:
            raise ValueError(f"index was built with {model!r}, query embedder is {self._embedder.name!r}")
        return self._embedder

    def search(self, query: str, k: int = 5, split: str | None = None) -> list[Hit]:
        col = self._collection()
        res = col.query(query_embeddings=[self.embedder().embed_query(query)], n_results=k,
                        where={"split": split} if split else None,
                        include=["documents", "metadatas", "distances"])
        return [Hit(cid, m["call_id"], m["turn_id"], m["speaker"], m["start_sec"], m["split"], doc, 1.0 - dist)
                for cid, doc, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0],
                                             res["distances"][0])]


def main() -> None:
    p = argparse.ArgumentParser(description="Search the call index.")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--split", choices=["train", "dev", "test"])
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = p.parse_args()
    for h in CallIndex(args.index_dir).search(args.query, args.k, args.split):
        print(f"{h.score:.3f}  {h.chunk_id:<14} {h.start_sec:>6.1f}s {h.speaker:<8} {h.text}")


if __name__ == "__main__":
    main()
