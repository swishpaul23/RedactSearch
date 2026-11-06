"""Phase 3: Vector store for chunk embeddings in data/index/ (Chroma, cosine distance).

python -m callsearch.rag.store "customer wants a refund" -k 5 [--split test] [--mode hybrid]
    Quick manual search. Prints call/turn ids, scores and redacted snippets.

Search modes: "dense" (embeddings), "bm25" (keywords, see bm25.py) and "hybrid" (both lists
fused with reciprocal rank fusion). The BM25 table is built in memory from the stored
documents on first use, so it always matches the index.

The collection metadata records the embedding model, so a query is always embedded
with the model that built the index.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from collections import Counter

from callsearch.rag.bm25 import BM25, rrf, terms
from callsearch.rag.chunk import Chunk
from callsearch.rag.embed import Embedder, SentenceTransformerEmbedder

DEFAULT_INDEX_DIR = Path("data/index/chroma")
COLLECTION = "calls"
ADD_BATCH = 1000
MODES = ("dense", "bm25", "hybrid")
DEFAULT_MODE = "dense"
FUSION_DEPTH = 50  # results taken from each list before fusing


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
        self._table = None  # (ids, docs, metadatas, BM25), built lazily
        self._df = None     # term -> number of stored chunks containing it (grounding check)

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
        self._table = self._df = None
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

    @staticmethod
    def _hit(cid: str, doc: str, m: dict, score: float) -> Hit:
        return Hit(cid, m["call_id"], m["turn_id"], m["speaker"], m["start_sec"], m["split"], doc, score)

    def _dense(self, query: str, n: int, split: str | None) -> list[Hit]:
        res = self._collection().query(query_embeddings=[self.embedder().embed_query(query)], n_results=n,
                                       where={"split": split} if split else None,
                                       include=["documents", "metadatas", "distances"])
        return [self._hit(cid, doc, m, 1.0 - dist) for cid, doc, m, dist in
                zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def _bm25_table(self):
        if self._table is None:
            got = self._collection().get(include=["documents", "metadatas"])
            self._table = (got["ids"], got["documents"], got["metadatas"], BM25(got["documents"]))
        return self._table

    def doc_freq(self) -> Counter:
        """How many stored chunks contain each term (bm25.terms), for the answer grounding check."""
        if self._df is None:
            _, docs, _, _ = self._bm25_table()
            self._df = Counter(t for d in docs for t in set(terms(d)))
        return self._df

    def _bm25(self, query: str, n: int, split: str | None) -> list[Hit]:
        ids, docs, metas, bm25 = self._bm25_table()
        allowed = None if split is None else {i for i, m in enumerate(metas) if m["split"] == split}
        return [self._hit(ids[i], docs[i], metas[i], s) for i, s in bm25.top(query, n, allowed)]

    def turns(self, ids: list[str]) -> dict[str, Hit]:
        """Stored turns by chunk id (missing ids are skipped), as Hits with score 0."""
        if not ids:
            return {}
        got = self._collection().get(ids=list(dict.fromkeys(ids)), include=["documents", "metadatas"])
        return {cid: self._hit(cid, doc, m, 0.0) for cid, doc, m in zip(got["ids"], got["documents"], got["metadatas"])}

    def with_neighbors(self, hits: list[Hit], n: int) -> list[list[Hit]]:
        """For each hit, the stored turns from n before to n after it in the same call, in order."""
        if n <= 0:
            return [[h] for h in hits]
        want = [f"{h.call_id}:{h.turn_id + d}" for h in hits for d in range(-n, n + 1) if d and h.turn_id + d >= 0]
        found = self.turns(want)
        out = []
        for h in hits:
            window = [found.get(f"{h.call_id}:{h.turn_id + d}") if d else h for d in range(-n, n + 1)]
            out.append([w for w in window if w is not None])
        return out

    def search(self, query: str, k: int = 5, split: str | None = None, mode: str = DEFAULT_MODE) -> list[Hit]:
        if mode == "dense":
            return self._dense(query, k, split)
        if mode == "bm25":
            return self._bm25(query, k, split)
        if mode != "hybrid":
            raise ValueError(f"unknown search mode {mode!r}; known: {MODES}")
        depth = max(k, FUSION_DEPTH)
        dense, keyword = self._dense(query, depth, split), self._bm25(query, depth, split)
        by_id = {h.chunk_id: h for h in keyword + dense}
        fused = rrf([[h.chunk_id for h in dense], [h.chunk_id for h in keyword]])[:k]
        return [Hit(**{**by_id[cid].__dict__, "score": s}) for cid, s in fused]


def main() -> None:
    p = argparse.ArgumentParser(description="Search the call index.")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--split", choices=["train", "dev", "test"])
    p.add_argument("--mode", default="bm25", choices=MODES)
    p.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = p.parse_args()
    for h in CallIndex(args.index_dir).search(args.query, args.k, args.split, args.mode):
        print(f"{h.score:.3f}  {h.chunk_id:<14} {h.start_sec:>6.1f}s {h.speaker:<8} {h.text}")


if __name__ == "__main__":
    main()
