"""Keyword retrieval (Okapi BM25) and reciprocal rank fusion with the dense results.

Dense embeddings match meaning but blur exact tokens: train ids, times, prices and venue
names. BM25 scores exact token overlap, so the two fail on different questions. Hybrid
search fuses both ranked lists with reciprocal rank fusion (RRF), which needs no score
calibration between them: score(d) = sum over lists of 1 / (RRF_K + rank of d).

The tokenizer keeps times, prices and ids whole ("05:00", "18.88", "tr7423") and drops
placeholders such as [PERSON], so redacted data can never become a match key.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:[.:'][a-z0-9]+)*")
PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from had has have he her his how i if in is it its "
    "me my of on or our she so that the their them then there they this to uh um was we were what when where "
    "which who why will with would you your".split())
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(PLACEHOLDER.sub(" ", text).lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tfs[i][t]
                norm = 1 - self.b + self.b * self.lens[i] / (self.avg_len or 1.0)
                out[i] = out.get(i, 0.0) + idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out

    def top(self, query: str, n: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        s = self.scores(query)
        items = [(i, v) for i, v in s.items() if allowed is None or i in allowed]
        return sorted(items, key=lambda x: (-x[1], x[0]))[:n]


def rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists; ties keep the order of first appearance."""
    score: dict[str, float] = {}
    first: dict[str, int] = {}
    pos = 0
    for ids in ranked_lists:
        for rank, cid in enumerate(ids, 1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in first:
                first[cid] = pos
                pos += 1
    return sorted(score.items(), key=lambda x: (-x[1], first[x[0]]))
