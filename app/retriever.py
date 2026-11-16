"""Hybrid retrieval: BM25 (exact terms, numbers, names) + dense embeddings (meaning,
cross-lingual), combined with Reciprocal Rank Fusion (RRF).

RRF is used instead of adding raw scores because BM25 scores (0‥30+) and cosine
similarities (0‥1) are on incomparable scales; RRF only needs ranks.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.ingest import Chunk
from app.tokenizer import tokenize

logger = logging.getLogger(__name__)

Mode = Literal["bm25", "dense", "hybrid"]
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """score(d) = Σ 1 / (k + rank_i(d)), rank starting at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    def __init__(
        self, chunks: list[Chunk], settings: Settings, embed_fn: EmbedFn | None = None, fingerprint: str = "nocache"
    ):
        self.chunks = chunks
        self.settings = settings
        self.embed_fn = embed_fn
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.matrix: np.ndarray | None = self._load_embeddings(fingerprint) if embed_fn else None

    # ------------------------------------------------------------------ #
    @property
    def dense_enabled(self) -> bool:
        return self.matrix is not None

    def _load_embeddings(self, fingerprint: str) -> np.ndarray:
        model = self.settings.embedding_model
        cache = self.settings.cache_dir / f"emb-{model}-{fingerprint}.npy"
        if cache.exists():
            logger.info("Loaded embeddings from cache (%s)", cache.name)
            return np.load(cache)
        logger.info("Embedding %d chunks with %s …", len(self.chunks), model)
        vectors: list[list[float]] = []
        texts = [f"{c.title}\n{c.text}" for c in self.chunks]  # title adds document context
        for i in range(0, len(texts), 128):
            vectors.extend(self.embed_fn(texts[i : i + 128]))
        matrix = _l2_normalize(np.asarray(vectors, dtype=np.float32))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, matrix)
        return matrix

    # ------------------------------------------------------------------ #
    def _allowed(self, doc_ids: set[str] | None) -> np.ndarray:
        if not doc_ids:
            return np.ones(len(self.chunks), dtype=bool)
        return np.array([c.doc_id in doc_ids for c in self.chunks])

    def _bm25_ranking(self, query: str, mask: np.ndarray, k: int) -> list[int]:
        scores = np.where(mask, self.bm25.get_scores(tokenize(query)), 0.0)
        order = np.argsort(-scores)[:k]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_ranking(self, q_vec: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
        sims = np.where(mask, self.matrix @ q_vec, -1.0)
        return [int(i) for i in np.argsort(-sims)[:k] if mask[i]]

    def search(
        self, queries: str | list[str], k: int | None = None, mode: Mode = "hybrid", doc_ids: set[str] | None = None
    ) -> list[Hit]:
        """Search with one or more query variants; all rankings are fused with RRF."""
        queries = [queries] if isinstance(queries, str) else queries
        k = k or self.settings.retrieve_k
        pool = max(k * 4, 20)  # look deeper than k before fusing
        mask = self._allowed(doc_ids)

        use_dense = mode in ("dense", "hybrid") and self.dense_enabled
        q_vecs = None
        if use_dense:
            try:  # one batched embedding call for all query variants
                q_vecs = _l2_normalize(np.asarray(self.embed_fn(queries), dtype=np.float32))
            except Exception:  # embedding API outage → keep answering with BM25 instead of failing
                logger.warning("Embedding call failed; falling back to BM25 for this search", exc_info=True)
                use_dense = False
                if mode == "dense":  # nothing to fall back on for a dense-only request
                    raise

        rankings, bm25_pos, dense_pos = [], {}, {}
        for qi, q in enumerate(queries):
            if mode in ("bm25", "hybrid"):
                r = self._bm25_ranking(q, mask, pool)
                rankings.append(r)
                for pos, idx in enumerate(r, 1):
                    bm25_pos[idx] = min(pos, bm25_pos.get(idx, pos))
            if use_dense:
                r = self._dense_ranking(q_vecs[qi], mask, pool)
                rankings.append(r)
                for pos, idx in enumerate(r, 1):
                    dense_pos[idx] = min(pos, dense_pos.get(idx, pos))

        fused = reciprocal_rank_fusion(rankings, k=self.settings.rrf_k)[:k]
        return [Hit(self.chunks[i], round(s, 5), bm25_pos.get(i), dense_pos.get(i)) for i, s in fused]


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
