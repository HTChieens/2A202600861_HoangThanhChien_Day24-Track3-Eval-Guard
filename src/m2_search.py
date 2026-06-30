from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import math
import os, sys
import re
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except (ImportError, OSError, RuntimeError):
        # Regex tokenization keeps the lexical retriever usable in lightweight
        # environments where the Vietnamese model is not installed.
        segmented = " ".join(re.findall(r"\w+", text, flags=re.UNICODE))
    return segmented.replace("_", " ")


def _tokenize(text: str) -> list[str]:
    """Normalize segmented text for case-insensitive BM25 matching."""
    return segment_vietnamese(text).casefold().split()


class _FallbackBM25:
    """Small BM25Okapi-compatible fallback used when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = [Counter(doc) for doc in corpus]
        document_frequency = Counter()
        for doc in corpus:
            document_frequency.update(set(doc))
        count = len(corpus)
        self.idf = {
            term: math.log(1.0 + (count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for frequencies, length in zip(self.term_frequencies, self.doc_lengths):
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                norm = 1.0 - self.b
                if self.avgdl:
                    norm += self.b * length / self.avgdl
                denominator = frequency + self.k1 * norm
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0) / denominator
                )
            scores.append(score)
        return scores


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [_tokenize(chunk.get("text", "")) for chunk in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = _FallbackBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: (-float(scores[index]), index)
        )
        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            document = self.documents[index]
            results.append(SearchResult(
                text=document.get("text", ""),
                score=score,
                metadata=dict(document.get("metadata") or {}),
                method="bm25",
            ))
            if len(results) >= top_k:
                break
        return results


class DenseSearch:
    def __init__(self):
        self._client = None
        self._encoder = None
        self._fallback_documents: list[dict] = []
        self._fallback_vectors: list[Counter] = []
        self._active_collection = COLLECTION_NAME

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        return self._client

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._active_collection = collection
        self._fallback_documents = list(chunks)
        self._fallback_vectors = [
            Counter(_tokenize(chunk.get("text", ""))) for chunk in chunks
        ]
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            if not chunks:
                return

            texts = [chunk.get("text", "") for chunk in chunks]
            vectors = self._get_encoder().encode(
                texts, show_progress_bar=True, convert_to_numpy=True
            )
            points = [
                PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={
                        **dict(chunk.get("metadata") or {}),
                        "text": chunk.get("text", ""),
                    },
                )
                for index, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception as error:
            print(f"  ⚠️  Dense backend unavailable, using local cosine search: {error}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query.strip() or top_k <= 0:
            return []
        try:
            if self._client is not None and self._encoder is not None:
                query_vector = self._get_encoder().encode(
                    query, convert_to_numpy=True
                ).tolist()
                response = self.client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                results = []
                for point in response.points:
                    payload = dict(point.payload or {})
                    results.append(SearchResult(
                        text=str(payload.get("text", "")),
                        score=float(point.score),
                        metadata=payload,
                        method="dense",
                    ))
                return results
        except Exception as error:
            print(f"  ⚠️  Dense query failed, using local cosine search: {error}")
        return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_vector = Counter(_tokenize(query))
        if not query_vector:
            return []
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        scored = []
        for index, document_vector in enumerate(self._fallback_vectors):
            doc_norm = math.sqrt(
                sum(value * value for value in document_vector.values())
            )
            if not doc_norm:
                continue
            dot_product = sum(
                value * document_vector.get(token, 0)
                for token, value in query_vector.items()
            )
            score = dot_product / (query_norm * doc_norm)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                text=str(self._fallback_documents[index].get("text", "")),
                score=float(score),
                metadata=dict(
                    self._fallback_documents[index].get("metadata") or {}
                ),
                method="dense",
            )
            for score, index in scored[:top_k]
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    fused: dict[str, dict] = {}
    order = 0
    for result_list in results_list:
        seen_in_list = set()
        for rank, result in enumerate(result_list):
            if result.text in seen_in_list:
                continue
            seen_in_list.add(result.text)
            if result.text not in fused:
                fused[result.text] = {
                    "score": 0.0,
                    "result": result,
                    "order": order,
                }
                order += 1
            fused[result.text]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(
        fused.values(), key=lambda item: (-item["score"], item["order"])
    )
    return [
        SearchResult(
            text=item["result"].text,
            score=item["score"],
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
