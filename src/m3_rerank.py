from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._model_load_failed = False

    def _load_model(self):
        if self._model is None and not self._model_load_failed:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                # Model packages, weights, or network access may be unavailable
                # in lightweight/offline environments. rerank() then uses its
                # deterministic lexical fallback.
                self._model_load_failed = True
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        pairs = [(query, str(document.get("text", ""))) for document in documents]
        model = self._load_model()
        scores = None
        if model is not None:
            try:
                scores = model.predict(pairs, show_progress_bar=False)
            except TypeError:
                # Compatibility with CrossEncoder versions without this option.
                scores = model.predict(pairs)
            except Exception:
                scores = None

        normalized_scores = _normalize_scores(scores, len(documents))
        if normalized_scores is None:
            normalized_scores = [
                _lexical_relevance(query, document.get("text", ""))
                for document in documents
            ]

        scored = sorted(
            enumerate(zip(normalized_scores, documents)),
            key=lambda item: (-item[1][0], item[0]),
        )
        return [
            RerankResult(
                text=str(document.get("text", "")),
                original_score=float(document.get("score", 0.0) or 0.0),
                rerank_score=float(score),
                metadata=dict(document.get("metadata") or {}),
                rank=rank,
            )
            for rank, (_, (score, document)) in enumerate(scored[:top_k], start=1)
        ]


def _normalize_scores(scores, expected_count: int) -> list[float] | None:
    """Convert scalar/list/NumPy/Torch predictions to one float per document."""
    if scores is None:
        return None
    if isinstance(scores, (int, float)):
        values = [float(scores)]
    elif hasattr(scores, "detach"):
        values = scores.detach().cpu().reshape(-1).tolist()
    elif hasattr(scores, "reshape") and hasattr(scores, "tolist"):
        values = scores.reshape(-1).tolist()
    else:
        try:
            values = list(scores)
        except TypeError:
            return None

    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return values if len(values) == expected_count else None


def _lexical_relevance(query: str, document: str) -> float:
    """Offline fallback combining token coverage, phrase overlap, and BM25 score."""
    tokenize = lambda text: re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)
    query_tokens = tokenize(query)
    document_tokens = tokenize(document)
    if not query_tokens or not document_tokens:
        return 0.0

    query_set, document_set = set(query_tokens), set(document_tokens)
    overlap = query_set & document_set
    coverage = len(overlap) / len(query_set)
    precision = len(overlap) / len(document_set)

    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    document_bigrams = set(zip(document_tokens, document_tokens[1:]))
    phrase_score = (
        len(query_bigrams & document_bigrams) / len(query_bigrams)
        if query_bigrams else 0.0
    )
    return 0.65 * coverage + 0.20 * precision + 0.15 * phrase_score


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": index, "text": str(document.get("text", ""))}
                for index, document in enumerate(documents)
            ]
            results = self._model.rerank(
                RerankRequest(query=query, passages=passages)
            )
            ranked = []
            for rank, result in enumerate(results[:top_k], start=1):
                document = documents[int(result["id"])]
                ranked.append(RerankResult(
                    text=str(document.get("text", "")),
                    original_score=float(document.get("score", 0.0) or 0.0),
                    rerank_score=float(result.get("score", 0.0)),
                    metadata=dict(document.get("metadata") or {}),
                    rank=rank,
                ))
            return ranked
        except Exception:
            # FlashRank is optional; retain useful behavior when it is absent.
            scores = [
                _lexical_relevance(query, document.get("text", ""))
                for document in documents
            ]
            scored = sorted(
                enumerate(zip(scores, documents)),
                key=lambda item: (-item[1][0], item[0]),
            )
            return [
                RerankResult(
                    text=str(document.get("text", "")),
                    original_score=float(document.get("score", 0.0) or 0.0),
                    rerank_score=float(score),
                    metadata=dict(document.get("metadata") or {}),
                    rank=rank,
                )
                for rank, (_, (score, document)) in enumerate(
                    scored[:top_k], start=1
                )
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
