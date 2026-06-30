from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

DIAGNOSTIC_TREE = {
    "faithfulness": (
        "The answer contains claims that are not supported by the retrieved context.",
        "Tighten the grounded-answer prompt, require citations, and lower generation temperature.",
    ),
    "context_recall": (
        "Retrieval missed information needed to answer the question completely.",
        "Improve chunking, increase retrieval depth, and combine semantic search with BM25.",
    ),
    "context_precision": (
        "Retrieval returned too many irrelevant or weakly related chunks.",
        "Add cross-encoder reranking, metadata filters, and reduce the final context count.",
    ),
    "answer_relevancy": (
        "The answer does not directly address the user's question.",
        "Improve the answer prompt and explicitly require a concise response to the exact question.",
    ),
}


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {
        len(questions), len(answers), len(contexts), len(ground_truths)
    }
    if len(lengths) != 1:
        raise ValueError(
            "questions, answers, contexts, and ground_truths must have equal lengths"
        )
    if not questions:
        return _empty_evaluation()

    normalized_contexts = [
        [str(context) for context in (context_list or [])]
        for context_list in contexts
    ]

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": [str(value) for value in questions],
            "answer": [str(value) for value in answers],
            "contexts": normalized_contexts,
            "ground_truth": [str(value) for value in ground_truths],
        })
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        frame = result.to_pandas()
        per_question = []
        for index, row in frame.iterrows():
            position = int(index) if isinstance(index, int) else len(per_question)
            position = min(position, len(questions) - 1)
            per_question.append(EvalResult(
                question=str(row.get("question", questions[position])),
                answer=str(row.get("answer", answers[position])),
                contexts=_as_context_list(
                    row.get("contexts", normalized_contexts[position])
                ),
                ground_truth=str(
                    row.get("ground_truth", ground_truths[position])
                ),
                **{
                    metric: _safe_score(row.get(metric, 0.0))
                    for metric in METRIC_NAMES
                },
            ))

        aggregate = {}
        for metric in METRIC_NAMES:
            row_scores = [getattr(item, metric) for item in per_question]
            aggregate[metric] = (
                sum(row_scores) / len(row_scores) if row_scores else 0.0
            )
        return {**aggregate, "per_question": per_question}
    except Exception as error:
        print(f"  ⚠️  RAGAS evaluation failed: {error}")
        return _evaluate_locally(
            questions, answers, normalized_contexts, ground_truths
        )


def _empty_evaluation() -> dict:
    return {**{metric: 0.0 for metric in METRIC_NAMES}, "per_question": []}


def _safe_score(value) -> float:
    """Convert metric values to finite floats suitable for JSON and sorting."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _as_context_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "và", "là", "có", "được", "cho", "của", "trong", "một", "các",
        "theo", "khi", "này", "đó", "thì", "phải", "với", "từ", "bao",
        "nhiêu", "nhân", "viên", "không", "gì", "như", "sau", "trước",
    }
    return {
        token for token in re.findall(r"\w+", str(text).casefold(), re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }


def _overlap_recall(reference: str, candidate: str) -> float:
    reference_tokens = _content_tokens(reference)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & _content_tokens(candidate)) / len(reference_tokens)


def _evaluate_locally(questions, answers, contexts, ground_truths) -> dict:
    """Deterministic lexical proxy when RAGAS/LLM evaluation is unavailable."""
    per_question = []
    for question, answer, context_list, ground_truth in zip(
        questions, answers, contexts, ground_truths
    ):
        joined_context = " ".join(context_list)
        faithfulness_score = _overlap_recall(answer, joined_context)
        answer_relevancy_score = max(
            _overlap_recall(question, answer),
            _overlap_recall(ground_truth, answer),
        )
        relevant_contexts = [
            _overlap_recall(ground_truth, context) for context in context_list
        ]
        context_precision_score = (
            sum(score > 0.12 for score in relevant_contexts) / len(relevant_contexts)
            if relevant_contexts else 0.0
        )
        context_recall_score = _overlap_recall(ground_truth, joined_context)
        per_question.append(EvalResult(
            question=str(question),
            answer=str(answer),
            contexts=list(context_list),
            ground_truth=str(ground_truth),
            faithfulness=faithfulness_score,
            answer_relevancy=answer_relevancy_score,
            context_precision=context_precision_score,
            context_recall=context_recall_score,
        ))
    aggregate = {
        metric: (
            sum(getattr(item, metric) for item in per_question) / len(per_question)
            if per_question else 0.0
        )
        for metric in METRIC_NAMES
    }
    return {**aggregate, "per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0 or not eval_results:
        return []

    analyzed = []
    for position, result in enumerate(eval_results):
        get_value = (
            result.get
            if isinstance(result, dict)
            else lambda name, default=None: getattr(result, name, default)
        )
        scores = {
            metric: _safe_score(get_value(metric, 0.0))
            for metric in METRIC_NAMES
        }
        worst_metric = min(
            METRIC_NAMES, key=lambda metric: (scores[metric], METRIC_NAMES.index(metric))
        )
        average_score = sum(scores.values()) / len(scores)
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[worst_metric]
        analyzed.append({
            "question": str(get_value("question", "")),
            "answer": str(get_value("answer", "")),
            "ground_truth": str(get_value("ground_truth", "")),
            "contexts": _as_context_list(get_value("contexts", [])),
            "average_score": round(average_score, 6),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "metric_scores": scores,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "_position": position,
        })

    analyzed.sort(key=lambda item: (item["average_score"], item["_position"]))
    failures = analyzed[:bottom_n]
    for failure in failures:
        failure.pop("_position", None)
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
