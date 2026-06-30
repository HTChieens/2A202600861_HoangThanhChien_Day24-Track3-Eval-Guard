from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os, sys
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


MODEL_NAME = "gpt-4o-mini"
VALID_METHODS = {"summary", "hyqa", "contextual", "metadata", "combined"}


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    text = str(text or "").strip()
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            return _chat(
                "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt. "
                "Chỉ trả về phần tóm tắt.",
                text,
                max_tokens=150,
            )
        except Exception as error:
            print(f"  ⚠️  OpenAI summarize failed: {error}")
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    text = str(text or "").strip()
    if not text or n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            response = _chat(
                f"Dựa trên đoạn văn, tạo đúng {n_questions} câu hỏi mà đoạn văn "
                "có thể trả lời. Trả về mỗi câu hỏi trên một dòng.",
                text,
                max_tokens=200,
            )
            questions = _parse_question_lines(response)
            if questions:
                return questions[:n_questions]
        except Exception as error:
            print(f"  ⚠️  OpenAI HyQA failed: {error}")
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    text = str(text or "")
    if not text:
        return ""
    if OPENAI_API_KEY:
        try:
            context = _chat(
                "Viết một câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và "
                "nói về chủ đề gì. Chỉ trả về một câu.",
                f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}",
                max_tokens=80,
            )
            if context:
                return f"{context}\n\n{text}"
        except Exception as error:
            print(f"  ⚠️  OpenAI contextual failed: {error}")

    topic = _infer_topic(text)
    source = document_title.strip() or "tài liệu hiện tại"
    return f"Ngữ cảnh: Trích từ {source}, nội dung về {topic}.\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    text = str(text or "").strip()
    if not text:
        return _fallback_metadata("")
    if OPENAI_API_KEY:
        try:
            response = _chat(
                'Trích xuất metadata và chỉ trả về JSON hợp lệ theo schema: '
                '{"topic":"...", "entities":["..."], '
                '"category":"policy|hr|it|finance", "language":"vi|en"}.',
                text,
                max_tokens=180,
                response_format={"type": "json_object"},
            )
            return _normalize_metadata(_parse_json_object(response), text)
        except Exception as error:
            print(f"  ⚠️  OpenAI metadata failed: {error}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    text = str(text or "").strip()
    source = str(source or "").strip()
    if OPENAI_API_KEY and text:
        try:
            response = _chat(
                """Phân tích đoạn văn và chỉ trả về JSON hợp lệ:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "một câu mô tả vị trí và chủ đề của đoạn văn",
  "metadata": {
    "topic": "...",
    "entities": ["..."],
    "category": "policy|hr|it|finance",
    "language": "vi|en"
  }
}""",
                f"Tài liệu: {source or 'Không rõ'}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            result = _parse_json_object(response)
            return {
                "summary": str(result.get("summary", "")).strip()
                or _extractive_summary(text),
                "questions": _normalize_questions(result.get("questions"), text, 3),
                "context": str(result.get("context", "")).strip()
                or _fallback_context(text, source),
                "metadata": _normalize_metadata(result.get("metadata"), text),
            }
        except Exception as error:
            print(f"  ⚠️  Enrichment API failed: {error}")
    return {
        "summary": _extractive_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": _fallback_context(text, source),
        "metadata": _fallback_metadata(text),
    }


def _chat(system_prompt: str, user_prompt: str, max_tokens: int,
          response_format: dict | None = None) -> str:
    from openai import OpenAI

    kwargs = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def _parse_json_object(value: str) -> dict:
    cleaned = str(value or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]


def _extractive_summary(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return text.strip()
    summary = " ".join(sentences[:2])
    return summary[:600].strip()


def _parse_question_lines(value: str) -> list[str]:
    return _normalize_questions(value.splitlines(), "", len(value.splitlines()))


def _normalize_questions(value, text: str, limit: int) -> list[str]:
    if isinstance(value, str):
        candidates = value.splitlines()
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []
    questions = []
    for candidate in candidates:
        question = re.sub(
            r"^\s*(?:[-*•]|\d+[\s.)-]+)\s*", "", str(candidate)
        ).strip()
        if not question:
            continue
        if not question.endswith("?"):
            question += "?"
        if question not in questions:
            questions.append(question)
    return questions[:limit] or _fallback_questions(text, limit)


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    questions = []
    for sentence in _sentences(text):
        statement = sentence.rstrip(".!?").strip()
        if len(statement) < 8:
            continue
        lower = statement.casefold()
        if re.search(r"\d", statement):
            question = f"Quy định hoặc giá trị cụ thể liên quan đến {statement} là gì?"
        elif any(word in lower for word in ("không", "phải", "cần", "được")):
            question = f"Điều kiện hoặc yêu cầu trong nội dung “{statement}” là gì?"
        else:
            question = f"Nội dung chính về “{statement}” là gì?"
        questions.append(question)
        if len(questions) >= n_questions:
            break
    return questions


def _fallback_context(text: str, source: str) -> str:
    return (
        f"Ngữ cảnh: Trích từ {source or 'tài liệu hiện tại'}, "
        f"nội dung về {_infer_topic(text)}."
    )


def _infer_topic(text: str) -> str:
    first_line = next(
        (line.strip().lstrip("#").strip() for line in text.splitlines()
         if line.strip()),
        "",
    )
    words = re.findall(r"\w+", first_line, flags=re.UNICODE)
    return " ".join(words[:10]) or "chính sách và quy định"


def _fallback_metadata(text: str) -> dict:
    lowered = text.casefold()
    category_keywords = {
        "it": ("mật khẩu", "vpn", "malware", "dữ liệu", "cntt", "bảo mật"),
        "finance": ("lương", "chi phí", "vnđ", "phụ cấp", "hoàn trả", "mua sắm"),
        "hr": ("nhân viên", "nghỉ phép", "thử việc", "mentor", "đào tạo"),
    }
    category = "policy"
    best_count = 0
    for candidate, keywords in category_keywords.items():
        count = sum(keyword in lowered for keyword in keywords)
        if count > best_count:
            category, best_count = candidate, count

    entities = re.findall(
        r"\b(?:[A-ZĐ][\wÀ-ỹ-]*(?:\s+[A-ZĐ][\wÀ-ỹ-]*)+|"
        r"[A-Z]{2,}(?:-\d+)?)\b",
        text,
        flags=re.UNICODE,
    )
    return {
        "topic": _infer_topic(text),
        "entities": list(dict.fromkeys(entities))[:10],
        "category": category,
        "language": "vi" if re.search(r"[À-ỹĐđ]", text) else "en",
    }


def _normalize_metadata(value, text: str) -> dict:
    fallback = _fallback_metadata(text)
    if not isinstance(value, dict):
        return fallback
    category = str(value.get("category", fallback["category"])).casefold()
    if category not in {"policy", "hr", "it", "finance"}:
        category = fallback["category"]
    language = str(value.get("language", fallback["language"])).casefold()
    if language not in {"vi", "en"}:
        language = fallback["language"]
    entities = value.get("entities", fallback["entities"])
    if not isinstance(entities, list):
        entities = [str(entities)] if entities else []
    return {
        "topic": str(value.get("topic", fallback["topic"])).strip()
        or fallback["topic"],
        "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]
    methods = list(dict.fromkeys(methods))
    unknown_methods = set(methods) - VALID_METHODS
    if unknown_methods:
        raise ValueError(
            f"Unknown enrichment methods: {', '.join(sorted(unknown_methods))}"
        )
    if not methods:
        return []
    if "combined" in methods and len(methods) > 1:
        raise ValueError("'combined' cannot be mixed with individual methods")

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = str(chunk.get("text", ""))
        original_metadata = dict(chunk.get("metadata") or {})
        source = str(original_metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enrichment_parts = []
            if context_line:
                enrichment_parts.append(context_line)
            if summary:
                enrichment_parts.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment_parts.append(
                    "Câu hỏi liên quan:\n- " + "\n- ".join(questions)
                )
            enrichment_parts.append(text)
            enriched_text = "\n\n".join(enrichment_parts)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            if summary:
                enriched_text = f"Tóm tắt: {summary}\n\n{enriched_text}"
            if questions:
                enriched_text = (
                    "Câu hỏi liên quan:\n- "
                    + "\n- ".join(questions)
                    + "\n\n"
                    + enriched_text
                )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**original_metadata, **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
