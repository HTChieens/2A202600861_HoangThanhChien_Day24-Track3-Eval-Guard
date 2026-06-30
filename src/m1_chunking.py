from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    similarities: list[float] = []
    if len(sentences) > 1:
        try:
            from numpy import dot
            from numpy.linalg import norm

            embeddings = _semantic_model().encode(
                sentences, convert_to_numpy=True, show_progress_bar=False
            )
            similarities = [
                float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
                for a, b in zip(embeddings, embeddings[1:])
            ]
        except (ImportError, OSError, RuntimeError, ValueError):
            # Keep the chunker usable in offline/minimal environments. This
            # fallback approximates topical continuity through token overlap.
            similarities = [
                _lexical_similarity(a, b)
                for a, b in zip(sentences, sentences[1:])
            ]

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


@lru_cache(maxsize=1)
def _semantic_model():
    """Load the embedding model once; avoid network access when it is not cached."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except TypeError:
        # Compatibility with older sentence-transformers releases.
        return SentenceTransformer("all-MiniLM-L6-v2")


def _lexical_similarity(left: str, right: str) -> float:
    """Jaccard similarity used only when sentence embeddings are unavailable."""
    tokenize = lambda value: set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))
    left_tokens, right_tokens = tokenize(left), tokenize(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    # Map lexical overlap to the upper half of the cosine range. This keeps a
    # moderate threshold useful even though sparse word sets score much lower
    # than dense sentence embeddings.
    return 0.5 + 0.5 * (len(left_tokens & right_tokens) / len(union))


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")

    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return ([], [])

    parent_texts = _pack_text_units(paragraphs, parent_size, "\n\n")
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": parent_id,
                "chunk_index": parent_index,
            },
        ))

        child_units = [
            unit.strip()
            for unit in re.split(r"(?<=[.!?])\s+|\n\s*\n", parent_text)
            if unit.strip()
        ]
        for child_text in _pack_text_units(child_units, child_size, " "):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **metadata,
                    "chunk_type": "child",
                    "chunk_index": len(children),
                },
                parent_id=parent_id,
            ))

    return parents, children


def _pack_text_units(units: list[str], max_size: int, separator: str) -> list[str]:
    """Pack logical text units without exceeding max_size, splitting only as needed."""
    normalized: list[str] = []
    for unit in units:
        unit = unit.strip()
        if len(unit) <= max_size:
            normalized.append(unit)
            continue

        # Prefer word boundaries; hard-split exceptionally long tokens.
        words = unit.split()
        current = ""
        for word in words:
            pieces = [
                word[i:i + max_size] for i in range(0, len(word), max_size)
            ] if len(word) > max_size else [word]
            for piece in pieces:
                candidate = f"{current} {piece}".strip()
                if current and len(candidate) > max_size:
                    normalized.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            normalized.append(current)

    packed: list[str] = []
    current = ""
    for unit in normalized:
        candidate = f"{current}{separator}{unit}" if current else unit
        if current and len(candidate) > max_size:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    sections: list[tuple[str, list[str]]] = []
    header = ""
    content: list[str] = []
    in_code_block = False

    def flush_section() -> None:
        nonlocal content
        section_text = "\n".join(
            ([header] if header else []) + content
        ).strip()
        if section_text:
            sections.append((header, [section_text]))
        content = []

    for line in text.splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            in_code_block = not in_code_block

        if not in_code_block and re.match(r"^#{1,3}\s+\S", line):
            flush_section()
            header = line.strip()
        else:
            content.append(line)
    flush_section()

    return [
        Chunk(
            text=lines[0],
            metadata={
                **metadata,
                "section": section_header,
                "strategy": "structure",
                "chunk_index": index,
            },
        )
        for index, (section_header, lines) in enumerate(sections)
    ]


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
