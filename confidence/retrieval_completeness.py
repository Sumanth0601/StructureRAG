from typing import List

from llm import client as llm
from models.response import DimensionScore


def _format_chunks(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = getattr(c, "summary", None) or getattr(c, "text", "")
        parts.append(f"[Source {i}] {text[:500]}")
    return "\n\n".join(parts)


def score_retrieval_completeness(question: str, retrieved_chunks: List) -> DimensionScore:
    formatted = _format_chunks(retrieved_chunks)
    result = llm.judge(
        "You are evaluating whether the retrieved documents contain enough information to fully answer a question.\n"
        "Assess whether the retrieved documents address all parts of the question. "
        "Are there aspects of the question that none of the documents speak to?\n"
        'Respond with JSON: {"fully_addressed": <bool>, "missing_aspects": [<strings>], "score": <float 0.0-1.0>}',
        f"Question: {question}\n\nRetrieved documents:\n{formatted}",
    )

    if "error" in result:
        return DimensionScore(name="Retrieval Completeness", score=0.5, explanation="Could not evaluate retrieval completeness.")

    score = float(result.get("score", 0.5))
    missing = result.get("missing_aspects", [])
    if missing:
        explanation = f"Missing coverage: {'; '.join(str(m) for m in missing[:2])}"
    else:
        explanation = "Retrieved documents fully address the question."

    return DimensionScore(name="Retrieval Completeness", score=score, explanation=explanation)
