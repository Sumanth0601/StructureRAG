from typing import List

from llm import client as llm
from models.response import DimensionScore


def _format_chunks(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = getattr(c, "summary", None) or getattr(c, "text", "")
        parts.append(f"[Source {i}] {text[:500]}")
    return "\n\n".join(parts)


def score_source_grounding(answer: str, retrieved_chunks: List) -> DimensionScore:
    formatted = _format_chunks(retrieved_chunks)
    result = llm.judge(
        "You are evaluating whether an AI answer is fully supported by the retrieved source documents.\n"
        "Identify each factual claim in the answer. For each claim, determine if it is directly supported by one of the retrieved sources.\n"
        'Respond with JSON: {"total_claims": <int>, "grounded_claims": <int>, "unsupported_claims": [<strings>], "score": <float 0.0-1.0>}',
        f"Retrieved sources:\n{formatted}\n\nAI Answer:\n{answer}",
    )

    if "error" in result:
        return DimensionScore(name="Source Grounding", score=0.5, explanation="Could not evaluate source grounding.")

    score = float(result.get("score", 0.5))
    unsupported = result.get("unsupported_claims", [])
    if unsupported:
        explanation = f"{len(unsupported)} claim(s) lack source support: {'; '.join(str(u) for u in unsupported[:2])}"
    else:
        explanation = "All claims are supported by retrieved sources."

    return DimensionScore(name="Source Grounding", score=score, explanation=explanation)
