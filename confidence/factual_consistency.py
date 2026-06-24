from typing import List

from llm import client as llm
from models.response import DimensionScore


def _format_chunks(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = getattr(c, "summary", None) or getattr(c, "text", "")
        parts.append(f"[Source {i}] {text[:500]}")
    return "\n\n".join(parts)


def score_factual_consistency(answer: str, retrieved_chunks: List) -> DimensionScore:
    formatted = _format_chunks(retrieved_chunks)
    result = llm.judge(
        "You are checking whether an AI answer contradicts any of the source documents it was based on.\n"
        "Identify any statements in the answer that directly contradict information in the source documents.\n"
        'Respond with JSON: {"contradictions_found": <int>, "contradictions": [<strings>], "score": <float>}\n'
        "Score: 1.0 for 0 contradictions, 0.7 for 1, 0.4 for 2, 0.0 for 3+.",
        f"Source documents:\n{formatted}\n\nAI Answer:\n{answer}",
    )

    if "error" in result:
        return DimensionScore(name="Factual Consistency", score=0.5, explanation="Could not evaluate factual consistency.")

    n = int(result.get("contradictions_found", 0))
    score_map = {0: 1.0, 1: 0.7, 2: 0.4}
    score = score_map.get(n, 0.0) if n <= 2 else 0.0
    # Use LLM-provided score if reasonable
    llm_score = result.get("score")
    if llm_score is not None:
        score = float(llm_score)

    contradictions = result.get("contradictions", [])
    if contradictions:
        explanation = f"{n} contradiction(s) found: {'; '.join(str(c) for c in contradictions[:2])}"
    else:
        explanation = "No contradictions found between answer and sources."

    return DimensionScore(name="Factual Consistency", score=score, explanation=explanation)
