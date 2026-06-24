from typing import List

from models.response import ConfidenceReport, DimensionScore
from confidence.source_grounding import score_source_grounding
from confidence.retrieval_completeness import score_retrieval_completeness
from confidence.factual_consistency import score_factual_consistency
from confidence.structural_completeness import score_structural_completeness
from confidence.confidence_calibration import score_confidence_calibration

WEIGHTS = {
    "Source Grounding": 0.30,
    "Retrieval Completeness": 0.25,
    "Factual Consistency": 0.25,
    "Structural Completeness": 0.10,
    "Confidence Calibration": 0.10,
}

RECOMMENDATIONS = {
    "Source Grounding": (
        "Answer contains claims not supported by retrieved sources. "
        "Consider re-querying with more specific terms or reviewing source documents directly."
    ),
    "Retrieval Completeness": (
        "Retrieved documents do not fully address the question. "
        "Try rephrasing the question or uploading additional documents."
    ),
    "Factual Consistency": (
        "The answer may contradict source documents. "
        "Verify the answer against the original document before using it."
    ),
    "Structural Completeness": (
        "Answer may be incomplete — retrieval did not return a structured table for this financial question. "
        "Recommend re-querying with table-aware retrieval."
    ),
    "Confidence Calibration": (
        "The answer's expressed certainty does not match retrieval quality. "
        "Treat definitive statements with caution when grounding is weak."
    ),
}


def score(
    question: str,
    answer: str,
    retrieved_chunks: List,
    question_type: str,
) -> ConfidenceReport:
    sg = score_source_grounding(answer, retrieved_chunks)
    rc = score_retrieval_completeness(question, retrieved_chunks)
    fc = score_factual_consistency(answer, retrieved_chunks)
    sc = score_structural_completeness(question, retrieved_chunks, question_type)
    cc = score_confidence_calibration(answer, sg.score)

    dimensions = [sg, rc, fc, sc, cc]

    overall = sum(d.score * WEIGHTS[d.name] for d in dimensions)

    weakest = min(dimensions, key=lambda d: d.score)
    recommendation = RECOMMENDATIONS.get(weakest.name, "Review the answer carefully.")

    return ConfidenceReport(
        overall_score=round(overall, 3),
        dimensions=dimensions,
        weakest_dimension=weakest.name,
        recommendation=recommendation,
    )
