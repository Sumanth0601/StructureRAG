from models.response import DimensionScore

_HEDGE_PHRASES = [
    "appears to", "based on the available", "approximately", "may be",
    "it seems", "i cannot confirm", "the document suggests", "likely",
    "possibly", "it is unclear",
]

_OVERCONFIDENT_PHRASES = [
    "exactly", "the answer is", "the total is", "definitively",
    "clearly states", "is exactly", "the figure is", "it is confirmed",
]


def score_confidence_calibration(answer: str, source_grounding_score: float) -> DimensionScore:
    answer_lower = answer.lower()
    has_hedge = any(p in answer_lower for p in _HEDGE_PHRASES)
    has_overconfidence = any(p in answer_lower for p in _OVERCONFIDENT_PHRASES)

    if source_grounding_score < 0.6 and has_overconfidence:
        return DimensionScore(
            name="Confidence Calibration",
            score=0.2,
            explanation="Answer uses definitive language but source grounding is weak.",
        )
    if source_grounding_score < 0.6 and has_hedge:
        return DimensionScore(
            name="Confidence Calibration",
            score=0.9,
            explanation="Answer appropriately hedges given uncertain retrieval.",
        )
    if source_grounding_score >= 0.8 and has_hedge:
        return DimensionScore(
            name="Confidence Calibration",
            score=0.7,
            explanation="Answer is overly cautious given strong source grounding.",
        )
    return DimensionScore(
        name="Confidence Calibration",
        score=1.0,
        explanation="Answer confidence aligns with retrieval quality.",
    )
