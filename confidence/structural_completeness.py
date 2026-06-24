from typing import List

from models.document import TableChunk
from models.response import DimensionScore


def score_structural_completeness(
    question: str, retrieved_chunks: List, question_type: str
) -> DimensionScore:
    if question_type != "table_query":
        return DimensionScore(
            name="Structural Completeness",
            score=1.0,
            explanation="Not applicable for this question type.",
        )

    has_table = any(isinstance(c, TableChunk) for c in retrieved_chunks)
    if has_table:
        return DimensionScore(
            name="Structural Completeness",
            score=1.0,
            explanation="A structured table was retrieved for this data question.",
        )
    else:
        return DimensionScore(
            name="Structural Completeness",
            score=0.3,
            explanation=(
                "This question asks for tabular data but only flat text was retrieved — "
                "the answer may be incomplete or incorrect."
            ),
        )
