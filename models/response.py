from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourceChunk:
    chunk_id: str
    chunk_type: str  # "flat", "table", or "section"
    text: str
    page_number: int


@dataclass
class DimensionScore:
    name: str
    score: float
    explanation: str


@dataclass
class ConfidenceReport:
    overall_score: float
    dimensions: List[DimensionScore]
    weakest_dimension: str
    recommendation: str


@dataclass
class PipelineAnswer:
    pipeline: str  # "flat_rag" or "structured_rag"
    answer: str
    retrieved_chunks: List[SourceChunk]
    confidence: ConfidenceReport


@dataclass
class QueryResponse:
    question: str
    flat_rag: PipelineAnswer
    structured_rag: PipelineAnswer
