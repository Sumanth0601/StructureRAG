from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class FlatChunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    chunk_index: int


@dataclass
class TableChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    section_title: Optional[str]
    rows: List[List[str]]
    summary: str
    embedding_text: str


@dataclass
class TextSectionChunk:
    chunk_id: str
    doc_id: str
    page_number: int
    section_title: Optional[str]
    text: str
    referenced_table_ids: List[str] = field(default_factory=list)


@dataclass
class IngestedDocument:
    doc_id: str
    filename: str
    flat_chunks: List[FlatChunk]
    table_chunks: List[TableChunk]
    section_chunks: List[TextSectionChunk]
