import json
import re
import uuid
from typing import List, Tuple

import pdfplumber

from llm import client as llm
from models.document import TableChunk, TextSectionChunk

# Patterns that suggest a table cross-reference in text
_TABLE_REF_PATTERNS = [
    re.compile(r"table\s+\d+", re.IGNORECASE),
    re.compile(r"see\s+table", re.IGNORECASE),
    re.compile(r"shown\s+in\s+the\s+table", re.IGNORECASE),
    re.compile(r"as\s+per\s+table", re.IGNORECASE),
]

# Patterns to detect section headers
_HEADER_PATTERNS = [
    re.compile(r"^\d+[\.\)]\s+\S"),        # "1. Something" or "1) Something"
    re.compile(r"^[A-Z][A-Z\s]{3,}$"),     # ALL CAPS line
    re.compile(r"^[A-Z][^.!?]{0,60}$"),    # Title-case short line
]


def _is_header(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:
        return False
    return any(p.match(line) for p in _HEADER_PATTERNS)


def _clean_table(rows: List[List]) -> List[List[str]]:
    cleaned = []
    for row in rows:
        cleaned_row = []
        for cell in row:
            cleaned_row.append(str(cell).strip() if cell is not None else "")
        cleaned.append(cleaned_row)
    return cleaned


def _summarize_table(rows: List[List[str]]) -> str:
    table_json = json.dumps(rows[:20])  # Limit to first 20 rows to save tokens
    return llm.complete(
        "You are given a table extracted from a document. Summarize what this table shows in 2-3 sentences, "
        "describing its structure, what each column represents, and the range or key values in each column.",
        f"Table (JSON):\n{table_json}",
    )


def ingest_structured(
    doc_id: str, pdf_path: str, chroma_client
) -> Tuple[List[TableChunk], List[TextSectionChunk]]:
    """Ingest a PDF preserving table structure and section hierarchy."""
    table_chunks: List[TableChunk] = []
    section_chunks: List[TextSectionChunk] = []

    tables_collection = chroma_client.get_or_create_collection(f"tables_{doc_id}")
    sections_collection = chroma_client.get_or_create_collection(f"sections_{doc_id}")

    # Track table chunks per page for cross-reference resolution
    tables_by_page: dict[int, List[TableChunk]] = {}

    with pdfplumber.open(pdf_path) as pdf:
        # --- Pass 1: Extract tables ---
        for page in pdf.pages:
            page_num = page.page_number
            page_tables = page.extract_tables() or []
            page_text_lines = (page.extract_text() or "").split("\n")

            for raw_table in page_tables:
                if not raw_table:
                    continue
                rows = _clean_table(raw_table)

                # Find nearest heading above table on the page
                section_title = None
                for line in reversed(page_text_lines):
                    line = line.strip()
                    if line and _is_header(line):
                        section_title = line
                        break

                summary = _summarize_table(rows)
                chunk_id = str(uuid.uuid4())

                tc = TableChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_number=page_num,
                    section_title=section_title,
                    rows=rows,
                    summary=summary,
                    embedding_text=summary,
                )
                table_chunks.append(tc)
                tables_by_page.setdefault(page_num, []).append(tc)

                embedding = llm.embed(summary)
                tables_collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[summary],
                    metadatas=[{
                        "chunk_id": chunk_id,
                        "page_number": page_num,
                        "section_title": section_title or "",
                    }],
                )

        # --- Pass 2: Extract text sections ---
        for page in pdf.pages:
            page_num = page.page_number
            raw_text = page.extract_text() or ""
            lines = raw_text.split("\n")

            current_title = None
            current_lines: List[str] = []

            def flush_section(title, body_lines, pnum):
                text = "\n".join(body_lines).strip()
                if not text:
                    return
                chunk_id = str(uuid.uuid4())

                # Detect table cross-references
                ref_table_ids = []
                for pattern in _TABLE_REF_PATTERNS:
                    if pattern.search(text):
                        # Link to table chunks on the same or adjacent pages
                        for p in [pnum - 1, pnum, pnum + 1]:
                            for tc in tables_by_page.get(p, []):
                                if tc.chunk_id not in ref_table_ids:
                                    ref_table_ids.append(tc.chunk_id)
                        break

                sc = TextSectionChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_number=pnum,
                    section_title=title,
                    text=text,
                    referenced_table_ids=ref_table_ids,
                )
                section_chunks.append(sc)

                embedding = llm.embed(text[:2000])  # Limit embedding text length
                sections_collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[{
                        "chunk_id": chunk_id,
                        "page_number": pnum,
                        "section_title": title or "",
                    }],
                )

            for line in lines:
                if _is_header(line.strip()):
                    flush_section(current_title, current_lines, page_num)
                    current_title = line.strip()
                    current_lines = []
                else:
                    current_lines.append(line)

            flush_section(current_title, current_lines, page_num)

    return table_chunks, section_chunks
