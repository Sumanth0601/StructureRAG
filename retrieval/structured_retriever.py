import json
import uuid
from typing import List, Union

from llm import client as llm
from models.document import TableChunk, TextSectionChunk


def classify_question(question: str) -> str:
    """Classify question as 'table_query', 'clause_query', or 'summary_query'."""
    result = llm.judge(
        "Classify the following question into exactly one category.\n"
        "Categories:\n"
        "- table_query: asks for numbers, totals, comparisons, or specific data points likely in a table\n"
        "- clause_query: asks about conditions, terms, policies, requirements, or legal/contractual provisions\n"
        "- summary_query: asks for a summary, overview, or synthesis\n"
        'Respond with JSON: {"type": "<category>"}',
        f"Question: {question}",
    )
    q_type = result.get("type", "summary_query")
    if q_type not in ("table_query", "clause_query", "summary_query"):
        q_type = "summary_query"
    return q_type


def retrieve_structured(
    question: str,
    doc_id: str,
    chroma_client,
    table_chunks_store: List[TableChunk],
    section_chunks_store: List[TextSectionChunk],
    top_k: int = 5,
) -> tuple[List[Union[TableChunk, TextSectionChunk]], str]:
    """Retrieve structured chunks. Returns (chunks, question_type)."""
    q_type = classify_question(question)
    embedding = llm.embed(question)

    tables_col = chroma_client.get_or_create_collection(f"tables_{doc_id}")
    sections_col = chroma_client.get_or_create_collection(f"sections_{doc_id}")

    # Helper to build chunk objects from chroma results
    table_map = {tc.chunk_id: tc for tc in table_chunks_store}
    section_map = {sc.chunk_id: sc for sc in section_chunks_store}

    def query_tables(n: int) -> List[TableChunk]:
        count = tables_col.count()
        if count == 0:
            return []
        results = tables_col.query(query_embeddings=[embedding], n_results=min(n, count))
        found = []
        for cid in results["ids"][0]:
            if cid in table_map:
                found.append(table_map[cid])
        return found

    def query_sections(n: int) -> List[TextSectionChunk]:
        count = sections_col.count()
        if count == 0:
            return []
        results = sections_col.query(query_embeddings=[embedding], n_results=min(n, count))
        found = []
        for cid in results["ids"][0]:
            if cid in section_map:
                found.append(section_map[cid])
        return found

    seen_ids = set()
    combined: List[Union[TableChunk, TextSectionChunk]] = []

    def add(chunk):
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            combined.append(chunk)

    if q_type == "table_query":
        for tc in query_tables(3):
            add(tc)
        for sc in query_sections(2):
            add(sc)
            for tid in sc.referenced_table_ids:
                if tid in table_map:
                    add(table_map[tid])

    elif q_type == "clause_query":
        for sc in query_sections(5):
            add(sc)
            for tid in sc.referenced_table_ids:
                if tid in table_map:
                    add(table_map[tid])

    else:  # summary_query
        for tc in query_tables(3):
            add(tc)
        for sc in query_sections(3):
            add(sc)

    return combined[:top_k + 2], q_type  # Allow a few extra for linked chunks
