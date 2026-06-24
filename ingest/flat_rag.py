import re
import uuid
from typing import List

import pdfplumber

from llm import client as llm
from models.document import FlatChunk


def _chunk_text(text: str, page_map: List[tuple], chunk_words: int = 400, overlap_words: int = 50) -> List[dict]:
    """Split text into overlapping word-count chunks, estimating page numbers."""
    # Tokenize by whitespace
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_words_list = words[start:end]
        chunk_text = " ".join(chunk_words_list)

        # Estimate page number from character offset
        char_offset = len(" ".join(words[:start]))
        page_number = 1
        for page_num, (page_start, page_end) in enumerate(page_map, start=1):
            if page_start <= char_offset < page_end:
                page_number = page_num
                break

        chunks.append({"text": chunk_text, "page_number": page_number})
        if end == len(words):
            break
        start = end - overlap_words

    return chunks


def ingest_flat(doc_id: str, pdf_path: str, chroma_client) -> List[FlatChunk]:
    """Ingest a PDF using flat chunking strategy."""
    full_text = ""
    page_map = []  # (start_char, end_char) per page

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            start = len(full_text)
            full_text += text + "\n"
            page_map.append((start, len(full_text)))

    raw_chunks = _chunk_text(full_text, page_map)

    collection = chroma_client.get_or_create_collection(f"flat_{doc_id}")
    flat_chunks = []

    for idx, chunk_data in enumerate(raw_chunks):
        chunk_id = str(uuid.uuid4())
        text = chunk_data["text"].strip()
        if not text:
            continue

        embedding = llm.embed(text)
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"chunk_id": chunk_id, "page_number": chunk_data["page_number"], "chunk_index": idx}],
        )

        flat_chunks.append(FlatChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            page_number=chunk_data["page_number"],
            chunk_index=idx,
        ))

    return flat_chunks


def retrieve_flat(question: str, doc_id: str, chroma_client, top_k: int = 5) -> List[FlatChunk]:
    """Retrieve top-k flat chunks for a question."""
    collection = chroma_client.get_or_create_collection(f"flat_{doc_id}")
    embedding = llm.embed(question)
    results = collection.query(query_embeddings=[embedding], n_results=min(top_k, collection.count()))

    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return chunks

    for i, chunk_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        text = results["documents"][0][i]
        chunks.append(FlatChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            page_number=meta.get("page_number", 1),
            chunk_index=meta.get("chunk_index", i),
        ))

    return chunks
