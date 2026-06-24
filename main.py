import asyncio
import dataclasses
import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Union

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from ingest.flat_rag import ingest_flat, retrieve_flat
from ingest.structured_rag import ingest_structured
from retrieval.structured_retriever import retrieve_structured
from confidence import scorer as confidence_scorer
from models.document import IngestedDocument, FlatChunk, TableChunk, TextSectionChunk
from models.response import PipelineAnswer, QueryResponse, SourceChunk
import llm.client as llm_client

app = FastAPI(title="StructureRAG Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores
chroma_client = chromadb.Client()
documents: Dict[str, dict] = {}  # doc_id -> {doc, ingested_at}


def _dataclass_to_dict(obj):
    """Recursively convert dataclasses to dicts for JSON serialization."""
    if dataclasses.is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_dataclass_to_dict(i) for i in obj]
    return obj


def _chunk_to_source(chunk) -> SourceChunk:
    if isinstance(chunk, FlatChunk):
        return SourceChunk(chunk_id=chunk.chunk_id, chunk_type="flat", text=chunk.text, page_number=chunk.page_number)
    if isinstance(chunk, TableChunk):
        return SourceChunk(chunk_id=chunk.chunk_id, chunk_type="table", text=chunk.summary, page_number=chunk.page_number)
    if isinstance(chunk, TextSectionChunk):
        return SourceChunk(chunk_id=chunk.chunk_id, chunk_type="section", text=chunk.text, page_number=chunk.page_number)
    return SourceChunk(chunk_id="unknown", chunk_type="unknown", text=str(chunk), page_number=0)


ANSWER_SYSTEM_PROMPT = (
    "You are answering a question based strictly on the provided source documents. "
    "If the information needed to answer is not in the sources, say so explicitly. "
    "Do not invent information."
)


def _build_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        text = getattr(c, "summary", None) or getattr(c, "text", "")
        chunk_type = "Table" if isinstance(c, TableChunk) else "Section" if isinstance(c, TextSectionChunk) else "Text"
        parts.append(f"[Source {i} — {chunk_type}, Page {c.page_number}]\n{text[:800]}")
    return "\n\n".join(parts)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"error": "Only PDF files are supported."})

    doc_id = str(uuid.uuid4())
    suffix = f"_{file.filename}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loop = asyncio.get_event_loop()

        flat_chunks = await loop.run_in_executor(
            None, ingest_flat, doc_id, tmp_path, chroma_client
        )
        table_chunks, section_chunks = await loop.run_in_executor(
            None, ingest_structured, doc_id, tmp_path, chroma_client
        )

        doc = IngestedDocument(
            doc_id=doc_id,
            filename=file.filename,
            flat_chunks=flat_chunks,
            table_chunks=table_chunks,
            section_chunks=section_chunks,
        )
        documents[doc_id] = {"doc": doc, "ingested_at": datetime.utcnow().isoformat()}

    finally:
        os.unlink(tmp_path)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "flat_chunks": len(flat_chunks),
        "table_chunks": len(table_chunks),
        "section_chunks": len(section_chunks),
    }


class QueryRequest(BaseModel):
    doc_id: str
    question: str


@app.post("/api/query")
async def query_document(req: QueryRequest):
    if req.doc_id not in documents:
        raise HTTPException(status_code=404, detail={"error": "Document not found.", "detail": f"doc_id={req.doc_id}"})

    doc: IngestedDocument = documents[req.doc_id]["doc"]
    loop = asyncio.get_event_loop()

    # Run both retrievals concurrently
    flat_future = loop.run_in_executor(
        None, retrieve_flat, req.question, req.doc_id, chroma_client
    )
    structured_future = loop.run_in_executor(
        None, retrieve_structured,
        req.question, req.doc_id, chroma_client,
        doc.table_chunks, doc.section_chunks,
    )

    flat_chunks_result, structured_result = await asyncio.gather(flat_future, structured_future)
    structured_chunks_result, q_type = structured_result

    # Generate answers
    def gen_answer(chunks):
        context = _build_context(chunks)
        return llm_client.complete(
            ANSWER_SYSTEM_PROMPT,
            f"Sources:\n{context}\n\nQuestion: {req.question}",
        )

    flat_answer_future = loop.run_in_executor(None, gen_answer, flat_chunks_result)
    struct_answer_future = loop.run_in_executor(None, gen_answer, structured_chunks_result)

    flat_answer, struct_answer = await asyncio.gather(flat_answer_future, struct_answer_future)

    # Score confidence
    def score_flat():
        return confidence_scorer.score(req.question, flat_answer, flat_chunks_result, q_type)

    def score_struct():
        return confidence_scorer.score(req.question, struct_answer, structured_chunks_result, q_type)

    flat_conf_future = loop.run_in_executor(None, score_flat)
    struct_conf_future = loop.run_in_executor(None, score_struct)

    flat_conf, struct_conf = await asyncio.gather(flat_conf_future, struct_conf_future)

    response = QueryResponse(
        question=req.question,
        flat_rag=PipelineAnswer(
            pipeline="flat_rag",
            answer=flat_answer,
            retrieved_chunks=[_chunk_to_source(c) for c in flat_chunks_result],
            confidence=flat_conf,
        ),
        structured_rag=PipelineAnswer(
            pipeline="structured_rag",
            answer=struct_answer,
            retrieved_chunks=[_chunk_to_source(c) for c in structured_chunks_result],
            confidence=struct_conf,
        ),
    )

    return JSONResponse(content=_dataclass_to_dict(response))


@app.get("/api/documents")
def list_documents():
    result = []
    for doc_id, entry in documents.items():
        doc = entry["doc"]
        result.append({
            "doc_id": doc_id,
            "filename": doc.filename,
            "ingested_at": entry["ingested_at"],
        })
    return result


# Serve frontend — must be last to not shadow API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
