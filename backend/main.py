"""
Main FastAPI Application - Document Chatbot API Server.
"""
import os
import uuid
import shutil
import json
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.config import (
    UPLOAD_DIR,
    SUPPORTED_EXTENSIONS,
    TOP_K,
    MAX_UPLOAD_BYTES,
    API_KEY,
    CORS_ORIGINS,
    ENABLE_QUERY_REWRITE,
    ENABLE_CLARIFICATION_GATE,
    ENABLE_SELF_CHECK,
    CLARIFICATION_MIN_TOP_SIMILARITY,
    CLARIFICATION_MARGIN_MIN,
    CLARIFICATION_HIGH_CONFIDENCE,
    SELF_CHECK_MIN_GROUNDEDNESS,
    FASTEST_RESPONSE_MODE,
)
from backend.document_processor import process_document
from backend.vector_store import add_documents, search, delete_document, get_all_documents, get_stats
from backend.generator import (
    generate_answer,
    is_model_loaded,
    rewrite_query,
    build_clarification_question,
    groundedness_score,
    generate_extractive_answer,
)

app = FastAPI(title="Company Document Chatbot", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=bool(API_KEY),
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Allow access when API_KEY is unset; otherwise require matching header."""
    if not API_KEY:
        return

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Request Models ---

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K


# --- API Endpoints ---

@app.post("/api/upload", dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Generate unique doc ID
    doc_id = str(uuid.uuid4())[:8]

    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if os.path.getsize(file_path) > MAX_UPLOAD_BYTES:
            os.remove(file_path)
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process document
    try:
        chunks = process_document(file_path)
        if not chunks:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No text content found in the document.")

        num_chunks = add_documents(chunks, doc_id)
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "chunks": num_chunks,
            "message": f"Successfully processed '{file.filename}' into {num_chunks} chunks."
        }
    except ValueError as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/search", dependencies=[Depends(verify_api_key)])
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    total_start = time.perf_counter()
    timings_ms = {}

    rewritten_query = request.query.strip()
    rewrite_start = time.perf_counter()
    if ENABLE_QUERY_REWRITE:
        try:
            rewritten_query = rewrite_query(request.query)
        except Exception as e:
            print(f"Query rewrite error: {e}")
            rewritten_query = request.query.strip()
    timings_ms["rewrite"] = round((time.perf_counter() - rewrite_start) * 1000, 1)

    retrieve_start = time.perf_counter()
    results = search(rewritten_query, request.top_k)
    timings_ms["retrieve"] = round((time.perf_counter() - retrieve_start) * 1000, 1)

    needs_clarification = False
    clarification_question = ""

    if ENABLE_CLARIFICATION_GATE:
        top_similarity = results[0]["similarity"] if results else 0
        second_similarity = results[1]["similarity"] if len(results) > 1 else 0
        margin = top_similarity - second_similarity

        low_confidence = top_similarity < CLARIFICATION_MIN_TOP_SIMILARITY
        ambiguous = (
            len(results) > 1
            and margin < CLARIFICATION_MARGIN_MIN
            and top_similarity < CLARIFICATION_HIGH_CONFIDENCE
        )

        if not results or low_confidence or ambiguous:
            needs_clarification = True
            clarification_question = build_clarification_question(request.query, results)

            return {
                "query": request.query,
                "rewritten_query": rewritten_query,
                "needs_clarification": True,
                "clarification_question": clarification_question,
                "ai_answer": clarification_question,
                "ai_sources": [],
                "results": results,
                "total": len(results),
                "timings_ms": {
                    **timings_ms,
                    "generate": 0.0,
                    "self_check": 0.0,
                    "total": round((time.perf_counter() - total_start) * 1000, 1)
                }
            }

    # Generate answer using fastest mode or LLM mode
    generate_start = time.perf_counter()
    generation_mode = "extractive" if FASTEST_RESPONSE_MODE else "llm"
    if FASTEST_RESPONSE_MODE:
        ai_result = generate_extractive_answer(results)
    else:
        ai_result = generate_answer(request.query, results)
    timings_ms["generate"] = round((time.perf_counter() - generate_start) * 1000, 1)

    quality_score = None
    self_check_start = time.perf_counter()
    if ENABLE_SELF_CHECK and not FASTEST_RESPONSE_MODE:
        try:
            quality_score = groundedness_score(ai_result["answer"], results)
            if quality_score < SELF_CHECK_MIN_GROUNDEDNESS:
                ai_result = generate_answer(request.query, results, strict_mode=True)
                quality_score = groundedness_score(ai_result["answer"], results)
        except Exception as e:
            print(f"Self-check error: {e}")
    timings_ms["self_check"] = round((time.perf_counter() - self_check_start) * 1000, 1)
    timings_ms["total"] = round((time.perf_counter() - total_start) * 1000, 1)

    return {
        "query": request.query,
        "rewritten_query": rewritten_query,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "generation_mode": generation_mode,
        "quality_score": round(quality_score, 3) if quality_score is not None else None,
        "ai_answer": ai_result["answer"],
        "ai_sources": ai_result["sources"],
        "results": results,
        "total": len(results),
        "timings_ms": timings_ms
    }


@app.get("/api/documents", dependencies=[Depends(verify_api_key)])
async def list_documents():
    """List all uploaded documents."""
    docs = get_all_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def remove_document(doc_id: str):
    """Delete a document and its chunks."""
    # Delete from vector store
    deleted_chunks = delete_document(doc_id)

    # Delete uploaded file
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(doc_id):
            file_path = os.path.join(UPLOAD_DIR, f)
            if os.path.exists(file_path):
                os.remove(file_path)
            break

    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "success": True,
        "deleted_chunks": deleted_chunks,
        "message": f"Deleted document {doc_id} ({deleted_chunks} chunks)."
    }


@app.get("/api/stats", dependencies=[Depends(verify_api_key)])
async def system_stats():
    """Get system statistics."""
    return get_stats()


@app.get("/api/health")
async def health_check():
    """Basic health check for deployment and monitoring."""
    return {
        "status": "ok",
        "api_auth_enabled": bool(API_KEY),
        "llm_loaded": is_model_loaded(),
        "upload_limit_mb": MAX_UPLOAD_BYTES // (1024 * 1024)
    }


# --- Serve Frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

@app.get("/")
async def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    config = {
        "apiKey": API_KEY,
        "uploadLimitMb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "apiAuthEnabled": bool(API_KEY)
    }
    config_script = f"<script>window.__DOCSEARCH_CONFIG__ = {json.dumps(config, ensure_ascii=False)};</script>"
    html = html.replace("</head>", f"    {config_script}\n</head>", 1)
    return HTMLResponse(content=html)

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
