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
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=False)

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
    ENABLE_HYDE,
    ENABLE_PERSISTENT_MEMORY,
)
from backend.document_processor import process_document
from backend.vector_store import add_documents, search, delete_document, get_all_documents, get_stats
from backend.generator import (
    generate_answer,
    generate_answer_stream,
    is_model_loaded,
    rewrite_query,
    rerank_results,
    build_clarification_question,
    groundedness_score,
    generate_extractive_answer,
    select_relevant_history,
    preload_models,
    hyde_expand_query,
    crag_check_relevance,
)
from backend.chat_memory import init_db, save_turn, get_history, delete_session, get_all_sessions
from backend.intent_classifier import classify_intent, get_chitchat_response

app = FastAPI(title="Company Document Chatbot", version="1.0.0")


@app.on_event("startup")
async def startup_preload():
    """Preload models and initialize DB at server start."""
    logger.info("[STARTUP] Initializing chat memory DB...")
    init_db()
    logger.info("[STARTUP] Preloading models (LLM, Reranker, Embedding)...")
    await run_in_threadpool(preload_models)
    logger.info("[STARTUP] All systems ready.")


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
    session_id: Optional[str] = None
    history: Optional[list[dict[str, str]]] = None


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

    # ── 0. Intent Detection (rule-based, ~0ms) ──────────────────────────────
    has_history = bool(
        (ENABLE_PERSISTENT_MEMORY and request.session_id) or request.history
    )
    intent_result = classify_intent(request.query, has_history=has_history)
    intent = intent_result["intent"]
    timings_ms["intent"] = round((time.perf_counter() - total_start) * 1000, 1)
    logger.debug("[Intent] {} — {}", intent, intent_result["reason"])

    # ── Chitchat: bypass RAG hoàn toàn ──────────────────────────────────────
    if intent == "chitchat":
        chitchat_answer = get_chitchat_response(request.query)
        # Lưu vào memory để maintain context
        if ENABLE_PERSISTENT_MEMORY and request.session_id:
            await run_in_threadpool(save_turn, request.session_id, request.query, chitchat_answer)
        return {
            "query": request.query,
            "session_id": request.session_id,
            "intent": "chitchat",
            "rewritten_query": request.query,
            "needs_clarification": False,
            "clarification_question": "",
            "crag_verdict": "skipped",
            "generation_mode": "chitchat",
            "self_check_status": "skipped",
            "quality_score": None,
            "ai_answer": chitchat_answer,
            "ai_sources": [],
            "results": [],
            "total": 0,
            "timings_ms": {**timings_ms, "total": round((time.perf_counter() - total_start) * 1000, 1)}
        }

    # ── Follow-up: relax clarification gate, force history ──────────────────
    is_followup = (intent == "followup")

    rewritten_query = request.query.strip()

    # 1. Query Rewrite
    rewrite_start = time.perf_counter()
    if ENABLE_QUERY_REWRITE:
        try:
            rewritten_query = await run_in_threadpool(rewrite_query, request.query)
        except Exception as e:
            logger.error("Query rewrite error: {}", e)
            rewritten_query = request.query.strip()
    timings_ms["rewrite"] = round((time.perf_counter() - rewrite_start) * 1000, 1)

    # 1b. HyDE — expand query with hypothetical answer (2.1)
    hyde_start = time.perf_counter()
    retrieval_query = rewritten_query
    if ENABLE_HYDE:
        try:
            retrieval_query = await run_in_threadpool(hyde_expand_query, rewritten_query)
        except Exception as e:
            logger.warning("[HyDE] error: {}", e)
    timings_ms["hyde"] = round((time.perf_counter() - hyde_start) * 1000, 1)

    # 2. Retrieve & Rerank
    retrieve_start = time.perf_counter()
    results = await run_in_threadpool(search, retrieval_query, request.top_k * 2)
    timings_ms["retrieve"] = round((time.perf_counter() - retrieve_start) * 1000, 1)

    rerank_start = time.perf_counter()
    results = await run_in_threadpool(rerank_results, rewritten_query, results, request.top_k)
    timings_ms["rerank"] = round((time.perf_counter() - rerank_start) * 1000, 1)

    needs_clarification = False
    clarification_question = ""

    # Follow-up: bỏ qua clarification gate vì người dùng đang hỏi tiếp, cứ generate
    if ENABLE_CLARIFICATION_GATE and not is_followup:
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
                "session_id": request.session_id,
                "intent": intent,
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

    # CRAG — kiểm tra độ liên quan trc khi generate (2.2)
    crag_start = time.perf_counter()
    crag_result = await run_in_threadpool(crag_check_relevance, request.query, results)
    timings_ms["crag"] = round((time.perf_counter() - crag_start) * 1000, 1)
    crag_verdict = crag_result["verdict"]
    results = crag_result["filtered_results"]  # Empty nếu irrelevant

    if crag_verdict == "irrelevant":
        return {
            "query": request.query,
            "rewritten_query": rewritten_query,
            "needs_clarification": False,
            "crag_verdict": "irrelevant",
            "ai_answer": "Không tìm thấy thông tin liên quan trong tài liệu. " + crag_result["reason"],
            "ai_sources": [],
            "results": [],
            "total": 0,
            "timings_ms": {**timings_ms, "generate": 0.0, "self_check": 0.0,
                           "total": round((time.perf_counter() - total_start) * 1000, 1)}
        }

    # Lấy history: follow-up luôn include history, rag_query dùng smart selection
    effective_history = []
    if ENABLE_PERSISTENT_MEMORY and request.session_id:
        db_history = await run_in_threadpool(get_history, request.session_id)
        effective_history = db_history if is_followup else select_relevant_history(request.query, db_history)
    elif request.history:
        effective_history = request.history if is_followup else select_relevant_history(request.query, request.history)

    # Generate answer using fastest mode or LLM mode
    generate_start = time.perf_counter()
    generation_mode = "extractive" if FASTEST_RESPONSE_MODE else "llm"
    if FASTEST_RESPONSE_MODE:
        ai_result = await run_in_threadpool(generate_extractive_answer, results)
    else:
        ai_result = await run_in_threadpool(generate_answer, request.query, results, False, effective_history)
    timings_ms["generate"] = round((time.perf_counter() - generate_start) * 1000, 1)

    quality_score = None
    self_check_status = "disabled"
    self_check_start = time.perf_counter()
    if ENABLE_SELF_CHECK and not FASTEST_RESPONSE_MODE:
        self_check_status = "running"
        try:
            answer_text = ai_result["answer"].split("\n\n📌 **Nguồn trích dẫn:**", 1)[0].strip()
            quality_score = await run_in_threadpool(groundedness_score, answer_text, results)
            if quality_score < SELF_CHECK_MIN_GROUNDEDNESS:
                print(f"[SELF-CHECK] Low groundedness ({quality_score:.3f}), retrying in strict mode.")
                self_check_status = "strict_retry"

                strict_result = await run_in_threadpool(generate_answer, request.query, results, True, effective_history)
                strict_answer_text = strict_result["answer"].split("\n\n📌 **Nguồn trích dẫn:**", 1)[0].strip()
                strict_score = await run_in_threadpool(groundedness_score, strict_answer_text, results)

                if strict_score >= quality_score:
                    ai_result = strict_result
                    quality_score = strict_score
                    self_check_status = "strict_retry_passed"
                else:
                    fallback_result = await run_in_threadpool(generate_extractive_answer, results)
                    fallback_answer_text = fallback_result["answer"].split("\n\n📌 **Nguồn trích dẫn:**", 1)[0].strip()
                    fallback_score = await run_in_threadpool(groundedness_score, fallback_answer_text, results)
                    ai_result = fallback_result
                    quality_score = fallback_score
                    self_check_status = "extractive_fallback"
            else:
                self_check_status = "passed"
        except Exception as e:
            print(f"Self-check error: {e}")
            self_check_status = "error"
    timings_ms["self_check"] = round((time.perf_counter() - self_check_start) * 1000, 1)
    timings_ms["total"] = round((time.perf_counter() - total_start) * 1000, 1)

    # Lưu lượt hội thoại vào Persistent Memory (2.3)
    if ENABLE_PERSISTENT_MEMORY and request.session_id:
        plain_answer = ai_result["answer"].split("\n\n\U0001f4cc", 1)[0].strip()
        await run_in_threadpool(save_turn, request.session_id, request.query, plain_answer)

    return {
        "query": request.query,
        "session_id": request.session_id,
        "intent": intent,
        "rewritten_query": rewritten_query,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "crag_verdict": crag_verdict if 'crag_verdict' in dir() else "skipped",
        "generation_mode": generation_mode,
        "self_check_status": self_check_status,
        "quality_score": round(quality_score, 3) if quality_score is not None else None,
        "ai_answer": ai_result["answer"],
        "ai_sources": ai_result["sources"],
        "results": results,
        "total": len(results),
        "timings_ms": timings_ms
    }


@app.post("/api/search/stream", dependencies=[Depends(verify_api_key)])
async def search_documents_stream(request: SearchRequest):
    """Search and stream AI answer tokens."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Rewrite
    rewritten_query = request.query.strip()
    if ENABLE_QUERY_REWRITE:
        try:
            rewritten_query = await run_in_threadpool(rewrite_query, request.query)
        except Exception as e:
            logger.error("Query rewrite error: {}", e)

    # 2. Retrieve & Rerank
    results = await run_in_threadpool(search, rewritten_query, request.top_k * 2)
    results = await run_in_threadpool(rerank_results, rewritten_query, results, request.top_k)

    # 3. History
    effective_history = select_relevant_history(request.query, request.history)

    # 4. Stream Response
    def stream_generator():
        for chunk in generate_answer_stream(request.query, results, False, effective_history):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


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


# --- Session / Memory Endpoints (2.3) ---

@app.get("/api/sessions", dependencies=[Depends(verify_api_key)])
async def list_sessions():
    """Liệt kê tất cả session đã có trong bộ nhớ."""
    sessions = await run_in_threadpool(get_all_sessions)
    return {"sessions": sessions, "total": len(sessions)}


@app.get("/api/sessions/{session_id}/history", dependencies=[Depends(verify_api_key)])
async def get_session_history(session_id: str, max_turns: Optional[int] = None):
    """Lấy lịch sử hội thoại của một session."""
    history = await run_in_threadpool(get_history, session_id, max_turns)
    return {"session_id": session_id, "history": history, "turns": len(history)}


@app.delete("/api/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def clear_session(session_id: str):
    """Xóa toàn bộ lịch sử của một session."""
    deleted = await run_in_threadpool(delete_session, session_id)
    return {"success": True, "deleted_rows": deleted, "session_id": session_id}


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
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>Frontend index.html not found</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

# Mount static files
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
