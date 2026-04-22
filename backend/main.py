"""
Main FastAPI Application - Document Chatbot API Server.
"""
import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from backend.config import UPLOAD_DIR, SUPPORTED_EXTENSIONS, TOP_K
from backend.document_processor import process_document
from backend.vector_store import add_documents, search, delete_document, get_all_documents, get_stats
from backend.generator import generate_answer

app = FastAPI(title="Company Document Chatbot", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request Models ---

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K


# --- API Endpoints ---

@app.post("/api/upload")
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


@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = search(request.query, request.top_k)

    # Generate answer using LLM (returns dict with 'answer' and 'sources')
    ai_result = generate_answer(request.query, results)

    return {
        "query": request.query,
        "ai_answer": ai_result["answer"],
        "ai_sources": ai_result["sources"],
        "results": results,
        "total": len(results)
    }


@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents."""
    docs = get_all_documents()
    return {"documents": docs, "total": len(docs)}


@app.delete("/api/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a document and its chunks."""
    # Delete from vector store
    deleted_chunks = delete_document(doc_id)

    # Delete uploaded file
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(doc_id):
            os.remove(os.path.join(UPLOAD_DIR, f))
            break

    if deleted_chunks == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "success": True,
        "deleted_chunks": deleted_chunks,
        "message": f"Deleted document {doc_id} ({deleted_chunks} chunks)."
    }


@app.get("/api/stats")
async def system_stats():
    """Get system statistics."""
    return get_stats()


# --- Serve Frontend ---

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
