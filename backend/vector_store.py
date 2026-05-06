"""
Vector Store - ChromaDB wrapper for document storage and retrieval.
"""
import chromadb
import re
import os
import pickle
from typing import List, Dict, Optional
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from loguru import logger

from backend.config import (
    CHROMA_DIR,
    TOP_K,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
)
from backend.embedding_service import embed_passages, embed_query

# Collection name
COLLECTION_NAME = "company_documents"

# Global client
_client = None
_collection = None

_STOPWORDS = {
    "va", "và", "la", "là", "cua", "của", "cho", "tren", "trên", "duoc", "được",
    "the", "thi", "thì", "khi", "co", "có", "khong", "không", "mot", "một", "nhung", "những",
    "to", "from", "in", "on", "at", "is", "are", "be", "a", "an", "the", "for", "of", "and", "or"
}


def _tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into a list of words for BM25."""
    tokens = re.findall(r"[\w\-]+", text.lower())
    return [token for token in tokens if len(token) > 1 and token not in _STOPWORDS]


# Global state for BM25
_bm25_index = None
_bm25_docs = [] # Stores (text, metadata) pairs corresponding to index


def _get_bm25_path():
    return os.path.join(CHROMA_DIR, "bm25_index.pkl")


def _initialize_bm25(force_rebuild: bool = False):
    """Load BM25 from disk or build from ChromaDB."""
    global _bm25_index, _bm25_docs
    
    path = _get_bm25_path()
    if not force_rebuild and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                _bm25_index = data["index"]
                _bm25_docs = data["docs"]
                logger.info("Loaded BM25 index with {} chunks.", len(_bm25_docs))
                return
        except Exception as e:
            logger.warning("Failed to load BM25 index: {}. Rebuilding...", e)

    # Rebuild from ChromaDB
    collection = get_collection()
    if collection.count() == 0:
        _bm25_index = None
        _bm25_docs = []
        return

    logger.info("Building BM25 index from ChromaDB ({} chunks)...", collection.count())
    all_data = collection.get(include=["documents", "metadatas"])
    
    docs = []
    tokenized_corpus = []
    for i in range(len(all_data["ids"])):
        text = all_data["documents"][i]
        meta = all_data["metadatas"][i]
        docs.append({"text": text, "metadata": meta, "id": all_data["ids"][i]})
        tokenized_corpus.append(_tokenize_for_bm25(text))
    
    if tokenized_corpus:
        _bm25_index = BM25Okapi(tokenized_corpus)
        _bm25_docs = docs
        # Save to disk
        os.makedirs(CHROMA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"index": _bm25_index, "docs": _bm25_docs}, f)
        logger.info("BM25 index built and saved.")
    else:
        _bm25_index = None
        _bm25_docs = []


def get_collection():
    """Get or create the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize BM25 after collection is ready
        _initialize_bm25()
    return _collection


def add_documents(chunks: List[Dict], doc_id: str) -> int:
    """
    Add document chunks to the vector store.
    
    Args:
        chunks: List of {"text": "...", "metadata": {...}}
        doc_id: Unique document identifier
    
    Returns:
        Number of chunks added.
    """
    collection = get_collection()

    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        meta = {**chunk["metadata"], "doc_id": doc_id}
        # ChromaDB only supports str, int, float, bool metadata values
        meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in meta.items()}
        metadatas.append(meta)
        ids.append(f"{doc_id}_chunk_{i}")

    # Embed passages
    embeddings = embed_passages(texts)

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    # Rebuild BM25 to include new docs
    _initialize_bm25(force_rebuild=True)

    return len(chunks)


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Search for relevant document chunks using Hybrid Search (Vector + BM25).
    Combined via Reciprocal Rank Fusion (RRF).
    """
    collection = get_collection()
    effective_top_k = TOP_K if top_k is None else max(1, int(top_k))

    if collection.count() == 0:
        return []

    # 1. Vector Search (Dense)
    query_embedding = embed_query(query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(collection.count(), 50), # Get candidates for fusion
        include=["documents", "metadatas", "distances"]
    )

    # 2. BM25 Search (Sparse)
    bm25_results_list = []
    if _bm25_index:
        tokenized_query = _tokenize_for_bm25(query)
        # Get scores for all docs
        bm25_scores = _bm25_index.get_scores(tokenized_query)
        # Pair with docs and sort
        scored_docs = []
        for i, score in enumerate(bm25_scores):
            if score > 0:
                scored_docs.append((score, _bm25_docs[i]))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        bm25_results_list = scored_docs[:50]

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF combines rankings from different sources.
    # Score = sum(1 / (rank + k)) where k=60 is standard.
    k = 60
    scores = {} # Key: doc_id, Value: {score, text, metadata, vector_sim}

    # Process Vector Results
    if vector_results and vector_results["ids"]:
        for rank, doc_id in enumerate(vector_results["ids"][0]):
            distance = vector_results["distances"][0][rank]
            similarity = max(0, (1 - distance / 2)) * 100
            
            if doc_id not in scores:
                scores[doc_id] = {
                    "score": 0,
                    "text": vector_results["documents"][0][rank],
                    "metadata": vector_results["metadatas"][0][rank],
                    "similarity": similarity
                }
            scores[doc_id]["score"] += 1.0 / (rank + 1 + k)

    # Process BM25 Results
    for rank, (bm25_score, doc_info) in enumerate(bm25_results_list):
        doc_id = doc_info["id"]
        if doc_id not in scores:
            scores[doc_id] = {
                "score": 0,
                "text": doc_info["text"],
                "metadata": doc_info["metadata"],
                "similarity": 0.0 # Will be updated if also in vector results
            }
        scores[doc_id]["score"] += 1.0 / (rank + 1 + k)

    # 4. Final Ranking
    final_results = list(scores.values())
    # If a result was only in BM25, similarity is 0. 
    # We don't filter by SIMILARITY_THRESHOLD here to allow keyword matches to surface.
    final_results.sort(key=lambda x: x["score"], reverse=True)

    # Prepare output
    formatted = []
    for item in final_results[:effective_top_k]:
        formatted.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "similarity": round(item["similarity"], 1)
        })
    
    return formatted


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document."""
    collection = get_collection()

    # Find all chunks with this doc_id
    results = collection.get(
        where={"doc_id": doc_id},
        include=[]
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        # Rebuild BM25 after deletion
        _initialize_bm25(force_rebuild=True)
        return len(results["ids"])
    return 0


def get_all_documents() -> List[Dict]:
    """Get list of all unique documents in the store."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    all_data = collection.get(include=["metadatas"])
    
    # Group by doc_id
    docs = {}
    for meta in all_data["metadatas"]:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "source": meta.get("source", "unknown"),
                "chunk_count": 0
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


def get_stats() -> Dict:
    """Get stats about the vector store."""
    collection = get_collection()
    docs = get_all_documents()
    return {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "documents": docs
    }
