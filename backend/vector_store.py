"""
Vector Store - ChromaDB wrapper for document storage and retrieval.
"""
import chromadb
import re
from typing import List, Dict
from backend.config import (
    CHROMA_DIR,
    TOP_K,
    SIMILARITY_THRESHOLD,
    RETRIEVAL_CANDIDATE_MULTIPLIER,
    RERANK_EMBEDDING_WEIGHT,
    RERANK_KEYWORD_WEIGHT,
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


def _tokenize(text: str) -> set:
    tokens = re.findall(r"[\w\-]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}


def _keyword_overlap_score(query_tokens: set, text: str) -> float:
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens.intersection(text_tokens)) / len(query_tokens)


def get_collection():
    """Get or create the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
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

    return len(chunks)


def search(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Search for relevant document chunks.
    
    Returns:
        List of results with text, metadata, and similarity score.
    """
    collection = get_collection()
    effective_top_k = TOP_K if top_k is None else max(1, int(top_k))

    if collection.count() == 0:
        return []

    candidate_count = min(
        max(effective_top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, effective_top_k),
        collection.count()
    )

    query_embedding = embed_query(query)
    query_tokens = _tokenize(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"]
    )

    formatted_results = []
    if results and results["documents"]:
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score (0-100%)
            similarity = max(0, (1 - distance / 2)) * 100

            # Filter out low-similarity results
            if similarity < SIMILARITY_THRESHOLD:
                continue

            keyword_score = _keyword_overlap_score(query_tokens, results["documents"][0][i])
            rerank_score = (
                RERANK_EMBEDDING_WEIGHT * (similarity / 100.0) +
                RERANK_KEYWORD_WEIGHT * keyword_score
            ) * 100

            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": round(similarity, 1),
                "rerank_score": round(rerank_score, 1)
            })

    formatted_results.sort(key=lambda item: item["rerank_score"], reverse=True)
    for item in formatted_results:
        item.pop("rerank_score", None)
    return formatted_results[:effective_top_k]


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
