"""
Vector Store - ChromaDB wrapper for document storage and retrieval.
"""
import chromadb
import hashlib
import re
import os
import pickle
from typing import List, Dict, Optional, Tuple
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from loguru import logger

from backend.config import (
    CHROMA_DIR,
    TOP_K,
    SIMILARITY_THRESHOLD,
    ENABLE_CONTEXT_EXPANSION,
    EMBEDDING_MODEL,
    VECTOR_COLLECTION,
)
from backend.embedding_service import embed_passages, embed_query

# Collection name is versioned per embedding model: changing embedding dims
# (e5-small 384 -> Vietnamese_Embedding_v2 1024) creates a NEW collection
# instead of corrupting the old index. Old data stays on disk for rollback.
# Pin via VECTOR_COLLECTION env to force the legacy name ("company_documents").
BASE_COLLECTION_NAME = "company_documents"


def _slugify_model(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[-48:] if len(slug) > 48 else slug


def get_active_collection_name() -> str:
    if VECTOR_COLLECTION:
        return VECTOR_COLLECTION
    return f"{BASE_COLLECTION_NAME}-{_slugify_model(EMBEDDING_MODEL)}"


# Legacy name (e5-small era) — kept for rollback via VECTOR_COLLECTION=company_documents
COLLECTION_NAME = get_active_collection_name()

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
    # BM25 index is per-collection so the embedding switch never mixes old/new docs.
    safe = re.sub(r"[^a-z0-9_-]+", "_", get_active_collection_name().lower())
    return os.path.join(CHROMA_DIR, f"bm25_index_{safe}.pkl")


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
    """Get or create the ChromaDB collection (name versioned per embedding model)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        _collection = _client.get_or_create_collection(
            name=get_active_collection_name(),
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize BM25 after collection is ready
        _initialize_bm25()
    return _collection


def add_documents(chunks: List[Dict], doc_id: str, rebuild_bm25: bool = True) -> int:
    """
    Add document chunks to the vector store.
    
    Args:
        chunks: List of {"text": "...", "metadata": {...}}
        doc_id: Unique document identifier
        rebuild_bm25: Rebuild BM25 right away. Batch re-ingest nên đặt False
            và rebuild một lần sau khi xong (tránh O(N) rebuild mỗi văn bản).
    
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
    if rebuild_bm25:
        _initialize_bm25(force_rebuild=True)

    # GraphRAG-lite: index entities/edges (opt-in, never breaks ingest on failure)
    try:
        from backend.config import ENABLE_GRAPH_RAG, GRAPH_USE_LLM
        if ENABLE_GRAPH_RAG:
            from backend.graph_store import build_from_chunks
            source_name = metadatas[0].get("source", "") if metadatas else ""
            build_from_chunks(doc_id, chunks, use_llm=GRAPH_USE_LLM, source_name=str(source_name))
    except Exception as e:
        logger.warning("Graph indexing skipped for doc {}: {}", doc_id, e)

    return len(chunks)


def _dedupe_by_text(items: List[Dict]) -> List[Dict]:
    """Drop các chunk có nội dung (đã normalize) trùng nhau, giữ bản hạng cao nhất.

    Corpus luật/doanh nghiệp thật thường chứa đoạn điều khoản mẫu lặp y hệt
    trong hàng chục văn bản; không dedupe thì chúng chiếm hết slot top-K.
    """
    deduped: List[Dict] = []
    seen = set()
    for item in items:
        norm = " ".join((item.get("text") or "").split()).lower()
        key = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    dropped = len(items) - len(deduped)
    if dropped:
        logger.info("Boilerplate dedupe dropped {} duplicate chunks.", dropped)
    return deduped


_QUERY_DOC_NUMBER_RE = re.compile(r"\d{1,4}/\d{4}(?:/[A-Za-zĐđ\-]+)?")
_QUERY_ARTICLE_RE = re.compile(r"[Đđ]i[eề]u\s+(\d+)")


def _extract_query_entities(query: str) -> List[str]:
    """Trích số hiệu văn bản (293/2025/NĐ-CP) và số điều (Điều 107) từ query."""
    entities: List[str] = []
    for m in _QUERY_DOC_NUMBER_RE.finditer(query):
        entities.append(m.group(0))
    for m in _QUERY_ARTICLE_RE.finditer(query):
        entities.append(f"Điều {m.group(1)}.")
    return entities


def _lookup_entity_sources(entity: str) -> set:
    """Tìm các source (file) có chunk chứa nguyên văn số hiệu văn bản.

    Số hiệu thường chỉ xuất hiện ở phần đầu phụ lục/tiêu đề chứ không lặp lại
    trong từng điều — lift cả văn bản để chunk trả lời (vd. Điều 5 "hết hiệu
    lực") được đưa lên mà không cần dense hiểu token số hiệu hiếm.
    """
    try:
        collection = get_collection()
        res = collection.get(
            where_document={"$contains": entity},
            include=["metadatas"],
            limit=500,
        )
        return {(m or {}).get("source") for m in res.get("metadatas", []) if m}
    except Exception as e:
        logger.warning("Entity source lookup failed for {}: {}", entity, e)
        return set()


def _apply_legal_entity_boost(query: str, items: List[Dict]) -> List[Dict]:
    """Cộng điểm cho chunk chứa đúng entity mà query nhắc tên đích danh.

    Vì sao cần: embedding kém với token số hiệu hiếm (293/2025/NĐ-CP -> dense
    rank ~92/100) trong khi fusion weighted cap nhánh BM25 ở (1-w)*100 điểm —
    chunk đúng dù BM25 rank 1 vẫn có thể thua hàng loạt chunk dense tầm thường.
    "Điều N." chỉ khớp dạng header (có dấu chấm) để tránh cross-reference
    "tại Điều 35 của Bộ luật này".
    """
    from backend.config import (
        LEGAL_ENTITY_BOOST, LEGAL_ENTITY_BOOST_DOC, LEGAL_ENTITY_BOOST_ARTICLE,
        LEGAL_ENTITY_BOOST_CAP, LEGAL_ENTITY_BOOST_SOURCE,
    )
    if not LEGAL_ENTITY_BOOST:
        return items
    entities = _extract_query_entities(query)
    if not entities:
        return items

    # Resolve các số hiệu văn bản -> tập source chứa chúng (một lần/văn bản).
    doc_entities = [e for e in entities if not e.startswith("Điều")]
    entity_sources: Dict[str, set] = {}
    for ent in doc_entities:
        entity_sources[ent] = _lookup_entity_sources(ent)

    boosted_any = False
    for item in items:
        text = item.get("match_text") or item.get("text") or ""
        text_lower = text.lower()
        source = (item.get("metadata") or {}).get("source", "")
        bonus = 0.0
        for ent in entities:
            if ent.startswith("Điều"):
                if ent in text:
                    bonus += LEGAL_ENTITY_BOOST_ARTICLE
            elif ent.lower() in text_lower:
                bonus += LEGAL_ENTITY_BOOST_DOC
            elif source and source in entity_sources.get(ent, ()):
                # Chunk cùng văn bản nhưng không chứa số hiệu (vd. Điều 5 ND293)
                bonus += LEGAL_ENTITY_BOOST_SOURCE
        if bonus > 0:
            bonus = min(bonus, LEGAL_ENTITY_BOOST_CAP)
            item["similarity"] = round(float(item.get("similarity", 0)) + bonus, 1)
            item["score"] = float(item.get("score", 0)) + bonus
            boosted_any = True

    if boosted_any:
        items = sorted(items, key=lambda r: float(r.get("score", 0)), reverse=True)
        logger.info("Legal entity boost applied for entities: {}", entities)
    return items


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
    # Pool 100 (thay vì 50): HNSW recall ở corpus 69k chunk thấp hơn với k nhỏ —
    # các chunk đúng có thể nằm ngoài top-50 khi hàng chục bản boilerplate
    # điểm cao chen chúc. Dedupe phía sau sẽ dọn các bản trùng.
    # Nếu embedding lỗi (đã quan sát NaN thoáng qua do môi trường), hạ cấp sang
    # BM25-only thay vì làm sập toàn bộ truy vấn.
    try:
        query_embedding = embed_query(query)
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(collection.count(), 100), # Get candidates for fusion
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error("Dense retrieval failed ({}); falling back to BM25-only.", e)
        vector_results = None

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
        bm25_results_list = scored_docs[:100]

    # 3. Fusion (dense + sparse) — weighted interpolation (mặc định SOTA)
    # hoặc RRF cổ điển qua env HYBRID_FUSION=rrf. Xem backend/hybrid_fusion.py.
    from backend.hybrid_fusion import fuse as _fuse_candidates
    from backend.config import HYBRID_FUSION, HYBRID_DENSE_WEIGHT, HYBRID_RRF_K

    vector_ranked: List[Tuple[str, float]] = []
    payloads: Dict[str, Dict] = {}
    if vector_results and vector_results["ids"]:
        for rank, doc_id in enumerate(vector_results["ids"][0]):
            distance = vector_results["distances"][0][rank]
            similarity = max(0, (1 - distance / 2)) * 100
            vector_ranked.append((doc_id, similarity))
            payloads[doc_id] = {
                "text": vector_results["documents"][0][rank],
                "metadata": vector_results["metadatas"][0][rank],
                "similarity": similarity,
            }

    bm25_ranked: List[Tuple[str, float]] = []
    for _bm25_score, doc_info in bm25_results_list:
        doc_id = doc_info["id"]
        bm25_ranked.append((doc_id, float(_bm25_score)))
        if doc_id not in payloads:
            payloads[doc_id] = {
                "text": doc_info["text"],
                "metadata": doc_info["metadata"],
                "similarity": 0.0,  # chỉ khớp keyword, không có điểm dense
            }

    fused_order = _fuse_candidates(
        vector_ranked, bm25_ranked,
        method=HYBRID_FUSION, dense_weight=HYBRID_DENSE_WEIGHT, rrf_k=HYBRID_RRF_K,
    )

    # 4. Final Ranking
    # Với weighted: fused_score đã ở thang 0-100 và thay thế similarity để
    # ngưỡng lọc downstream (CRAG/clarification) phản ánh cả 2 nguồn.
    # Với rrf: giữ similarity dense gốc như hành vi cũ.
    final_results = []
    for doc_id, fused_score, dense_sim in fused_order:
        item = payloads[doc_id]
        if (HYBRID_FUSION or "weighted").strip().lower() == "rrf":
            final_results.append({**item, "score": fused_score})
        else:
            final_results.append({**item, "score": fused_score, "similarity": round(fused_score, 1)})
    # Doc chỉ khớp BM25 vẫn được giữ (không lọc ngưỡng ở đây để keyword lọt lên).

    # 4b. Boilerplate dedupe: nhiều văn bản luật chứa các đoạn điều khoản mẫu
    # giống hệt nhau (vd. điều về dân số xuất hiện trong ~100 luật). Nếu không
    # lọc, các bản sao chiếm hết slot ngữ cảnh và đẩy chunk trả lời đúng xuống.
    final_results = _dedupe_by_text(final_results)

    # 4c. Legal entity boost: query nhắc đích danh số hiệu/số điều -> ưu tiên
    # chunk chứa đúng chuỗi đó (xem _apply_legal_entity_boost).
    final_results = _apply_legal_entity_boost(query, final_results)

    # 5. Context Expansion (Sliding Window)
    top_items = final_results[:effective_top_k]
    
    if ENABLE_CONTEXT_EXPANSION and top_items:
        # Collect IDs of adjacent chunks
        adjacent_ids = set()
        for item in top_items:
            doc_id = item["metadata"].get("doc_id")
            chunk_idx = item["metadata"].get("chunk_index")
            if doc_id is not None and chunk_idx is not None:
                try:
                    idx = int(chunk_idx)
                    if idx > 0:
                        adjacent_ids.add(f"{doc_id}_chunk_{idx - 1}")
                    adjacent_ids.add(f"{doc_id}_chunk_{idx + 1}")
                except ValueError:
                    pass
        
        # Fetch adjacent chunks in one batch
        if adjacent_ids:
            adjacent_data = collection.get(ids=list(adjacent_ids), include=["documents"])
            adj_dict = {}
            if adjacent_data and adjacent_data["ids"]:
                for i, aid in enumerate(adjacent_data["ids"]):
                    adj_dict[aid] = adjacent_data["documents"][i]
            
            # Merge text
            for item in top_items:
                doc_id = item["metadata"].get("doc_id")
                chunk_idx = item["metadata"].get("chunk_index")
                if doc_id is not None and chunk_idx is not None:
                    try:
                        idx = int(chunk_idx)
                        prev_text = adj_dict.get(f"{doc_id}_chunk_{idx - 1}", "")
                        next_text = adj_dict.get(f"{doc_id}_chunk_{idx + 1}", "")
                        
                        merged_parts = []
                        if prev_text:
                            merged_parts.append(prev_text)
                        merged_parts.append(item["text"])
                        if next_text:
                            merged_parts.append(next_text)
                            
                        item["match_text"] = item["text"]
                        item["text"] = "\n...\n".join(merged_parts)
                    except ValueError:
                        item["match_text"] = item["text"]
                        pass
        
        # fallback cho các item không expand được
        for item in top_items:
            if "match_text" not in item:
                item["match_text"] = item["text"]
    else:
        for item in top_items:
            item["match_text"] = item["text"]

    # Prepare output
    formatted = []
    for item in top_items:
        formatted.append({
            "text": item["text"],
            "match_text": item["match_text"],
            "metadata": item["metadata"],
            "similarity": round(item["similarity"], 1)
        })
    
    return formatted


def delete_document(doc_id: str, rebuild_bm25: bool = True) -> int:
    """Delete all chunks belonging to a document.

    Args:
        rebuild_bm25: Rebuild BM25 right away (False cho batch re-ingest).
    """
    collection = get_collection()

    # GraphRAG-lite: clean graph first (best-effort, never breaks deletion)
    try:
        from backend.graph_store import delete_doc_graph
        delete_doc_graph(doc_id)
    except Exception as e:
        logger.warning("Graph cleanup skipped for doc {}: {}", doc_id, e)

    # Find all chunks with this doc_id
    results = collection.get(
        where={"doc_id": doc_id},
        include=[]
    )

    if results["ids"]:
        collection.delete(ids=results["ids"])
        # Rebuild BM25 after deletion
        if rebuild_bm25:
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
    stats = {
        "total_chunks": collection.count(),
        "total_documents": len(docs),
        "documents": docs,
        "embedding_model": EMBEDDING_MODEL,
        "collection": get_active_collection_name(),
    }
    try:
        from backend.config import ENABLE_GRAPH_RAG
        if ENABLE_GRAPH_RAG:
            from backend.graph_store import graph_stats
            stats["graph"] = graph_stats()
    except Exception:
        pass
    return stats
