"""
Embedding Service - Handles text embedding using intfloat/multilingual-e5-small.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL

# Global model instance (loaded once)
_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("Model loaded successfully!")
    return _model


def embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Embed document passages.
    E5 models require 'passage: ' prefix for documents.
    """
    model = get_model()
    prefixed_texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()


import time

def embed_query(query: str) -> List[float]:
    """
    Embed a search query.
    E5 models require 'query: ' prefix for queries.
    """
    model = get_model()
    prefixed_query = f"query: {query}"
    
    start_time = time.time()
    embedding = model.encode([prefixed_query], normalize_embeddings=True)
    end_time = time.time()
    
    print(f"⚡ Embedding Speed (Query): {(end_time - start_time) * 1000:.2f} ms")
    return embedding[0].tolist()
