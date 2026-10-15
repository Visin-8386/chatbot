"""
Embedding Service - Vietnamese SOTA default (AITeamVN/Vietnamese_Embedding_v2,
bge-m3-based, 1024 dims) with e5-small fallback. Free + offline after first download.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL, EMBEDDING_USE_E5_PREFIX, MODELS_DIR

# Global model instance (loaded once)
_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model."""
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODELS_DIR)
        print("Model loaded successfully!")
    return _model


def _maybe_prefix(texts: List[str], kind: str) -> List[str]:
    """Apply E5 query/passage prefixes only when the active model needs them."""
    if not EMBEDDING_USE_E5_PREFIX:
        return texts
    prefix = "query: " if kind == "query" else "passage: "
    return [f"{prefix}{t}" for t in texts]


def embed_passages(texts: List[str]) -> List[List[float]]:
    """
    Embed document passages.
    E5 models require 'passage: ' prefix; bge-m3-based Vietnamese models use raw text.
    """
    model = get_model()
    prefixed_texts = _maybe_prefix(texts, "passage")
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings.tolist()


import time

def embed_query(query: str) -> List[float]:
    """
    Embed a search query.
    E5 models require 'query: ' prefix; bge-m3-based Vietnamese models use raw text.
    """
    model = get_model()
    prefixed_query = _maybe_prefix([query], "query")[0]

    start_time = time.time()
    embedding = model.encode([prefixed_query], normalize_embeddings=True)
    end_time = time.time()

    return embedding[0].tolist()

def get_embedding_dim() -> int:
    """Return the active embedding dimension (used to validate the vector collection)."""
    model = get_model()
    return int(model.get_sentence_embedding_dimension())

def get_cosine_similarities(embeddings: List[List[float]]) -> List[float]:
    """Calculate cosine similarities between adjacent embeddings."""
    import numpy as np
    if len(embeddings) < 2:
        return []
    
    embs = np.array(embeddings)
    # Since embeddings are normalized, cosine similarity is just dot product
    sims = np.sum(embs[:-1] * embs[1:], axis=1)
    return sims.tolist()
