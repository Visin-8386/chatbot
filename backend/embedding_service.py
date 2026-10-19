"""
Embedding Service - Vietnamese SOTA default (AITeamVN/Vietnamese_Embedding_v2,
bge-m3-based, 1024 dims) with e5-small fallback. Free + offline after first download.
"""
from typing import List
from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL, EMBEDDING_USE_E5_PREFIX, EMBEDDING_DEVICE, EMBEDDING_MAX_SEQ_LENGTH, MODELS_DIR

# Global model instance (loaded once)
_model = None
_model_f64 = None
_fp32_nan_until = 0.0


def _get_model_f64() -> SentenceTransformer:
    """Bản fp64 của embedding model (load riêng một lần, dùng khi fp32 NaN).

    KHÔNG dùng copy.deepcopy: đã kiểm nghiệm deepcopy làm input path hỏng —
    model trả về cùng một vector cho mọi câu (cos=1.0 giữa hai câu bất kỳ).
    """
    global _model_f64
    if _model_f64 is None:
        import numpy as np
        device = _resolve_device()
        print("Loading float64 embedding model for numerical fallback...")
        _model_f64 = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODELS_DIR, device=device).double()
        _model_f64.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
        # Self-check semantics: hai câu khác nhau phải cho vector khác hướng.
        probe = _model_f64.encode(["nghỉ hằng năm", "thời tiết hôm nay"], normalize_embeddings=True)
        c = float(np.dot(np.asarray(probe[0]), np.asarray(probe[1])))
        if c > 0.99:
            raise RuntimeError(f"fp64 model self-check failed (cos={c:.3f}); input path appears broken.")
    return _model_f64


def _resolve_device() -> str:
    if EMBEDDING_DEVICE == "cuda":
        return "cuda"
    if EMBEDDING_DEVICE == "cpu":
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model() -> SentenceTransformer:
    """Load and cache the embedding model (device per EMBEDDING_DEVICE)."""
    global _model
    if _model is None:
        device = _resolve_device()
        print(f"Loading embedding model: {EMBEDDING_MODEL} (device={device})...")
        _model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODELS_DIR, device=device)
        # Model card khuyến nghị 2048: query huấn luyện ở 256 token, passage 2048.
        # Không cap sẽ nổ activation với chunk khổng lồ (từng OOM 7.4GB khi embed).
        _model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH
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
    embeddings = _encode_checked(model, prefixed_texts, show_progress_bar=True)
    return embeddings.tolist()


import time

def _encode_checked(model, texts, normalize_embeddings=True, retries=2, show_progress_bar=False):
    """Encode với guard NaN và fallback fp64.

    Đã quan sát NaN thoáng qua trên CPU fp32 (torch/oneDNN): LayerNorm của
    XLM-R tính variance gần 0 bị catastrophic cancellation, dấu phụ thuộc thứ
    tự reduction của OMP threads (đổi theo tải hệ thống) -> sqrt(số âm) = NaN.
    fp64 không bị (không cancellation) nên dùng làm fallback; kết quả cast về
    float32 để tương thích index hiện có. Trong "cửa sổ lỗi", đi thẳng fp64
    để khỏi tốn 3 lần retry mỗi query.
    """
    global _fp32_nan_until
    import numpy as np

    if time.time() < _fp32_nan_until:
        model_f64 = _get_model_f64()
        embeddings = model_f64.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
        arr = np.asarray(embeddings, dtype="float32")
        if not np.isnan(arr).any():
            return arr
        _fp32_nan_until = 0.0

    for attempt in range(retries + 1):
        embeddings = model.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
        arr = np.asarray(embeddings, dtype="float32")
        if not np.isnan(arr).any():
            _fp32_nan_until = 0.0
            return arr
        print(f"WARNING: embedding NaN detected (attempt {attempt + 1}/{retries + 1}), retrying...")
        time.sleep(0.5)

    _fp32_nan_until = time.time() + 300
    print("WARNING: fp32 encode still NaN; falling back to float64 encode (slower but numerically stable).")
    model_f64 = _get_model_f64()
    embeddings = model_f64.encode(texts, normalize_embeddings=normalize_embeddings, show_progress_bar=show_progress_bar)
    arr = np.asarray(embeddings, dtype="float32")
    if np.isnan(arr).any():
        raise RuntimeError("Embedding model produced NaN vectors in both fp32 and fp64; retrieval aborted to avoid garbage results.")
    return arr


def embed_query(query: str) -> List[float]:
    """
    Embed a search query.
    E5 models require 'query: ' prefix; bge-m3-based Vietnamese models use raw text.
    """
    model = get_model()
    prefixed_query = _maybe_prefix([query], "query")[0]

    embedding = _encode_checked(model, [prefixed_query])

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
