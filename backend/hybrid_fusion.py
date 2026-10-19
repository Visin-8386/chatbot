"""
Hybrid Fusion - Pure score-fusion strategies for dense + sparse retrieval.

Không dependency nặng, không chạm DB: dễ unit-test, dễ thay chiến lược qua env.
- "weighted" (mặc định, SOTA theo paper EACL-2026 cho tiếng Việt: interpolation > RRF):
  score = dense_weight * norm(dense) + (1 - dense_weight) * norm(bm25),
  với dense đã ở thang 0-100 và BM25 chuẩn hóa max-norm (neo 0) về 0-100.
- "rrf" (fallback cổ điển): Reciprocal Rank Fusion với k=60.

Doc vắng mặt ở một nguồn bị tính điểm 0 ở nguồn đó (chuẩn cascade IR).
"""
from typing import Dict, List, Tuple


def _max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """Chuẩn hóa BM25 theo max, neo gốc 0.

    Min-max trên top-100 bị "saturation" khi corpus luật chứa boilerplate
    trùng lặp (hàng nghìn văn bản cùng câu 'Nghị định này có hiệu lực...'):
    top-100 dồn vào [42.36, 42.9] nên hit BM25 hạng #2 chỉ được ~39/100,
    tụt khỏi top-K dù là lexical match mạnh nhất. Neo 0 giữ đúng tỉ lệ
    sức mạnh tuyệt đối giữa các hit (42.6/43 → ~99)."""
    if not scores:
        return {}
    hi = max(scores.values())
    if hi <= 0:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: s / hi * 100.0 for doc_id, s in scores.items()}


def rrf_fuse(ranked_id_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)), rank bắt đầu từ 1."""
    fused: Dict[str, float] = {}
    for ranked in ranked_id_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def weighted_fuse(
    dense_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    dense_weight: float = 0.7,
) -> Dict[str, float]:
    """α-weighted interpolation trên thang điểm 0-100 đã chuẩn hóa."""
    w = max(0.0, min(1.0, float(dense_weight)))
    dense_norm = {doc_id: max(0.0, min(100.0, s)) for doc_id, s in dense_scores.items()}
    bm25_norm = _max_norm(bm25_scores)
    fused: Dict[str, float] = {}
    for doc_id in set(dense_norm) | set(bm25_norm):
        fused[doc_id] = round(w * dense_norm.get(doc_id, 0.0) + (1.0 - w) * bm25_norm.get(doc_id, 0.0), 3)
    return fused


def fuse(
    vector_ranked: List[Tuple[str, float]],
    bm25_ranked: List[Tuple[str, float]],
    method: str = "weighted",
    dense_weight: float = 0.7,
    rrf_k: int = 60,
) -> List[Tuple[str, float, float]]:
    """Fuse 2 nguồn retrieval.

    Args:
        vector_ranked: [(doc_id, similarity 0-100)] theo thứ tự dense rank.
        bm25_ranked: [(doc_id, bm25 raw score)] theo thứ tự BM25 rank.

    Returns:
        [(doc_id, fused_score, dense_similarity)] sắp xếp giảm dần fused_score.
        dense_similarity giữ lại để hiển thị % liên quan như cũ.
    """
    dense_map = {doc_id: sim for doc_id, sim in vector_ranked}
    if (method or "weighted").strip().lower() == "rrf":
        rrf = rrf_fuse(
            [[doc_id for doc_id, _ in vector_ranked], [doc_id for doc_id, _ in bm25_ranked]],
            k=rrf_k,
        )
        # Tie-break bằng doc_id để kết quả ổn định giữa các process
        # (iter/sort trên dict/set không đảm bảo thứ tự khi điểm bằng nhau).
        ordered = sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(doc_id, score, dense_map.get(doc_id, 0.0)) for doc_id, score in ordered]

    fused = weighted_fuse(dense_map, {doc_id: s for doc_id, s in bm25_ranked}, dense_weight)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(doc_id, score, dense_map.get(doc_id, 0.0)) for doc_id, score in ordered]
