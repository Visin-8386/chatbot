"""Hybrid fusion tests: weighted interpolation vs RRF (pure, no DB/GPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.hybrid_fusion import rrf_fuse, weighted_fuse, fuse


def test_weighted_prefers_both_signals():
    dense = {"a": 90.0, "b": 85.0}          # a nhỉnh dense hơn b
    bm25 = {"b": 8.0, "c": 12.0, "d": 2.0}  # b khớp keyword mức giữa
    out = weighted_fuse(dense, bm25, dense_weight=0.7)
    assert out["b"] > out["a"] > out["c"], out
    assert all(0.0 <= v <= 100.0 for v in out.values())


def test_weighted_missing_source_counts_zero():
    out = weighted_fuse({"a": 80.0}, {}, dense_weight=0.7)
    assert out == {"a": 56.0}, out  # 0.7*80 + 0.3*0


def test_weighted_single_bm25_scales_to_100():
    out = weighted_fuse({}, {"x": 5.0}, dense_weight=0.7)
    assert out == {"x": 30.0}, out  # min-max(đơn trị)=100 -> 0.3*100


def test_rrf_matches_classic_formula():
    out = rrf_fuse([["a", "b"], ["b", "a"]], k=60)
    assert abs(out["a"] - (1 / 61 + 1 / 62)) < 1e-9
    assert abs(out["b"] - (1 / 61 + 1 / 62)) < 1e-9


def test_fuse_weighted_order_and_dense_passthrough():
    vector = [("a", 90.0), ("b", 60.0)]
    bm25 = [("b", 11.0), ("c", 5.0), ("d", 1.0)]
    ordered = fuse(vector, bm25, method="weighted", dense_weight=0.7)
    ids = [doc_id for doc_id, _, _ in ordered]
    assert ids[0] == "b", ids  # cả 2 nguồn thắng
    by_id = {doc_id: (fused, dense) for doc_id, fused, dense in ordered}
    assert by_id["a"][1] == 90.0 and by_id["c"][1] == 0.0


def test_fuse_rrf_rollback_path():
    vector = [("a", 90.0), ("b", 60.0)]
    bm25 = [("b", 9.0)]
    ordered = fuse(vector, bm25, method="rrf", rrf_k=60)
    ids = [doc_id for doc_id, _, _ in ordered]
    assert ids[0] == "b", ids  # b có mặt ở cả 2 rank
    by_id = {doc_id: (fused, dense) for doc_id, fused, dense in ordered}
    assert by_id["b"][1] == 60.0  # rrf giữ similarity dense gốc


def test_fuse_empty_inputs():
    assert fuse([], [], method="weighted") == []
    assert fuse([], [], method="rrf") == []
