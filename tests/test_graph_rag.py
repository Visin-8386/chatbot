"""GraphRAG-lite tests: entity extraction, 1-hop expansion, router, cleanup.

Chạy offline, không cần LLM/GPU: dùng rule-based extraction + SQLite tạm.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.graph_store as gs


def _fresh_db(tmp_path):
    gs.GRAPH_DB_PATH = str(tmp_path / "kg_test.db")
    gs._conn = None
    gs.init_graph_db()


def test_rulebased_extracts_code_and_term():
    text = "Mã thiết bị MAINT-2024-SERVER đặt tại DC-01. Phòng Nhân Sự ban hành quy định Nghỉ Phép Năm."
    ents = gs.extract_entities_rulebased(text)
    names = [n for n, _t in ents]
    assert any("MAINT-2024-SERVER" in n for n in names), ents
    assert any("Nhân Sự" in n for n in names), ents
    # Stop entities phổ biến bị loại
    assert not any(n.casefold() == "quy định" for n in names), ents


def test_global_query_router():
    assert gs.is_global_query("Tổng hợp các thay đổi về chính sách lương trong 5 năm qua")
    assert gs.is_global_query("So sánh quy định nghỉ phép và nghỉ bù")
    assert not gs.is_global_query("Mã thiết bị MAINT-2024-SERVER ở đâu?")
    assert not gs.is_global_query("Xin chào")


def test_build_and_expand_1hop(tmp_path):
    _fresh_db(tmp_path)
    chunks = [
        {"text": "Phòng Nhân Sự ban hành quy định Nghỉ Phép Năm 12 ngày.", "metadata": {"source": "a.pdf"}},
        {"text": "Phòng Nhân Sự quản lý chế độ Bảo Hiểm Xã Hội cho nhân viên.", "metadata": {"source": "b.pdf"}},
        {"text": "Kho bãi nhập linh kiện điện tử theo quý.", "metadata": {"source": "c.pdf"}},
    ]
    stats = gs.build_from_chunks("doc1", chunks)
    assert stats["chunks"] == 3
    assert stats["entities"] > 0
    assert stats["edges"] > 0

    base = [{
        "text": chunks[0]["text"],
        "match_text": chunks[0]["text"],
        "metadata": {"source": "a.pdf", "doc_id": "doc1", "chunk_index": 0},
        "similarity": 80.0,
    }]
    out = gs.expand_with_graph("Nghỉ Phép Năm do ai ban hành?", base, max_extra=3)
    # Chunk b.pdf chung entity "Phòng Nhân Sự" phải được kéo vào
    assert len(out) == 2, out
    assert out[1]["metadata"].get("graph_expanded") is True
    assert "Bảo Hiểm" in out[1]["text"]
    assert out[1]["similarity"] < 80.0  # penalty áp dụng


def test_expand_empty_graph_is_safe(tmp_path):
    _fresh_db(tmp_path)
    base = [{"text": "abc", "match_text": "abc", "metadata": {}, "similarity": 50.0}]
    out = gs.expand_with_graph("câu hỏi bất kỳ", base)
    assert out == base


def test_delete_doc_graph(tmp_path):
    _fresh_db(tmp_path)
    chunks = [
        {"text": "Phòng Nhân Sự ban hành Quy Định Mới.", "metadata": {"source": "a.pdf"}},
    ]
    gs.build_from_chunks("docX", chunks)
    assert gs.graph_stats()["edges"] >= 0
    gs.delete_doc_graph("docX")
    stats = gs.graph_stats()
    assert stats["chunks"] == 0
    assert stats["edges"] == 0
