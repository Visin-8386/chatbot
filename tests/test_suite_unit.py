"""
Unit Test Suite — Không cần server chạy.
Covers: config, chunking, embedding, vector_store, chat_memory, CRAG, HyDE-disabled.
"""
import os, sys, uuid, tempfile, time, re
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Patch DB path trước khi import ─────────────────────────────────────────
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()

# Mock llama_cpp trước khi bất kỳ module nào của backend được import
from unittest.mock import MagicMock
import sys
sys.modules["llama_cpp"] = MagicMock()

import backend.chat_memory as _cm
_cm.DB_PATH = _tmp_db.name

from backend.chat_memory import init_db, save_turn, get_history, delete_session, get_all_sessions
from backend.generator import crag_check_relevance, hyde_expand_query


# ══════════════════════════════════════════════════════════════════
# Fixture
# ══════════════════════════════════════════════════════════════════
@pytest.fixture(autouse=True)
def fresh_db():
    import sqlite3
    conn = sqlite3.connect(_tmp_db.name)
    conn.execute("DROP TABLE IF EXISTS chat_turns")
    conn.commit(); conn.close()
    init_db()
    yield


# ══════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════
class TestConfig:
    def test_critical_paths_exist(self):
        from backend.config import UPLOAD_DIR, CHROMA_DIR, MODELS_DIR
        assert os.path.isdir(UPLOAD_DIR)
        assert os.path.isdir(CHROMA_DIR)
        assert os.path.isdir(MODELS_DIR)

    def test_top_k_positive(self):
        from backend.config import TOP_K
        assert TOP_K >= 1

    def test_max_context_chars_reasonable(self):
        from backend.config import MAX_CONTEXT_CHARS
        assert 500 <= MAX_CONTEXT_CHARS <= 20000

    def test_supported_extensions(self):
        from backend.config import SUPPORTED_EXTENSIONS
        for ext in [".pdf", ".docx", ".txt"]:
            assert ext in SUPPORTED_EXTENSIONS

    def test_phase2_config_defaults(self):
        from backend.config import ENABLE_HYDE, CRAG_RELEVANCE_THRESHOLD, ENABLE_PERSISTENT_MEMORY
        assert isinstance(ENABLE_HYDE, bool)
        assert 0 < CRAG_RELEVANCE_THRESHOLD < 100
        assert isinstance(ENABLE_PERSISTENT_MEMORY, bool)


# ══════════════════════════════════════════════════════════════════
# 2. CHUNKING
# ══════════════════════════════════════════════════════════════════
class TestChunking:
    def test_basic_chunking(self):
        from backend.document_processor import chunk_text
        text = "Từ " * 300
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2
        assert all(chunk.strip() for chunk in chunks)

    def test_chunk_size_respected(self):
        from backend.document_processor import chunk_text
        text = "abcde " * 500
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert all(len(c) <= 130 for c in chunks), "Chunk quá lớn"

    def test_overlap_creates_continuity(self):
        """Semantic chunker không dùng overlap cứng -- kiểm tra toàn bộ nội dung được phân chia đầy đủ."""
        from backend.document_processor import chunk_text
        words = [f"W{i:03d}" for i in range(60)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=80, overlap=30)
        assert len(chunks) >= 2, "Phải tạo từ 2 chunk trở lên"
        # Kiểm tra không mất nội dung: mọi word phải xuất hiện trong tập hợp tất cả chunk
        all_recovered = " ".join(chunks)
        for word in words:
            assert word in all_recovered, f"Đã mất từ: {word}"

    def test_short_text_single_chunk(self):
        from backend.document_processor import chunk_text
        text = "Nội dung ngắn."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_table_header_preserved(self):
        from backend.document_processor import chunk_text
        table = "Col A | Col B | Col C\n" + "val | " * 3 + "\n" * 20
        text = table * 5
        chunks = chunk_text(text, chunk_size=80, overlap=0)
        assert len(chunks) >= 2

    def test_margin_cleanup(self):
        from backend.document_processor import (
            _split_lines, _collect_repeated_margin_signatures,
            _remove_repeated_margin_lines
        )
        pages = [f"HEADER\nContent page {i}\nFOOTER" for i in range(3)]
        page_lines = [_split_lines(p) for p in pages]
        repeated = _collect_repeated_margin_signatures(page_lines)
        assert "header" in repeated and "footer" in repeated
        cleaned = [_remove_repeated_margin_lines(ls, repeated) for ls in page_lines]
        for page in cleaned:
            assert not any("HEADER" in l or "FOOTER" in l for l in page)


# ══════════════════════════════════════════════════════════════════
# 3. EMBEDDING SERVICE
# ══════════════════════════════════════════════════════════════════
class TestEmbedding:
    def test_embed_query_returns_vector(self):
        from backend.embedding_service import embed_query
        vec = embed_query("Quy trình nghỉ phép")
        assert isinstance(vec, list)
        assert len(vec) > 100
        assert all(isinstance(v, float) for v in vec[:5])

    def test_embed_passages_batch(self):
        from backend.embedding_service import embed_passages
        texts = ["Văn bản thứ nhất.", "Văn bản thứ hai.", "Văn bản thứ ba."]
        vecs = embed_passages(texts)
        assert len(vecs) == 3
        assert all(len(v) == len(vecs[0]) for v in vecs)

    def test_embedding_dimension_consistent(self):
        from backend.embedding_service import embed_query, embed_passages
        q_vec = embed_query("test")
        p_vecs = embed_passages(["test passage"])
        assert len(q_vec) == len(p_vecs[0])

    def test_different_queries_different_vectors(self):
        from backend.embedding_service import embed_query
        v1 = embed_query("Nghỉ phép")
        v2 = embed_query("Lương thưởng")
        diff = sum(abs(a - b) for a, b in zip(v1, v2))
        assert diff > 0.1, "Hai query khác nhau phải có embedding khác nhau"


# ══════════════════════════════════════════════════════════════════
# 4. VECTOR STORE (dùng collection riêng để không ảnh hưởng data thật)
# ══════════════════════════════════════════════════════════════════
TEST_DOC_ID = "unit_test_doc_" + uuid.uuid4().hex[:8]

@pytest.fixture(scope="class")
def seeded_collection():
    """Thêm docs test, yield, rồi xóa."""
    from backend.vector_store import add_documents, delete_document
    docs = [
        {"text": "Quy trình nghỉ phép năm được quy định tại điều 5 nội quy lao động.", "metadata": {"source": "nq_lao_dong.pdf"}},
        {"text": "Mức lương tối thiểu vùng 2024 là 4.680.000 đồng/tháng cho vùng I.", "metadata": {"source": "luong_2024.pdf"}},
        {"text": "Mã thiết bị MAINT-2024-SERVER dùng cho máy chủ tại DC-01.", "metadata": {"source": "equipment.txt"}},
    ]
    add_documents(docs, TEST_DOC_ID)
    yield docs
    delete_document(TEST_DOC_ID)

class TestVectorStore:
    def test_add_and_count(self, seeded_collection):
        from backend.vector_store import get_collection
        col = get_collection()
        assert col.count() >= 3

    def test_semantic_search_finds_relevant(self, seeded_collection):
        from backend.vector_store import search
        results = search("làm thế nào để xin nghỉ phép?", top_k=3)
        assert len(results) >= 1
        top_text = results[0]["text"].lower()
        assert any(w in top_text for w in ["nghỉ", "phép", "lao động"])

    def test_keyword_search_finds_exact_code(self, seeded_collection):
        from backend.vector_store import search
        results = search("MAINT-2024-SERVER", top_k=3)
        assert any("MAINT-2024-SERVER" in r["text"] for r in results)

    def test_search_result_has_required_fields(self, seeded_collection):
        from backend.vector_store import search
        results = search("lương", top_k=3)
        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "similarity" in r

    def test_similarity_in_valid_range(self, seeded_collection):
        from backend.vector_store import search
        results = search("quy định nội quy", top_k=3)
        for r in results:
            assert 0 <= r["similarity"] <= 100

    def test_delete_document(self):
        from backend.vector_store import add_documents, delete_document, get_collection
        tmp_id = "tmp_" + uuid.uuid4().hex[:6]
        add_documents([{"text": "Tài liệu tạm để xóa.", "metadata": {"source": "tmp.txt"}}], tmp_id)
        before = get_collection().count()
        deleted = delete_document(tmp_id)
        after = get_collection().count()
        assert deleted >= 1
        assert after < before


# ══════════════════════════════════════════════════════════════════
# 5. CHAT MEMORY
# ══════════════════════════════════════════════════════════════════
class TestChatMemory:
    def test_init_creates_table(self, fresh_db):
        """DB phải có bảng chat_turns sau init_db()."""
        import sqlite3
        init_db()  # gọi lại để chắc chắn đã tạo table vào DB hiện tại
        conn = sqlite3.connect(_cm.DB_PATH)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        assert "chat_turns" in tables

    def test_save_and_retrieve(self):
        sid = uuid.uuid4().hex
        save_turn(sid, "Câu hỏi 1", "Trả lời 1")
        hist = get_history(sid)
        assert len(hist) == 1
        assert hist[0]["user"] == "Câu hỏi 1"

    def test_multiple_turns_ordered(self):
        sid = uuid.uuid4().hex
        for i in range(5):
            save_turn(sid, f"Q{i}", f"A{i}")
        hist = get_history(sid)
        assert len(hist) == 5
        assert hist[0]["user"] == "Q0"
        assert hist[4]["user"] == "Q4"

    def test_max_turns_limit(self):
        sid = uuid.uuid4().hex
        for i in range(10):
            save_turn(sid, f"Q{i}", f"A{i}")
        hist = get_history(sid, max_turns=3)
        assert len(hist) <= 3

    def test_session_isolation(self):
        s1, s2 = uuid.uuid4().hex, uuid.uuid4().hex
        save_turn(s1, "S1 Q", "S1 A")
        save_turn(s2, "S2 Q", "S2 A")
        assert get_history(s1)[0]["user"] == "S1 Q"
        assert get_history(s2)[0]["user"] == "S2 Q"

    def test_delete_session(self):
        sid = uuid.uuid4().hex
        save_turn(sid, "Q", "A")
        delete_session(sid)
        assert get_history(sid) == []

    def test_empty_session_id_noop(self):
        save_turn("", "Q", "A")
        save_turn(None, "Q", "A")  # type: ignore

    def test_get_all_sessions_lists_all(self):
        s1, s2 = uuid.uuid4().hex, uuid.uuid4().hex
        save_turn(s1, "Q", "A")
        save_turn(s2, "Q", "A")
        ids = [s["session_id"] for s in get_all_sessions()]
        assert s1 in ids and s2 in ids


# ══════════════════════════════════════════════════════════════════
# 6. CRAG
# ══════════════════════════════════════════════════════════════════
def _docs(sim): return [{"text": "Nội dung tài liệu.", "metadata": {}, "similarity": sim}]

class TestCRAG:
    def test_no_results_irrelevant(self):
        assert crag_check_relevance("q", [])["verdict"] == "irrelevant"

    def test_high_similarity_relevant(self):
        r = crag_check_relevance("nghỉ phép", _docs(85))
        assert r["verdict"] == "relevant"
        assert len(r["filtered_results"]) > 0

    def test_very_low_similarity_irrelevant(self):
        r = crag_check_relevance("xyz", _docs(5))
        assert r["verdict"] == "irrelevant"
        assert r["filtered_results"] == []

    def test_relevant_keeps_results(self):
        docs = _docs(90)
        r = crag_check_relevance("lương", docs)
        assert r["filtered_results"] == docs

    def test_has_reason_field(self):
        r = crag_check_relevance("test", _docs(80))
        assert "reason" in r and r["reason"]


# ══════════════════════════════════════════════════════════════════
# 7. HYDE (disabled path — không cần LLM)
# ══════════════════════════════════════════════════════════════════
class TestHyDE:
    def test_disabled_returns_original(self, monkeypatch):
        import backend.generator as gen
        monkeypatch.setattr(gen, "ENABLE_HYDE", False)
        q = "Chính sách nghỉ phép?"
        assert hyde_expand_query(q) == q

    def test_result_is_string(self, monkeypatch):
        import backend.generator as gen
        monkeypatch.setattr(gen, "ENABLE_HYDE", False)
        assert isinstance(hyde_expand_query("test"), str)
