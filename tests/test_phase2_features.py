"""
Tests cho Giai đoạn 2: Agentic RAG Features
  - 2.1 HyDE (Query Expansion)
  - 2.2 CRAG (Corrective RAG Relevance Gate)
  - 2.3 Persistent Chat Memory (SQLite)
"""
import os
import sys
import uuid
import tempfile
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Override DB path to temp file trước khi import ──────────────────────────
_tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
os.environ["CHAT_MEMORY_DB_PATH"] = _tmp_db.name  # chat_memory sẽ đọc biến này nếu có

# Patch DB_PATH trực tiếp sau khi import
import backend.chat_memory as _cm
_cm.DB_PATH = _tmp_db.name

from backend.chat_memory import init_db, save_turn, get_history, delete_session, get_all_sessions
from backend.generator import hyde_expand_query, crag_check_relevance


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def fresh_db():
    """Khởi tạo DB mới trước mỗi test."""
    import sqlite3
    # Xóa sạch tables nếu tồn tại
    conn = sqlite3.connect(_tmp_db.name)
    conn.execute("DROP TABLE IF EXISTS chat_turns")
    conn.commit()
    conn.close()
    init_db()
    yield


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3 — Persistent Chat Memory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChatMemory:

    def test_init_db_creates_table(self):
        """DB phải có bảng chat_turns sau khi init."""
        import sqlite3
        conn = sqlite3.connect(_tmp_db.name)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "chat_turns" in table_names

    def test_save_and_get_history_single_turn(self):
        """Lưu 1 lượt rồi lấy về phải khớp."""
        session_id = str(uuid.uuid4())
        save_turn(session_id, "Lương của tôi là bao nhiêu?", "Lương của bạn là 10 triệu/tháng.")
        history = get_history(session_id)
        assert len(history) == 1
        assert history[0]["user"] == "Lương của tôi là bao nhiêu?"
        assert "10 triệu" in history[0]["assistant"]

    def test_save_multiple_turns_ordering(self):
        """Nhiều lượt phải được trả về theo thứ tự chronological."""
        session_id = str(uuid.uuid4())
        save_turn(session_id, "Câu hỏi 1", "Trả lời 1")
        save_turn(session_id, "Câu hỏi 2", "Trả lời 2")
        save_turn(session_id, "Câu hỏi 3", "Trả lời 3")
        history = get_history(session_id)
        assert len(history) == 3
        assert history[0]["user"] == "Câu hỏi 1"
        assert history[2]["user"] == "Câu hỏi 3"

    def test_get_history_empty_session(self):
        """Session chưa có lịch sử phải trả về list rỗng."""
        history = get_history("session-khong-ton-tai")
        assert history == []

    def test_get_history_with_max_turns(self):
        """max_turns phải giới hạn số lượt trả về."""
        session_id = str(uuid.uuid4())
        for i in range(10):
            save_turn(session_id, f"Q{i}", f"A{i}")
        history = get_history(session_id, max_turns=3)
        assert len(history) <= 3

    def test_delete_session(self):
        """Xóa session phải xóa hết lịch sử."""
        session_id = str(uuid.uuid4())
        save_turn(session_id, "Q", "A")
        deleted = delete_session(session_id)
        assert deleted > 0
        history = get_history(session_id)
        assert history == []

    def test_session_isolation(self):
        """Hai session không được lẫn lộn lịch sử."""
        s1 = str(uuid.uuid4())
        s2 = str(uuid.uuid4())
        save_turn(s1, "Session 1 Q", "Session 1 A")
        save_turn(s2, "Session 2 Q", "Session 2 A")
        h1 = get_history(s1)
        h2 = get_history(s2)
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["user"] != h2[0]["user"]

    def test_get_all_sessions(self):
        """get_all_sessions phải liệt kê đúng các session đã lưu."""
        s1 = str(uuid.uuid4())
        s2 = str(uuid.uuid4())
        save_turn(s1, "Q", "A")
        save_turn(s2, "Q", "A")
        sessions = get_all_sessions()
        session_ids = [s["session_id"] for s in sessions]
        assert s1 in session_ids
        assert s2 in session_ids

    def test_save_turn_empty_session_id_is_noop(self):
        """session_id rỗng không được ghi vào DB."""
        save_turn("", "Q", "A")  # Không raise exception
        save_turn(None, "Q", "A")  # type: ignore
        sessions = get_all_sessions()
        assert all(s["session_id"] for s in sessions)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2 — CRAG Tests (không cần LLM — chỉ test fast-path bằng similarity score)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCRAG:

    def _make_results(self, similarity: float) -> list:
        return [{"text": "Nội dung tài liệu mẫu.", "metadata": {}, "similarity": similarity}]

    def test_irrelevant_when_no_results(self):
        result = crag_check_relevance("câu hỏi bất kỳ", [])
        assert result["verdict"] == "irrelevant"

    def test_relevant_when_high_similarity(self):
        """Similarity cao (>= threshold) phải được phán là relevant ngay."""
        result = crag_check_relevance("quy trình nghỉ phép", self._make_results(80.0))
        assert result["verdict"] == "relevant"
        assert len(result["filtered_results"]) > 0

    def test_irrelevant_when_very_low_similarity(self):
        """Similarity cực thấp (<= threshold * 0.5) phải là irrelevant."""
        result = crag_check_relevance("câu hỏi ngẫu nhiên", self._make_results(10.0))
        assert result["verdict"] == "irrelevant"
        assert result["filtered_results"] == []

    def test_filtered_results_empty_when_irrelevant(self):
        """filtered_results phải rỗng khi verdict là irrelevant."""
        result = crag_check_relevance("xyz", self._make_results(5.0))
        assert result["filtered_results"] == []

    def test_relevant_returns_original_results(self):
        """Khi relevant, filtered_results phải trả về đủ kết quả gốc."""
        docs = self._make_results(90.0)
        result = crag_check_relevance("lương thưởng", docs)
        assert result["filtered_results"] == docs


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 — HyDE Tests (test khi ENABLE_HYDE=False — trả về query gốc)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHyDE:

    def test_hyde_disabled_returns_original_query(self, monkeypatch):
        """Khi ENABLE_HYDE=False, phải trả về query gốc không thay đổi."""
        import backend.generator as gen
        monkeypatch.setattr(gen, "ENABLE_HYDE", False)
        query = "Chính sách nghỉ phép hàng năm là gì?"
        result = hyde_expand_query(query)
        assert result == query

    def test_hyde_disabled_type_is_string(self, monkeypatch):
        """Kết quả phải luôn là string."""
        import backend.generator as gen
        monkeypatch.setattr(gen, "ENABLE_HYDE", False)
        result = hyde_expand_query("test query")
        assert isinstance(result, str)
