import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure backend is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock llama_cpp before any backend imports that might use it
from unittest.mock import MagicMock
sys.modules["llama_cpp"] = MagicMock()

from backend.intent_classifier import classify_intent
from backend.document_processor import _remove_repeated_margin_lines, _split_into_blocks, _is_heading_block
from backend.generator import select_relevant_history, groundedness_score, _prepare_context_and_sources

class TestRealisticScenarios:

    # --- Intent Classification Scenarios ---
    
    @pytest.mark.parametrize("query, has_history, expected_intent", [
        ("Xin chào bạn", False, "chitchat"),
        ("Cảm ơn nhé", True, "chitchat"),
        ("Quy trình nghỉ phép như thế nào?", False, "rag_query"),
        ("Thế còn lương thưởng thì sao?", True, "followup"),  # followup pattern
        ("Nói rõ hơn đi", True, "followup"),                # followup pattern
        ("Nó là cái gì?", True, "followup"),                 # short with history
        ("Ai là người phê duyệt?", False, "rag_query"),      # no history, default to rag
        ("ok hiểu rồi", True, "chitchat"),
    ])
    def test_intent_classification_real_world(self, query, has_history, expected_intent):
        """Kiểm tra phân loại ý định với các câu hỏi thực tế."""
        result = classify_intent(query, has_history=has_history)
        assert result["intent"] == expected_intent

    # --- Document Processing Scenarios ---

    def test_margin_removal_realistic(self):
        """Kiểm tra việc loại bỏ header/footer lặp lại (ví dụ: 'Công ty ABC - Quy định nội bộ')."""
        page_lines = [
            ["Công ty ABC", "Quy định nghỉ phép", "Nội dung trang 1", "Trang 1/2"],
            ["Công ty ABC", "Quy định nghỉ phép", "Nội dung trang 2", "Trang 2/2"]
        ]
        repeated = {"công ty abc", "quy định nghỉ phép"} # signatures
        
        cleaned_p1 = _remove_repeated_margin_lines(page_lines[0], repeated)
        assert "Công ty ABC" not in cleaned_p1
        assert "Quy định nghỉ phép" not in cleaned_p1
        assert "Nội dung trang 1" in cleaned_p1
        # Trang 1/2 is usually also removed if detected as page number
        assert not any("Trang" in l for l in cleaned_p1)

    def test_heading_detection_vietnamese(self):
        """Kiểm tra nhận diện tiêu đề tiếng Việt."""
        assert _is_heading_block("I. QUY ĐỊNH CHUNG") is True
        assert _is_heading_block("1.1. Điều khoản thi hành") is True
        assert _is_heading_block("Chương II: Quyền lợi nhân viên") is True
        assert _is_heading_block("Đây là một đoạn văn bản bình thường không phải tiêu đề.") is False

    def test_semantic_block_splitting_with_headings(self):
        """Kiểm tra việc gộp tiêu đề vào khối nội dung tiếp theo."""
        text = "I. TIÊU ĐỀ 1\n\nNội dung của phần 1.\n\nII. TIÊU ĐỀ 2\n\nNội dung của phần 2."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 2
        assert "TIÊU ĐỀ 1" in blocks[0]
        assert "Nội dung của phần 1" in blocks[0]
        assert "TIÊU ĐỀ 2" in blocks[1]

    # --- RAG Logic Scenarios ---

    def test_history_selection_relevance(self):
        """Kiểm tra việc giữ lại history chỉ khi nó liên quan đến topic hiện tại."""
        history = [
            {"user": "Quy trình nghỉ phép năm?", "assistant": "Bạn cần nộp đơn trước 3 ngày..."},
        ]
        
        # Câu hỏi liên quan (nghỉ phép) -> Giữ history
        h1 = select_relevant_history("Vậy nếu nghỉ 5 ngày thì sao?", history)
        assert len(h1) == 1
        
        # Câu hỏi đổi chủ đề hoàn toàn -> Bỏ history
        h2 = select_relevant_history("Mức lương tối thiểu là bao nhiêu?", history)
        assert len(h2) == 0

    def test_groundedness_score_calculation(self):
        """Kiểm tra tính điểm groundedness (độ bám sát tài liệu)."""
        answer = "Nhân viên được nghỉ 12 ngày phép mỗi năm."
        # Case 1: Relevant context
        context = [{"text": "Theo quy định, mỗi năm nhân viên có 12 ngày nghỉ phép hưởng lương."}]
        score1 = groundedness_score(answer, context)
        assert score1 > 0.5
        
        # Case 2: Irrelevant context
        context_irrelevant = [{"text": "Công ty cung cấp trà và cà phê miễn phí tại văn phòng."}]
        score2 = groundedness_score(answer, context_irrelevant)
        assert score2 < 0.2

    # --- Integration Flow (Mocked LLM) ---

    @patch("backend.generator._run_chat")
    def test_prepare_context_budget(self, mock_chat):
        """Kiểm tra việc giới hạn context window theo số ký tự tối đa."""
        results = [
            {"text": "A" * 1000, "metadata": {"source": "doc1.pdf"}, "similarity": 90},
            {"text": "B" * 1000, "metadata": {"source": "doc2.pdf"}, "similarity": 85},
            {"text": "C" * 1000, "metadata": {"source": "doc3.pdf"}, "similarity": 80},
        ]
        
        # Giả sử MAX_CONTEXT_CHARS = 2500
        prepared = _prepare_context_and_sources(results, max_context_chars=2500)
        
        # Nên chỉ lấy doc1 và doc2 (vì doc1 + doc2 > 1500 nếu tính cả prefix)
        # Thực tế prefix là "[Nguồn 1: doc1.pdf]\n" ~ 20 chars
        assert "doc1.pdf" in prepared["context_text"]
        assert "doc2.pdf" in prepared["context_text"]
        assert "doc3.pdf" not in prepared["context_text"]
        assert len(prepared["sources_info"]) == 2

    def test_is_follow_up_query_logic(self):
        """Kiểm tra logic nhận diện follow-up query."""
        from backend.generator import _is_follow_up_query
        assert _is_follow_up_query("Vậy còn nó thì sao?") is True
        assert _is_follow_up_query("Giải thích thêm") is True
        assert _is_follow_up_query("Mức lương?") is True  # Short query
        assert _is_follow_up_query("Quy định về thời gian làm việc chính thức là gì?") is False

if __name__ == "__main__":
    pytest.main([__file__])
