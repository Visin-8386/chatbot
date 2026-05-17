"""
Intent Classifier — Phân loại ý định người dùng trước khi vào pipeline RAG.

3 loại intent:
  - "chitchat"  : Chào hỏi, cảm ơn, hỏi về bot, trò chuyện phổ thông.
  - "followup"  : Hỏi tiếp / làm rõ câu trước đó (cần history).
  - "rag_query" : Câu hỏi tra cứu tài liệu thực sự.

Chiến lược: Rule-based trước (0ms), LLM fallback nếu mơ hồ.
"""
import re
import random
from typing import Literal, Dict, List

IntentType = Literal["chitchat", "followup", "rag_query"]

# ── Rule-based patterns ───────────────────────────────────────────────────────

_CHITCHAT_RE = re.compile(
    r"""
    ^\s*(
        # Chào hỏi
        (xin\s+)?ch[àa]o\b | hi\b | hello\b | hey\b | yo\b |
        good\s*(morning|afternoon|evening) | chúc\s+(buổi|ngày) |
        # Cảm ơn
        c[aả]m\s+[oơ]n | c[aá]m\s+[oơ]n | thanks?\b | thank\s+you | merci | ty\b |
        # Xác nhận ngắn / khen
        ok[eê]?\b | oke\b | đ[uưựợ][oơ]c\s*r[oồ]i | hi[eể]u\s*r[oồ]i |
        r[oõ]\s*r[oồ]i | vâng\b | d[aạ]\b | tốt | tuy[eệ]t | đỉnh | hay\s+(qu[aá]|v[aậ]y|đ[aấ]y|nh[ỉỉ]) |
        gi[oỏ]i | thông\s+minh | gi[uỏ]p\s+ích |
        # Hỏi về bot
        b[aạ]n\s+(l[aà]\s+ai|t[eê]n\s+g[ìi]|c[oó]\s+kh[oỏ]e|làm\s+đ[uư][oợ]c\s+g[ìi]) |
        m[aà]y\s+l[aà]\s+g[ìi] | em\s+(l[aà]\s+ai|l[aà]m\s+đ[uư][oợ]c\s+g[ìi]|c[oó]\s+th[eể]\s+gi[uú]p) |
        # Tạm biệt
        t[aạ]m\s+bi[eệ]t | bye\b | goodbye | chào\s+nhé | th[oô]i\s+nh[eé]
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_FOLLOWUP_RE = re.compile(
    r"""
    ^\s*(
        # Hỏi tiếp về cùng chủ đề
        v[aậ]y\s+(c[oò]n|th[ìi]|sao) | c[oò]n\s+v[eề] | th[eế]\s+c[oò]n |
        thêm\s+v[eề] | n[oó]i\s+(thêm|r[oõ]\s+h[oơ]n|chi\s+ti[eế]t) |
        gi[aả]i\s+thích\s+thêm | bi[eế]t\s+thêm | cho\s+(tôi|mình|em)\s+bi[eế]t\s+thêm |
        # Tham chiếu về phần đã nói
        [yý]\s+(b[aạ]n|em|anh)\s+n[oó]i | ph[aầ]n\s+(đ[oó]|n[aà]y|trên|d[uư][oớ]i) |
        đi[eề]u\s+\d+ | kho[aả]n\s+\d+ | m[uụ]c\s+\d+ | [àa]nh\s+nh[aư]\s+th[eế] |
        # Yêu cầu làm rõ hơn
        c[uụ]\s+th[eể]\s+h[oơ]n | v[ìi]\s+d[uụ] | t[aạ]i\s+sao\s+v[aậ]y |
        li[eê]n\s+quan | th[eế]\s+n[aà]o | nh[uư]\s+th[eế]\s+n[aà]o |
        # Tiếp nối câu trước
        ti[eế]p\s+theo | sau\s+đ[oó] | c[oò]n\s+g[ìi]\s+n[uữ]a | v[aà]\s+c[oò]n |
        # Hỏi ngắn mơ hồ (cần history mới hiểu)
        n[oó]\s+l[aà]\s+g[ìi] | c[aá]i\s+đ[oó] | đi[eề]u\s+đ[oó] | ph[aầ]n\s+đ[oó] |
        # === BỔ SUNG: Các câu phản hồi "chưa đủ / muốn thêm" ===
        ch[uư]a\s+(đ[uủ]|đ[aầ]y\s+đ[uủ]|r[oõ]) |       # chưa đủ, chưa đầy đủ, chưa rõ
        h[ìi]nh\s+nh[uư]\s+thi[eế]u | h[ìi]nh\s+nh[uư]\s+ch[uư]a |  # hình như thiếu, hình như chưa
        thi[eế]u\s+(th[oô]ng\s+tin|ph[aầ]n|m[uụ]c|n[oộ]i\s+dung) |  # thiếu thông tin/phần/mục
        th[eê]m\s+(đi|n[uữ]a|th[oô]ng\s+tin) |            # thêm đi, thêm nữa, thêm thông tin
        mu[oố]n\s+(bi[eế]t|xem|đ[oọ]c)\s+thêm |           # muốn biết/xem thêm
        c[oó]\s+g[ìi]\s+(kh[aá]c|thêm|n[uữ]a) |           # có gì khác/thêm/nữa
        đ[aã]\s+(h[eế]t|xong)\s+ch[uư]a |                 # đã hết/xong chưa
        c[oò]n\s+(n[uữ]a|kh[oô]ng|g[ìi]) |               # còn nữa, còn không, còn gì
        b[oổ]\s+sung | đ[aầ]y\s+đ[uủ]\s+h[oơ]n           # bổ sung, đầy đủ hơn
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ── Chitchat response templates ───────────────────────────────────────────────

_RESPONSES: Dict[str, List[str]] = {
    "greeting": [
        "Xin chào! 👋 Tôi là trợ lý AI nội bộ, chuyên tra cứu tài liệu công ty. Bạn cần tìm thông tin gì?",
        "Chào bạn! Tôi sẵn sàng giúp bạn tra cứu tài liệu. Hỏi đi nào!",
        "Xin chào! Bạn muốn tìm hiểu về quy trình, quy định, hay tài liệu kỹ thuật nào?",
    ],
    "thanks": [
        "Không có gì! Nếu cần tra cứu thêm, cứ hỏi mình nhé. 😊",
        "Rất vui khi giúp được bạn! Còn tài liệu nào cần tra cứu không?",
        "Không có gì ạ! Bạn có thắc mắc gì khác không?",
    ],
    "ack": [
        "Vâng! Bạn cần tìm hiểu thêm điều gì không?",
        "OK! Nếu có câu hỏi khác về tài liệu, mình luôn sẵn sàng.",
        "Hiểu rồi! Còn gì cần tra cứu thêm không?",
    ],
    "farewell": [
        "Tạm biệt! Chúc bạn một ngày làm việc hiệu quả! 👋",
        "Bái bai! Có việc gì cần tra cứu thì quay lại nhé!",
    ],
    "identity": [
        (
            "Tôi là **trợ lý AI nội bộ** được xây dựng để giúp bạn:\n"
            "- 🔍 Tra cứu tài liệu công ty (quy trình, quy định, hướng dẫn kỹ thuật)\n"
            "- 💬 Trả lời câu hỏi dựa trên nội dung tài liệu đã upload\n"
            "- 📌 Trích dẫn nguồn cụ thể từ tài liệu\n\n"
            "Bạn muốn tìm hiểu về vấn đề gì?"
        ),
    ],
    "compliment": [
        "Cảm ơn bạn! Tôi sẽ cố gắng giúp đỡ tốt hơn. 😊 Còn điều gì cần tra cứu không?",
        "Cảm ơn lời khen! Bạn có câu hỏi nào về tài liệu không?",
    ],
    "default": [
        "Tôi chuyên tra cứu tài liệu nội bộ công ty. Bạn muốn hỏi về quy trình, quy định, hay tài liệu kỹ thuật nào?",
        "Mình có thể giúp bạn tìm thông tin từ tài liệu công ty. Bạn cần tra cứu gì?",
    ],
}


def _chitchat_category(text: str) -> str:
    """Phân loại chitchat để lấy response phù hợp."""
    t = text.lower().strip()
    if re.search(r"ch[àa]o|hi\b|hello|hey|good\s*morning|chúc\s+buổi|chúc\s+ngày", t):
        return "greeting"
    if re.search(r"c[aả]m\s+[oơ]n|thanks|thank\s+you|merci|\bty\b", t):
        return "thanks"
    if re.search(r"ok|đư[oợ]c\s*r[oồ]i|hi[eể]u|r[oõ]\s*r[oồ]i|vâng|dạ\b", t):
        return "ack"
    if re.search(r"t[aạ]m\s+bi[eệ]t|bye|goodbye|chào\s+nhé", t):
        return "farewell"
    if re.search(r"b[aạ]n\s+l[aà]\s+ai|em\s+l[aà]\s+ai|mày\s+l[aà]\s+gì|l[aà]m\s+đ[uư][oợ]c\s+g[ìi]", t):
        return "identity"
    if re.search(r"tuy[eệ]t|đỉnh|gi[oỏ]i|hay\s+qu|thông\s+minh|gi[uú]p\s+ích", t):
        return "compliment"
    return "default"


def get_chitchat_response(text: str) -> str:
    """Trả về response chitchat phù hợp (không dùng LLM)."""
    category = _chitchat_category(text)
    return random.choice(_RESPONSES[category])


def classify_intent(query: str, has_history: bool = False) -> Dict:
    """
    Phân loại intent của query.

    Args:
        query: Câu hỏi của người dùng.
        has_history: True nếu đang có lịch sử hội thoại.

    Returns:
        {"intent": IntentType, "confidence": "high"|"low", "reason": str}
    """
    q = query.strip()
    if not q:
        return {"intent": "chitchat", "confidence": "high", "reason": "empty_query"}

    # 1. Chitchat rule-based (ưu tiên cao nhất)
    if _CHITCHAT_RE.match(q):
        return {"intent": "chitchat", "confidence": "high", "reason": "chitchat_pattern"}

    # 2. Câu quá ngắn + có history → follow-up
    word_count = len(q.split())
    if word_count <= 5 and has_history:
        if _FOLLOWUP_RE.search(q):
            return {"intent": "followup", "confidence": "high", "reason": "followup_pattern_short"}
        # Câu ngắn mơ hồ với history → coi là follow-up
        return {"intent": "followup", "confidence": "low", "reason": "short_with_history"}

    # 3. Follow-up pattern rõ ràng
    if _FOLLOWUP_RE.match(q):
        return {"intent": "followup", "confidence": "high", "reason": "followup_pattern"}

    # 4. Mặc định: RAG query
    return {"intent": "rag_query", "confidence": "high", "reason": "default_rag"}
