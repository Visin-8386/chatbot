"""
Chat Memory Service - Persistent conversation history using SQLite.

Lưu lịch sử hội thoại theo session_id để hỗ trợ hội thoại dài
mà không cần truyền toàn bộ history qua API mỗi lần.
"""
import sqlite3
import json
import os
import time
from typing import List, Dict, Optional
from loguru import logger

from backend.config import BASE_DIR, MAX_HISTORY_TURNS

DB_PATH = os.path.join(BASE_DIR, "data", "chat_memory.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    """Mở kết nối SQLite (thread-safe check_same_thread=False cho FastAPI)."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo schema database nếu chưa tồn tại."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_turns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,  -- 'user' | 'assistant'
                content     TEXT    NOT NULL,
                created_at  REAL    NOT NULL   -- Unix timestamp
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_time
            ON chat_turns (session_id, created_at)
        """)
        conn.commit()
    logger.info("Chat memory DB initialized at {}", DB_PATH)


def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Lưu một lượt hội thoại (user + assistant) vào DB."""
    if not session_id:
        return
    ts = time.time()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_msg, ts)
        )
        conn.execute(
            "INSERT INTO chat_turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", assistant_msg, ts + 0.001)
        )
        conn.commit()


def get_history(session_id: str, max_turns: Optional[int] = None) -> List[Dict]:
    """
    Lấy lịch sử hội thoại của một session, dạng list[{user, assistant}].

    Args:
        session_id: ID phiên chat.
        max_turns:  Số lượt tối đa cần lấy (mặc định dùng MAX_HISTORY_TURNS).

    Returns:
        List[{"user": str, "assistant": str}]
    """
    if not session_id:
        return []

    limit = (max_turns or MAX_HISTORY_TURNS) * 2  # *2 vì mỗi lượt có 2 rows
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM chat_turns
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit)
        ).fetchall()

    # Đảo ngược để có thứ tự chronological
    rows = list(reversed(rows))

    # Ghép cặp user-assistant
    turns = []
    i = 0
    while i < len(rows):
        if rows[i]["role"] == "user":
            user_content = rows[i]["content"]
            assistant_content = rows[i + 1]["content"] if i + 1 < len(rows) else ""
            turns.append({"user": user_content, "assistant": assistant_content})
            i += 2
        else:
            i += 1  # Bỏ qua row lẻ không khớp

    return turns


def delete_session(session_id: str) -> int:
    """Xoá toàn bộ lịch sử của một session. Trả về số row đã xoá."""
    if not session_id:
        return 0
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount


def get_all_sessions() -> List[Dict]:
    """Lấy danh sách tất cả session và số lượt hội thoại."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id,
                   COUNT(*) / 2  AS turn_count,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_active_at
            FROM chat_turns
            GROUP BY session_id
            ORDER BY last_active_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]
