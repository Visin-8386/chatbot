"""
Chat Memory Service - Persistent conversation history using SQLite.

Lưu lịch sử hội thoại theo session_id để hỗ trợ hội thoại dài
mà không cần truyền toàn bộ history qua API mỗi lần.

Changes vs v1:
- Module-level connection cache (avoids per-call connect/close overhead).
- Session title column: stores the first user message as a friendly name.
- All public function signatures remain identical.
"""
import sqlite3
import os
import time
import threading
from typing import List, Dict, Optional
from loguru import logger

from backend.config import BASE_DIR, MAX_HISTORY_TURNS

DB_PATH = os.path.join(BASE_DIR, "data", "chat_memory.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ── Cached connection (thread-safe) ──────────────────────────────────────────
# SQLite allows sharing a single connection across threads when
# check_same_thread=False; writes are serialised by a lock.
_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    """Return the shared SQLite connection, creating it on first call."""
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                _conn.row_factory = sqlite3.Row
                # WAL mode: allows concurrent reads while writing
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_db():
    """Khởi tạo schema database nếu chưa tồn tại."""
    conn = _get_conn()
    with _conn_lock:
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
        # Session metadata: friendly title (first user message)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_meta (
                session_id  TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT '',
                created_at  REAL NOT NULL
            )
        """)
        conn.commit()
    logger.info("Chat memory DB initialized at {}", DB_PATH)


def save_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Lưu một lượt hội thoại (user + assistant) vào DB."""
    if not session_id:
        return
    ts = time.time()
    conn = _get_conn()
    with _conn_lock:
        conn.execute(
            "INSERT INTO chat_turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_msg, ts)
        )
        # assistant slightly after user so ORDER BY created_at,id is deterministic
        conn.execute(
            "INSERT INTO chat_turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", assistant_msg, ts + 0.0001)
        )
        # Upsert session meta — title = first user message (truncated)
        title = user_msg[:80].strip()
        conn.execute(
            """
            INSERT INTO session_meta (session_id, title, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (session_id, title, ts)
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
    conn = _get_conn()
    # Subquery: take the LAST `limit` rows by id DESC, then re-sort ASC
    rows = conn.execute(
        """
        SELECT role, content FROM (
            SELECT id, role, content FROM chat_turns
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        ) ORDER BY id ASC
        """,
        (session_id, limit)
    ).fetchall()

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
    conn = _get_conn()
    with _conn_lock:
        cur = conn.execute("DELETE FROM chat_turns WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_meta WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount


def get_all_sessions() -> List[Dict]:
    """Lấy danh sách tất cả session và số lượt hội thoại (kèm title thân thiện)."""
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT s.session_id,
               s.turn_count,
               s.started_at,
               s.last_active_at,
               COALESCE(m.title, '') AS title,
               (SELECT content FROM chat_turns
                WHERE session_id = s.session_id AND role = 'user'
                ORDER BY created_at DESC LIMIT 1) AS last_query
        FROM (
            SELECT session_id,
                   COUNT(*) / 2  AS turn_count,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_active_at
            FROM chat_turns
            GROUP BY session_id
        ) s
        LEFT JOIN session_meta m ON m.session_id = s.session_id
        ORDER BY s.last_active_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]
