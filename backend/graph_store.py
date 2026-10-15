"""
Graph Store - Lightweight Knowledge Graph for GraphRAG-lite (SQLite, stdlib only).

Mục tiêu: bổ sung vector search bằng quan hệ thực thể để trả lời câu hỏi
tổng hợp / multi-hop ("tổng hợp chính sách 5 năm", "so sánh quy định A và B").

Thiết kế an toàn cho RTX 3060 6GB + 100% offline:
- Không thêm dependency, không tốn VRAM (SQLite quan hệ + rule-based extraction).
- LLM triple extraction là optional (GRAPH_USE_LLM=1), mặc định tắt.
- Mọi lỗi graph không bao giờ làm hỏng ingest/retrieval chính (try/except ở caller).
- Bật/tắt qua env ENABLE_GRAPH_RAG (mặc định 0 = giữ nguyên hành vi cũ).
"""
import os
import re
import sqlite3
import threading
from typing import List, Dict, Set, Tuple, Optional
from loguru import logger

from backend.config import BASE_DIR

GRAPH_DB_PATH = os.getenv(
    "GRAPH_DB_PATH",
    os.path.join(BASE_DIR, "data", "kg.db"),
).strip()
os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)

_conn: Optional[sqlite3.Connection] = None
_conn_lock = threading.Lock()

#Nhận diện query tổng hợp (global) để bật graph expansion
_GLOBAL_RE = re.compile(
    r"tổng\s*hợp|so\s*sánh|liệt\s*kê|toàn\s*bộ|tất\s*cả|các\s*thay\s*đổi"
    r"|phân\s*tích\s*chung|đánh\s*giá\s*tổng|mối\s*quan\s*hệ|liên\s*quan.*(nào|gì)",
    re.IGNORECASE,
)

# Mã nội bộ: MAINT-2024-SERVER, PCCC, DC-01, QD-12/2024 ...
_CODE_RE = re.compile(r"\b[A-ZĐÂÊÔƠƯ]{1,}[A-ZĐÂÊÔƠƯ0-9\-_/]{2,}\b")
# Cụm danh từ viết hoa: "Phòng Nhân Sự", "Nghỉ Phép Năm", "Bảo Hiểm Xã Hội"
_TERM_RE = re.compile(
    r"\b[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+"
    r"(?:\s+[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]"
    r"[a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+){0,4}\b"
)

_STOP_ENTITIES = {
    "công ty", "nhân viên", "quy định", "quy trình", "chính sách", "tài liệu",
    "thông tin", "trợ lý", "doanh nghiệp", "việt nam",
}


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _conn_lock:
            if _conn is None:
                _conn = sqlite3.connect(GRAPH_DB_PATH, check_same_thread=False)
                _conn.row_factory = sqlite3.Row
                _conn.execute("PRAGMA journal_mode=WAL")
                _conn.execute("PRAGMA synchronous=NORMAL")
    return _conn


def init_graph_db() -> None:
    conn = _get_conn()
    with _conn_lock:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT NOT NULL,
                name_cf TEXT NOT NULL UNIQUE,
                etype   TEXT NOT NULL DEFAULT 'TERM'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_chunks (
                chunk_key TEXT PRIMARY KEY,
                doc_id    TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                source    TEXT NOT NULL DEFAULT '',
                page      INTEGER,
                snippet   TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_chunks_doc
            ON graph_chunks (doc_id)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                src       INTEGER NOT NULL REFERENCES graph_entities(id),
                dst       INTEGER NOT NULL REFERENCES graph_entities(id),
                relation  TEXT NOT NULL DEFAULT 'related_to',
                doc_id    TEXT NOT NULL DEFAULT '',
                chunk_idx INTEGER NOT NULL DEFAULT -1,
                weight    REAL NOT NULL DEFAULT 1.0,
                UNIQUE (src, dst, relation, doc_id, chunk_idx)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_src ON graph_edges (src)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_edges_dst ON graph_edges (dst)
        """)
        conn.commit()
    logger.info("Graph store initialized at {}", GRAPH_DB_PATH)


def normalize_entity(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def extract_entities_rulebased(text: str, max_entities: int = 20) -> List[Tuple[str, str]]:
    """Trích thực thể không cần LLM: mã nội bộ (CODE) + cụm viết hoa (TERM)."""
    found: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    for m in _CODE_RE.finditer(text or ""):
        name = m.group(0).strip(".,;:()")
        key = normalize_entity(name)
        if len(name) >= 3 and key not in seen:
            seen.add(key)
            found.append((name, "CODE"))

    for m in _TERM_RE.finditer(text or ""):
        name = re.sub(r"\s+", " ", m.group(0)).strip(".,;:()")
        key = normalize_entity(name)
        if (
            len(name.split()) >= 2
            and len(name) >= 6
            and key not in seen
            and key not in _STOP_ENTITIES
        ):
            seen.add(key)
            found.append((name, "TERM"))
        if len(found) >= max_entities:
            break

    return found[:max_entities]


def extract_triples_llm(text: str, max_triples: int = 12) -> List[Tuple[str, str, str]]:
    """Trích (subject, relation, object) bằng chính LLM local. Trả [] khi lỗi."""
    try:
        from backend.generator import _run_chat
        from backend.config import REWRITE_MAX_NEW_TOKENS
    except Exception as e:
        logger.debug("LLM triple extraction unavailable: {}", e)
        return []

    system_msg = (
        "Trích các quan hệ thực thể từ văn bản hành chính tiếng Việt. "
        "Mỗi dòng đúng định dạng: SUBJECT | RELATION | OBJECT. "
        "Tối đa 12 dòng, không giải thích, không thêm gì khác."
    )
    try:
        raw = _run_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": (text or "")[:1500]},
            ],
            max_new_tokens=REWRITE_MAX_NEW_TOKENS * 6,
        )
    except Exception as e:
        logger.warning("LLM triple extraction failed: {}", e)
        return []

    triples = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts) and all(len(p) <= 80 for p in parts):
            triples.append((parts[0], parts[1], parts[2]))
        if len(triples) >= max_triples:
            break
    return triples


def _upsert_entity(conn: sqlite3.Connection, name: str, etype: str) -> Optional[int]:
    key = normalize_entity(name)
    if not key:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO graph_entities (name, name_cf, etype) VALUES (?, ?, ?)",
        (name.strip(), key, etype),
    )
    row = conn.execute(
        "SELECT id FROM graph_entities WHERE name_cf = ?", (key,)
    ).fetchone()
    return int(row["id"]) if row else None


def build_from_chunks(
    doc_id: str,
    chunks: List[Dict],
    use_llm: bool = False,
    source_name: str = "",
) -> Dict[str, int]:
    """Index chunk vào KG: node thực thể + edge đồng xuất hiện (+ triple LLM nếu bật).

    Không bao giờ raise — lỗi chỉ được log để ingest chính không bị ảnh hưởng.
    """
    stats = {"entities": 0, "edges": 0, "chunks": 0}
    try:
        init_graph_db()
        conn = _get_conn()
        with _conn_lock:
            for i, chunk in enumerate(chunks):
                text = (chunk or {}).get("text", "") or ""
                if not text.strip():
                    continue
                meta = (chunk or {}).get("metadata", {}) or {}
                chunk_key = f"{doc_id}_chunk_{i}"
                conn.execute(
                    """INSERT OR REPLACE INTO graph_chunks
                       (chunk_key, doc_id, chunk_idx, source, page, snippet)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_key,
                        doc_id,
                        i,
                        str(meta.get("source", source_name or "")),
                        meta.get("page"),
                        text[:600],
                    ),
                )
                stats["chunks"] += 1

                entity_ids: List[int] = []
                for name, etype in extract_entities_rulebased(text):
                    eid = _upsert_entity(conn, name, etype)
                    if eid is not None:
                        entity_ids.append(eid)
                stats["entities"] += len(entity_ids)

                # Edge đồng xuất hiện: mọi cặp thực thể trong cùng chunk liên quan nhau.
                for a in range(len(entity_ids)):
                    for b in range(a + 1, len(entity_ids)):
                        src, dst = sorted((entity_ids[a], entity_ids[b]))
                        conn.execute(
                            """INSERT OR IGNORE INTO graph_edges
                               (src, dst, relation, doc_id, chunk_idx, weight)
                               VALUES (?, ?, 'related_to', ?, ?, 1.0)""",
                            (src, dst, doc_id, i),
                        )
                        stats["edges"] += 1

                if use_llm:
                    for s, r, o in extract_triples_llm(text):
                        sid = _upsert_entity(conn, s, "TERM")
                        oid = _upsert_entity(conn, o, "TERM")
                        if sid is None or oid is None:
                            continue
                        src, dst = sorted((sid, oid))
                        conn.execute(
                            """INSERT OR IGNORE INTO graph_edges
                               (src, dst, relation, doc_id, chunk_idx, weight)
                               VALUES (?, ?, ?, ?, ?, 2.0)""",
                            (src, dst, r[:60], doc_id, i),
                        )
                        stats["edges"] += 1
            conn.commit()
        logger.info("[Graph] indexed doc {}: {} chunks, {} entities, {} edges.",
                    doc_id, stats["chunks"], stats["entities"], stats["edges"])
    except Exception as e:
        logger.warning("[Graph] build failed for doc {}: {}", doc_id, e)
    return stats


def is_global_query(query: str) -> bool:
    """Query tổng hợp/so sánh đa tài liệu -> nên bật graph expansion."""
    return bool(_GLOBAL_RE.search(query or ""))


def _seed_entity_ids(query: str, conn: sqlite3.Connection, limit: int = 8) -> List[int]:
    """Tìm entity có tên xuất hiện trong query (substring trên dạng chuẩn hóa)."""
    q = normalize_entity(query or "")
    if not q:
        return []
    rows = conn.execute("SELECT id, name_cf FROM graph_entities").fetchall()
    matched = [(int(r["id"]), r["name_cf"] or "") for r in rows if r["name_cf"] and r["name_cf"] in q]
    # Entity tên dài khớp trước (cụ thể hơn).
    matched.sort(key=lambda item: -len(item[1]))
    return [eid for eid, _name in matched[:limit]]


def expand_with_graph(
    query: str,
    results: List[Dict],
    max_extra: int = 3,
    similarity_penalty: float = 5.0,
) -> List[Dict]:
    """Mở rộng kết quả bằng 1-hop graph: chunk chứa entity kề với seed.

    - Seed = entity khớp trong query + entity của top-2 kết quả vector.
    - Không bao giờ raise; trả về list mới (không sửa list gốc).
    """
    try:
        init_graph_db()
        conn = _get_conn()
        existing_keys: Set[str] = set()
        for r in results or []:
            meta = (r or {}).get("metadata", {}) or {}
            if meta.get("doc_id") is not None and meta.get("chunk_index") is not None:
                try:
                    existing_keys.add(f"{meta.get('doc_id')}_chunk_{int(meta.get('chunk_index'))}")
                except (TypeError, ValueError):
                    pass

        seed_ids: Set[int] = set(_seed_entity_ids(query, conn))
        # Entity của top kết quả vector cũng làm seed (lan sang chunk liên quan).
        for r in (results or [])[:2]:
            for name, _etype in extract_entities_rulebased((r or {}).get("match_text", (r or {}).get("text", ""))):
                row = conn.execute(
                    "SELECT id FROM graph_entities WHERE name_cf = ?",
                    (normalize_entity(name),),
                ).fetchone()
                if row:
                    seed_ids.add(int(row["id"]))

        if not seed_ids:
            return list(results or [])

        placeholders = ",".join("?" for _ in seed_ids)
        neighbor_rows = conn.execute(
            f"""SELECT DISTINCT e.dst AS nid FROM graph_edges e WHERE e.src IN ({placeholders})
                UNION
                SELECT DISTINCT e.src AS nid FROM graph_edges e WHERE e.dst IN ({placeholders})""",
            (*seed_ids, *seed_ids),
        ).fetchall()
        neighbor_ids = {int(r["nid"]) for r in neighbor_rows} - seed_ids
        if not neighbor_ids:
            return list(results or [])

        # Chunk chứa neighbor entity, ưu tiên edge weight cao.
        ph2 = ",".join("?" for _ in neighbor_ids)
        chunk_rows = conn.execute(
            f"""SELECT gc.chunk_key, gc.doc_id, gc.chunk_idx, gc.source, gc.page, gc.snippet,
                       MAX(e.weight) AS w
                FROM graph_edges e
                JOIN graph_chunks gc ON (
                    gc.doc_id = e.doc_id AND gc.chunk_idx = e.chunk_idx
                )
                WHERE (e.src IN ({ph2}) OR e.dst IN ({ph2}))
                GROUP BY gc.chunk_key
                ORDER BY w DESC""",
            (*neighbor_ids, *neighbor_ids),
        ).fetchall()

        base_sim = float((results or [{}])[0].get("similarity", 30.0) or 30.0)
        expanded = list(results or [])
        added = 0
        for row in chunk_rows:
            if added >= max_extra:
                break
            if row["chunk_key"] in existing_keys:
                continue
            sim = max(0.0, round(base_sim - similarity_penalty, 1))
            expanded.append({
                "text": row["snippet"],
                "match_text": row["snippet"],
                "metadata": {
                    "source": row["source"] or "unknown",
                    "doc_id": row["doc_id"],
                    "chunk_index": row["chunk_idx"],
                    "page": row["page"],
                    "graph_expanded": True,
                },
                "similarity": sim,
            })
            existing_keys.add(row["chunk_key"])
            added += 1

        if added:
            logger.info("[Graph] expanded {} extra chunk(s) for query.", added)
        return expanded
    except Exception as e:
        logger.warning("[Graph] expansion failed: {}", e)
        return list(results or [])


def delete_doc_graph(doc_id: str) -> int:
    """Xóa toàn bộ node/edge/chunk của 1 document. Trả về số edge đã xóa."""
    try:
        init_graph_db()
        conn = _get_conn()
        with _conn_lock:
            cur = conn.execute("DELETE FROM graph_edges WHERE doc_id = ?", (doc_id,))
            removed = cur.rowcount or 0
            conn.execute("DELETE FROM graph_chunks WHERE doc_id = ?", (doc_id,))
            # Dọn entity mồ côi
            conn.execute("""
                DELETE FROM graph_entities WHERE id NOT IN
                (SELECT src FROM graph_edges UNION SELECT dst FROM graph_edges)
            """)
            conn.commit()
            return removed
    except Exception as e:
        logger.warning("[Graph] delete failed for doc {}: {}", doc_id, e)
        return 0


def graph_stats() -> Dict:
    try:
        init_graph_db()
        conn = _get_conn()
        n_ent = conn.execute("SELECT COUNT(*) AS c FROM graph_entities").fetchone()["c"]
        n_edge = conn.execute("SELECT COUNT(*) AS c FROM graph_edges").fetchone()["c"]
        n_chunk = conn.execute("SELECT COUNT(*) AS c FROM graph_chunks").fetchone()["c"]
        return {
            "entities": int(n_ent or 0),
            "edges": int(n_edge or 0),
            "chunks": int(n_chunk or 0),
            "db": GRAPH_DB_PATH,
        }
    except Exception as e:
        logger.warning("[Graph] stats failed: {}", e)
        return {"entities": 0, "edges": 0, "chunks": 0, "db": GRAPH_DB_PATH}
