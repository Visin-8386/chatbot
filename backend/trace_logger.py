import json
import os
from datetime import datetime
from typing import List, Dict, Optional

TRACE_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "traces.jsonl")

def init_trace_dir():
    os.makedirs(os.path.dirname(TRACE_LOG_FILE), exist_ok=True)

def log_trace(trace_data: dict):
    """
    Lưu vết (trace) của toàn bộ request RAG để phục vụ Observability.
    Dữ liệu được lưu dạng JSON Lines (JSONL).
    """
    init_trace_dir()
    trace_data["timestamp"] = datetime.now().isoformat()
    try:
        with open(TRACE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_data, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Trace Logger Error: {e}")

    # Langfuse forward (opt-in, best-effort — xem backend/langfuse_hook.py)
    try:
        from backend.langfuse_hook import forward_trace
        forward_trace(dict(trace_data))
    except Exception:
        pass

def read_traces(limit: int = 50, offset: int = 0, query: Optional[str] = None) -> Dict:
    """Đọc traces mới nhất trước, lọc theo query substring, bỏ qua dòng hỏng.

    Returns:
        {"traces": [...], "total": int} — total là số dòng hợp lệ khớp filter.
    """
    limit = max(1, min(int(limit or 50), 500))
    offset = max(0, int(offset or 0))
    q = (query or "").strip().casefold()

    if not os.path.exists(TRACE_LOG_FILE):
        return {"traces": [], "total": 0}

    matched: List[Dict] = []
    try:
        with open(TRACE_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Trace Read Error: {e}")
        return {"traces": [], "total": 0}

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue  # bỏ qua dòng hỏng, không sập endpoint
        if not isinstance(item, dict):
            continue
        if q:
            haystack = json.dumps(item, ensure_ascii=False).casefold()
            if q not in haystack:
                continue
        matched.append(item)

    return {"traces": matched[offset:offset + limit], "total": len(matched)}
