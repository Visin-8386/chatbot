import json
import os
from datetime import datetime

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
