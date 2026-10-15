"""
Langfuse Hook - Optional forwarding of RAG traces to self-hosted Langfuse.

Triết lý "lite": mặc định TẮT, không thêm dependency bắt buộc, không tốn RAM/VRAM.
Bật khi đã có Langfuse self-host (docker compose) bằng:
  LANGFUSE_ENABLED=1
  LANGFUSE_PUBLIC_KEY=...  LANGFUSE_SECRET_KEY=...  LANGFUSE_HOST=http://langfuse:3000
  pip install langfuse>=2.0

Mọi lỗi đều nuốt lặng (observability không bao giờ được làm sập request chính).
"""
import os


def _enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "0") == "1"


def forward_trace(trace_data: dict) -> bool:
    """Gửi 1 trace sang Langfuse. Trả True nếu gửi được, False trong mọi trường hợp còn lại."""
    if not _enabled():
        return False
    try:
        from langfuse import Langfuse  # optional dependency
    except Exception:
        return False

    try:
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
        query = str((trace_data or {}).get("query", ""))[:200]
        trace = client.trace(
            name="rag-search",
            input={"query": query},
            output={"answer": str((trace_data or {}).get("ai_answer", ""))[:2000]},
            metadata={
                "intent": (trace_data or {}).get("intent", ""),
                "crag_verdict": (trace_data or {}).get("crag_verdict", ""),
                "generation_mode": (trace_data or {}).get("generation_mode", ""),
                "quality_score": (trace_data or {}).get("quality_score"),
                "timings_ms": (trace_data or {}).get("timings_ms", {}),
            },
        )
        client.flush()
        return trace is not None
    except Exception:
        return False
