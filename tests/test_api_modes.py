import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
SUPPORTED_EXTS = {".pdf", ".docx", ".xlsx", ".txt"}
QUERIES = [
    "Quy trinh nghi phep nhu the nao?",
    "Quy uoc danh so thiet bi la gi?",
    "Huong dan phan mach trong tai lieu?",
]


def wait_for_server(timeout_sec: int = 120) -> None:
    deadline = time.time() + timeout_sec
    health_url = f"{BASE_URL}/api/health"
    while time.time() < deadline:
        try:
            res = requests.get(health_url, timeout=5)
            if res.ok:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Server did not become ready in time")


def upload_documents(data_dir: Path) -> list:
    uploaded = []
    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue

        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f)}
            res = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=600)
        if not res.ok:
            raise RuntimeError(f"Upload failed for {file_path.name}: {res.status_code} {res.text}")

        body = res.json()
        uploaded.append(
            {
                "filename": file_path.name,
                "doc_id": body.get("doc_id"),
                "chunks": body.get("chunks"),
            }
        )
    return uploaded


def run_search_tests() -> list:
    results = []
    for q in QUERIES:
        payload = {"query": q, "top_k": 3}
        start = time.time()
        res = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=600)
        elapsed_ms = round((time.time() - start) * 1000, 1)
        if not res.ok:
            raise RuntimeError(f"Search failed for '{q}': {res.status_code} {res.text}")

        body = res.json()
        results.append(
            {
                "query": q,
                "elapsed_ms": elapsed_ms,
                "needs_clarification": body.get("needs_clarification"),
                "generation_mode": body.get("generation_mode"),
                "self_check_status": body.get("self_check_status"),
                "quality_score": body.get("quality_score"),
                "timings_ms": body.get("timings_ms"),
                "answer_preview": (body.get("ai_answer") or "")[:220],
                "sources": body.get("ai_sources", []),
            }
        )
    return results


def get_stats() -> dict:
    res = requests.get(f"{BASE_URL}/api/stats", timeout=30)
    if not res.ok:
        raise RuntimeError(f"Stats request failed: {res.status_code} {res.text}")
    return res.json()


def main() -> None:
    mode_label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    data_dir = Path("d:/chatbot/test_data")

    wait_for_server()
    uploaded = upload_documents(data_dir)
    stats_after_upload = get_stats()
    search_results = run_search_tests()

    output = {
        "mode_label": mode_label,
        "uploaded_count": len(uploaded),
        "uploaded": uploaded,
        "stats": stats_after_upload,
        "search_results": search_results,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
