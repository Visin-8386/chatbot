"""
Performance & Integration Benchmark — Cần server đang chạy.
Đo chi tiết: TTFB (time-to-first-byte stream), total latency từng bước.

Chạy: python tests/benchmark_latency.py
Hoặc khi server đang chạy: pytest tests/benchmark_latency.py -v -s
"""
import os, sys, time, json, statistics, uuid
import requests
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# Ngưỡng latency chấp nhận được (ms)
SLA = {
    "health_ms":        200,    # /api/health phải trả lời < 200ms
    "upload_ms":       60_000,  # upload + process PDF < 60s
    "search_total_ms": 60_000,  # toàn bộ /api/search (LLM included) < 60s
    "retrieve_ms":      3_000,  # chỉ bước retrieve < 3s
    "rerank_ms":        5_000,  # chỉ bước rerank < 5s
    "ttfb_stream_ms":  30_000,  # time-to-first-byte stream < 30s
}

QUERIES = [
    "Quy trình nghỉ phép hàng năm như thế nào?",
    "Mã thiết bị MAINT-2024-SERVER là gì?",
    "Hướng dẫn phân mạch tài liệu kỹ thuật?",
]


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
def server_is_up(timeout: int = 5) -> bool:
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=timeout)
        return r.ok
    except Exception:
        return False


def wait_for_server(timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_is_up():
            return
        time.sleep(2)
    pytest.skip("Server không chạy — bỏ qua integration tests")


@pytest.fixture(scope="module", autouse=True)
def require_server():
    wait_for_server()


# ════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ════════════════════════════════════════════════════════════════
class TestHealth:
    def test_health_returns_ok(self):
        t0 = time.perf_counter()
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        print(f"\n  [Health] {elapsed:.0f}ms — llm_loaded={body.get('llm_loaded')}")

    def test_health_latency_under_sla(self):
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            requests.get(f"{BASE_URL}/api/health", timeout=5)
            times.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(times)
        print(f"\n  [Health p50] avg={avg:.0f}ms max={max(times):.0f}ms")
        assert avg < SLA["health_ms"], f"Health avg {avg:.0f}ms vượt SLA {SLA['health_ms']}ms"


# ════════════════════════════════════════════════════════════════
# 2. UPLOAD
# ════════════════════════════════════════════════════════════════
_uploaded_doc_id = None

class TestUpload:
    def test_upload_txt_file(self):
        global _uploaded_doc_id
        # Tạo file txt tạm
        content = (
            "Quy trình nghỉ phép:\n"
            "Nhân viên được nghỉ phép 12 ngày/năm theo điều 113 Bộ luật Lao động.\n"
            "Đơn xin nghỉ phép phải nộp trước 3 ngày làm việc.\n"
            "Trưởng bộ phận phê duyệt trong vòng 24 giờ.\n\n"
            "Bảng lương 2024: Vùng I: 4.680.000 đồng. Vùng II: 4.160.000 đồng.\n"
            "Mã thiết bị MAINT-2024-SERVER được dùng cho máy chủ tại DC-01.\n" * 10
        )
        tmp = os.path.join(os.path.dirname(__file__), "_bench_test.txt")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)

        t0 = time.perf_counter()
        with open(tmp, "rb") as f:
            r = requests.post(f"{BASE_URL}/api/upload", files={"file": ("bench_test.txt", f)}, timeout=120)
        elapsed = (time.perf_counter() - t0) * 1000
        os.remove(tmp)

        assert r.status_code == 200, f"Upload failed: {r.text}"
        body = r.json()
        _uploaded_doc_id = body.get("doc_id")
        print(f"\n  [Upload] {elapsed:.0f}ms — {body.get('chunks')} chunks, doc_id={_uploaded_doc_id}")
        assert body.get("chunks", 0) >= 1
        assert elapsed < SLA["upload_ms"]

    def test_upload_unsupported_extension_rejected(self):
        import io
        r = requests.post(
            f"{BASE_URL}/api/upload",
            files={"file": ("test.exe", io.BytesIO(b"fake data"))},
            timeout=10
        )
        assert r.status_code == 400

    def test_stats_reflects_upload(self):
        r = requests.get(f"{BASE_URL}/api/stats", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert body["total_chunks"] >= 1
        print(f"\n  [Stats] {body['total_documents']} docs, {body['total_chunks']} chunks")


# ════════════════════════════════════════════════════════════════
# 3. SEARCH LATENCY — Pipeline từng bước
# ════════════════════════════════════════════════════════════════
class TestSearchLatency:
    def _search(self, query: str, session_id: str = None) -> dict:
        payload = {"query": query, "top_k": 3}
        if session_id:
            payload["session_id"] = session_id
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/api/search", json=payload, timeout=120)
        wall_ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"Search failed: {r.text}"
        body = r.json()
        body["_wall_ms"] = wall_ms
        return body

    def test_search_returns_valid_structure(self):
        body = self._search(QUERIES[0])
        for field in ["ai_answer", "ai_sources", "results", "timings_ms"]:
            assert field in body, f"Thiếu field: {field}"

    def test_search_has_answer(self):
        body = self._search(QUERIES[0])
        assert body.get("ai_answer"), "Không có câu trả lời"
        print(f"\n  [Answer preview] {body['ai_answer'][:200]}")

    def test_retrieve_step_latency(self):
        body = self._search(QUERIES[0])
        retrieve_ms = body["timings_ms"].get("retrieve", 999999)
        print(f"\n  [Retrieve] {retrieve_ms:.0f}ms")
        assert retrieve_ms < SLA["retrieve_ms"], f"Retrieve {retrieve_ms:.0f}ms > SLA {SLA['retrieve_ms']}ms"

    def test_rerank_step_latency(self):
        body = self._search(QUERIES[0])
        rerank_ms = body["timings_ms"].get("rerank", 999999)
        print(f"\n  [Rerank] {rerank_ms:.0f}ms")
        assert rerank_ms < SLA["rerank_ms"], f"Rerank {rerank_ms:.0f}ms > SLA {SLA['rerank_ms']}ms"

    def test_total_search_latency(self):
        body = self._search(QUERIES[0])
        total_ms = body["_wall_ms"]
        print(f"\n  [Total] {total_ms:.0f}ms")
        assert total_ms < SLA["search_total_ms"], f"Total {total_ms:.0f}ms > SLA {SLA['search_total_ms']}ms"

    def test_all_queries_latency_summary(self):
        """Chạy tất cả queries, in bảng latency chi tiết."""
        print("\n")
        print(f"  {'Query':<45} {'wall':>7} {'retrieve':>9} {'rerank':>8} {'crag':>6} {'generate':>10}")
        print(f"  {'-'*45} {'-'*7} {'-'*9} {'-'*8} {'-'*6} {'-'*10}")
        wall_times = []
        for q in QUERIES:
            body = self._search(q)
            t = body["timings_ms"]
            wall = body["_wall_ms"]
            wall_times.append(wall)
            print(
                f"  {q[:45]:<45} {wall:>6.0f}ms"
                f" {t.get('retrieve',0):>8.0f}ms"
                f" {t.get('rerank',0):>7.0f}ms"
                f" {t.get('crag',0):>5.0f}ms"
                f" {t.get('generate',0):>9.0f}ms"
            )
        print(f"\n  Avg wall: {statistics.mean(wall_times):.0f}ms | Max: {max(wall_times):.0f}ms")

    def test_keyword_search_finds_exact_match(self):
        body = self._search("MAINT-2024-SERVER")
        texts = [r["text"] for r in body.get("results", [])]
        assert any("MAINT-2024-SERVER" in t for t in texts), "Keyword search không tìm thấy mã chính xác"


# ════════════════════════════════════════════════════════════════
# 4. STREAMING — TTFB (Time To First Byte)
# ════════════════════════════════════════════════════════════════
class TestStreaming:
    def test_stream_endpoint_returns_200(self):
        r = requests.post(
            f"{BASE_URL}/api/search/stream",
            json={"query": QUERIES[0], "top_k": 3},
            timeout=120,
            stream=True
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_ttfb_stream(self):
        """Đo thời gian từ khi gửi request đến khi nhận được byte đầu tiên."""
        t0 = time.perf_counter()
        first_byte_ms = None
        full_ms = None
        tokens = []

        with requests.post(
            f"{BASE_URL}/api/search/stream",
            json={"query": QUERIES[0], "top_k": 3},
            timeout=120,
            stream=True
        ) as r:
            assert r.status_code == 200
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    elapsed = (time.perf_counter() - t0) * 1000
                    if first_byte_ms is None:
                        first_byte_ms = elapsed
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get("type") == "token":
                            tokens.append(chunk.get("token", ""))
                        elif chunk.get("type") == "done":
                            full_ms = elapsed
                            break
                    except Exception:
                        pass

        print(f"\n  [Stream TTFB] {first_byte_ms:.0f}ms | Full: {full_ms:.0f}ms | Tokens: {len(tokens)}")
        assert first_byte_ms is not None, "Không nhận được byte nào từ stream"
        assert first_byte_ms < SLA["ttfb_stream_ms"], f"TTFB {first_byte_ms:.0f}ms > SLA {SLA['ttfb_stream_ms']}ms"

    def test_stream_produces_tokens(self):
        tokens = []
        with requests.post(
            f"{BASE_URL}/api/search/stream",
            json={"query": QUERIES[0], "top_k": 3},
            timeout=120,
            stream=True
        ) as r:
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        if chunk.get("type") == "token":
                            tokens.append(chunk["token"])
                        elif chunk.get("type") == "done":
                            break
                    except Exception:
                        pass
        assert len(tokens) > 0, "Stream không trả về token nào"
        full_text = "".join(tokens)
        assert len(full_text) > 10, "Câu trả lời stream quá ngắn"


# ════════════════════════════════════════════════════════════════
# 5. PERSISTENT MEMORY qua API
# ════════════════════════════════════════════════════════════════
class TestSessionAPI:
    def test_session_history_saved_after_search(self):
        sid = uuid.uuid4().hex
        requests.post(f"{BASE_URL}/api/search", json={"query": QUERIES[0], "top_k": 3, "session_id": sid}, timeout=120)
        r = requests.get(f"{BASE_URL}/api/sessions/{sid}/history", timeout=10)
        assert r.status_code == 200
        body = r.json()
        print(f"\n  [Session history] {body['turns']} turns for session {sid[:8]}")
        assert body["turns"] >= 1

    def test_delete_session(self):
        sid = uuid.uuid4().hex
        requests.post(f"{BASE_URL}/api/search", json={"query": QUERIES[0], "top_k": 3, "session_id": sid}, timeout=120)
        r = requests.delete(f"{BASE_URL}/api/sessions/{sid}", timeout=10)
        assert r.status_code == 200
        assert r.json()["success"] is True
        # Verify deleted
        r2 = requests.get(f"{BASE_URL}/api/sessions/{sid}/history", timeout=10)
        assert r2.json()["turns"] == 0

    def test_list_sessions(self):
        r = requests.get(f"{BASE_URL}/api/sessions", timeout=10)
        assert r.status_code == 200
        body = r.json()
        assert "sessions" in body and "total" in body


# ════════════════════════════════════════════════════════════════
# 6. CLEANUP — Xóa doc test đã upload
# ════════════════════════════════════════════════════════════════
class TestCleanup:
    def test_delete_uploaded_doc(self):
        if not _uploaded_doc_id:
            pytest.skip("Không có doc_id để xóa")
        r = requests.delete(f"{BASE_URL}/api/documents/{_uploaded_doc_id}", timeout=10)
        assert r.status_code == 200
        print(f"\n  [Cleanup] Deleted doc {_uploaded_doc_id}: {r.json()}")


# ════════════════════════════════════════════════════════════════
# CLI runner (không dùng pytest)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not server_is_up():
        print("❌ Server không chạy tại", BASE_URL)
        print("   Chạy server trước: uvicorn backend.main:app --reload")
        sys.exit(1)

    print("=" * 65)
    print(" LATENCY BENCHMARK — DocSearch RAG")
    print(f" Server: {BASE_URL}")
    print("=" * 65)

    # Health
    t0 = time.perf_counter()
    r = requests.get(f"{BASE_URL}/api/health", timeout=5)
    print(f"\n[Health] {(time.perf_counter()-t0)*1000:.0f}ms — {r.json()}")

    # Upload test doc
    content = (
        "Quy trình nghỉ phép:\nNhân viên được nghỉ 12 ngày/năm.\n"
        "Đơn phải nộp trước 3 ngày.\nMã thiết bị MAINT-2024-SERVER tại DC-01.\n"
    ) * 15
    tmp = "_bench_upload.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    t0 = time.perf_counter()
    with open(tmp, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": ("bench.txt", f)}, timeout=120)
    upload_ms = (time.perf_counter() - t0) * 1000
    os.remove(tmp)
    doc_id = r.json().get("doc_id") if r.ok else None
    print(f"[Upload] {upload_ms:.0f}ms — status={r.status_code} chunks={r.json().get('chunks') if r.ok else 'ERR'}")

    # Search benchmark
    print(f"\n{'Query':<50} {'wall':>7} {'retr':>7} {'rerank':>8} {'crag':>6} {'gen':>8}")
    print("-" * 90)
    wall_all = []
    for q in QUERIES:
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/api/search", json={"query": q, "top_k": 3}, timeout=120)
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_all.append(wall_ms)
        t = r.json().get("timings_ms", {}) if r.ok else {}
        status = "OK" if r.ok else "ERR"
        print(
            f"{status} {q[:48]:<48} {wall_ms:>6.0f}ms"
            f" {t.get('retrieve',0):>6.0f}ms"
            f" {t.get('rerank',0):>7.0f}ms"
            f" {t.get('crag',0):>5.0f}ms"
            f" {t.get('generate',0):>7.0f}ms"
        )

    print(f"\n  ► Avg wall: {statistics.mean(wall_all):.0f}ms | Max: {max(wall_all):.0f}ms")

    # TTFB stream
    print("\n[Stream TTFB test]")
    t0 = time.perf_counter()
    first_byte_ms = None
    done_ms = None
    token_count = 0
    with requests.post(f"{BASE_URL}/api/search/stream",
                       json={"query": QUERIES[0], "top_k": 3}, timeout=120, stream=True) as resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if line.startswith("data: "):
                elapsed = (time.perf_counter() - t0) * 1000
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("type") == "token":
                        if first_byte_ms is None:
                            first_byte_ms = elapsed
                        token_count += 1
                    elif chunk.get("type") == "done":
                        done_ms = elapsed
                        break
                except Exception:
                    pass
    print(f"  TTFB: {first_byte_ms:.0f}ms | Full: {done_ms:.0f}ms | Tokens: {token_count}")

    # Cleanup
    if doc_id:
        requests.delete(f"{BASE_URL}/api/documents/{doc_id}", timeout=10)
        print(f"\n[Cleanup] Deleted doc {doc_id}")

    print("\n" + "=" * 65)
    print(" BENCHMARK HOÀN TẤT")
    print("=" * 65)
