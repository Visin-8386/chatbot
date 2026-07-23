"""
test_app_direct.py — Test trực tiếp FastAPI App với TestClient (không cần uvicorn)
=============================================================================
"""
import os
import sys
import json
import time

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("=" * 60)
print("  DIRECT FASTAPI APP INTEGRATION TESTS (TestClient)")
print("=" * 60)

# 1. Health Check
print("\n[1/7] Testing GET /api/health...")
res = client.get("/api/health")
print("Status:", res.status_code)
print("Response:", res.json())
assert res.status_code == 200
assert res.json()["status"] == "ok"
print("--> PASS")

# 2. Chitchat Search
print("\n[2/7] Testing POST /api/search (Chitchat Intent)...")
res = client.post("/api/search", json={"query": "xin chào bạn", "top_k": 3})
print("Status:", res.status_code)
body = res.json()
print("Intent:", body.get("intent"))
print("AI Answer:", repr(body.get("ai_answer")))
print("CRAG Verdict:", body.get("crag_verdict"))
assert res.status_code == 200
assert body["intent"] == "chitchat"
assert body["crag_verdict"] == "skipped"
print("--> PASS")

# 3. Input Validation (Empty Query)
print("\n[3/7] Testing POST /api/search (Empty Query Validation)...")
res = client.post("/api/search", json={"query": "   ", "top_k": 3})
print("Status:", res.status_code)
print("Detail:", res.json())
assert res.status_code == 400
print("--> PASS")

# 4. RAG Search Pipeline Structure
print("\n[4/7] Testing POST /api/search (RAG Pipeline)...")
res = client.post("/api/search", json={"query": "quy trình xin nghỉ phép hàng năm như thế nào?", "top_k": 3})
print("Status:", res.status_code)
body = res.json()
print("Intent:", body.get("intent"))
print("Rewritten Query:", body.get("rewritten_query"))
print("CRAG Verdict:", body.get("crag_verdict"))
print("Timings (ms):", body.get("timings_ms"))
assert res.status_code == 200
assert "intent" in body
assert "timings_ms" in body
print("--> PASS")

# 5. Session Memory Integration
print("\n[5/7] Testing Session & Memory Persistence...")
test_session = f"session-direct-test-{int(time.time())}"

# Send turn 1 (chitchat query to guarantee save turn)
res1 = client.post("/api/search", json={"query": "xin chào trợ lý", "top_k": 3, "session_id": test_session})
print("Search response:", res1.json())
assert res1.status_code == 200

# Get history
res_hist = client.get(f"/api/sessions/{test_session}/history")
print("History response:", res_hist.json())
assert res_hist.status_code == 200
assert res_hist.json()["turns"] == 1, f"Expected 1 turn, got {res_hist.json()['turns']}"

# List sessions (check title)
res_sess = client.get("/api/sessions")
sessions = res_sess.json()["sessions"]
my_sess = next((s for s in sessions if s["session_id"] == test_session), None)
print("Session metadata:", my_sess)
assert my_sess is not None
assert my_sess["title"] == "xin chào trợ lý"

# Delete session
res_del = client.delete(f"/api/sessions/{test_session}")
assert res_del.status_code == 200
print("--> PASS")

# 6. Documents & Stats API
print("\n[6/7] Testing GET /api/documents & GET /api/stats...")
res_docs = client.get("/api/documents")
res_stats = client.get("/api/stats")
print("Documents count:", res_docs.json()["total"])
print("Stats:", res_stats.json())
assert res_docs.status_code == 200
assert res_stats.status_code == 200
print("--> PASS")

# 7. Frontend Serving
print("\n[7/7] Testing GET / (Frontend index.html)...")
res_index = client.get("/")
print("Status:", res_index.status_code)
assert res_index.status_code == 200
assert "<!DOCTYPE html" in res_index.text or "<html" in res_index.text
print("--> PASS")

print("\n" + "=" * 60)
print("  ALL DIRECT INTEGRATION TESTS PASSED PERFECTLY!")
print("=" * 60)
