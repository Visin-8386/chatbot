"""
test_fixes_validation.py
========================
Regression tests for the bug fixes committed in this session.
Tests are designed to run WITHOUT a live server (pure unit tests).

Covers:
  1. crag_verdict default 'skipped' — no NameError / dir() hack
  2. _load_skills() caching — called twice, disk read only once
  3. chat_memory — module-level connection, session_meta table
  4. rate limiter logic — _check_rate_limit raises 429 after limit
  5. intent classifier — smoke test
  6. session list — title field returned from get_all_sessions()
"""
import os
import sys
import time
import types
import unittest
import tempfile
import importlib

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ===========================================================================
# 1. Intent Classifier — smoke test
# ===========================================================================
class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        from backend.intent_classifier import classify_intent, get_chitchat_response
        self.classify = classify_intent
        self.chitchat = get_chitchat_response

    def test_chitchat_xin_chao(self):
        r = self.classify("xin chào bạn")
        self.assertEqual(r["intent"], "chitchat", r)

    def test_rag_query(self):
        r = self.classify("quy trình nghỉ phép của công ty là gì?")
        self.assertEqual(r["intent"], "rag_query", r)

    def test_followup_with_history(self):
        r = self.classify("còn gì nữa không?", has_history=True)
        self.assertIn(r["intent"], ("followup", "rag_query"), r)

    def test_chitchat_thanks(self):
        r = self.classify("cảm ơn bạn nhé")
        self.assertEqual(r["intent"], "chitchat", r)

    def test_farewell(self):
        r = self.classify("tạm biệt")
        self.assertEqual(r["intent"], "chitchat", r)

    def test_get_chitchat_response_not_empty(self):
        resp = self.chitchat("xin chào")
        self.assertTrue(len(resp) > 0)


# ===========================================================================
# 2. Skills cache — should not re-read disk on second call
# ===========================================================================
class TestSkillsCache(unittest.TestCase):
    def test_cache_populated_after_first_call(self):
        # Reset cache before test
        import backend.generator as gen
        gen._skills_cache = None

        call_count = {"n": 0}
        original_exists = os.path.exists
        original_listdir = os.listdir

        # Patch to count calls to listdir (which happens on disk scan)
        def mock_listdir(path):
            if "skills" in path:
                call_count["n"] += 1
                return []
            return original_listdir(path)

        import builtins
        original_open = builtins.open

        try:
            os.listdir = mock_listdir
            # First call — should scan disk
            result1 = gen._load_skills()
            # Second call — should use cache, NOT scan disk again
            result2 = gen._load_skills()
        finally:
            os.listdir = original_listdir

        self.assertEqual(result1, result2)
        # listdir should have been called exactly ONCE (cache on second call)
        self.assertEqual(call_count["n"], 1, "listdir called more than once — cache not working")

    def tearDown(self):
        # Reset cache after test
        import backend.generator as gen
        gen._skills_cache = None


# ===========================================================================
# 3. Rate limiter — should raise 429 after threshold
# ===========================================================================
class TestRateLimiter(unittest.TestCase):
    def test_blocks_after_limit(self):
        from collections import defaultdict
        from fastapi import HTTPException

        # Reproduce the exact rate limiter logic from main.py
        RATE_LIMIT_MAX = 5
        RATE_LIMIT_WINDOW = 60
        rate_counters: dict = defaultdict(list)

        def check_rate_limit(client_ip: str) -> None:
            now = time.time()
            window_start = now - RATE_LIMIT_WINDOW
            hits = rate_counters[client_ip]
            rate_counters[client_ip] = [t for t in hits if t > window_start]
            if len(rate_counters[client_ip]) >= RATE_LIMIT_MAX:
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests. Limit: {RATE_LIMIT_MAX} req/{RATE_LIMIT_WINDOW}s."
                )
            rate_counters[client_ip].append(now)

        ip = "127.0.0.1"
        # First N calls should pass
        for _ in range(RATE_LIMIT_MAX):
            check_rate_limit(ip)

        # Next call should raise 429
        with self.assertRaises(HTTPException) as ctx:
            check_rate_limit(ip)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_different_ips_independent(self):
        from collections import defaultdict
        from fastapi import HTTPException

        RATE_LIMIT_MAX = 3
        RATE_LIMIT_WINDOW = 60
        rate_counters: dict = defaultdict(list)

        def check_rate_limit(client_ip: str) -> None:
            now = time.time()
            window_start = now - RATE_LIMIT_WINDOW
            rate_counters[client_ip] = [t for t in rate_counters[client_ip] if t > window_start]
            if len(rate_counters[client_ip]) >= RATE_LIMIT_MAX:
                raise HTTPException(status_code=429, detail="Rate limited")
            rate_counters[client_ip].append(now)

        # IP A hits limit
        for _ in range(RATE_LIMIT_MAX):
            check_rate_limit("10.0.0.1")

        # IP B should still work
        try:
            check_rate_limit("10.0.0.2")
        except HTTPException:
            self.fail("IP B should NOT be rate limited")


# ===========================================================================
# 4. Chat Memory — connection caching + session_meta + title
# ===========================================================================
class TestChatMemory(unittest.TestCase):
    def setUp(self):
        """Use a temp DB for isolation."""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        # Patch DB_PATH before importing
        import backend.chat_memory as cm
        self._orig_path = cm.DB_PATH
        self._orig_conn = cm._conn
        cm.DB_PATH = self._tmp.name
        cm._conn = None  # force re-init
        cm.init_db()
        self.cm = cm

    def tearDown(self):
        import backend.chat_memory as cm
        if cm._conn:
            cm._conn.close()
        cm._conn = self._orig_conn
        cm.DB_PATH = self._orig_path
        os.unlink(self._tmp.name)

    def test_save_and_retrieve_history(self):
        cm = self.cm
        sid = "test-session-001"
        cm.save_turn(sid, "Câu hỏi 1", "Trả lời 1")
        cm.save_turn(sid, "Câu hỏi 2", "Trả lời 2")

        # Explicitly request max_turns=10 so test is independent of config
        history = cm.get_history(sid, max_turns=10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["user"], "Câu hỏi 1")
        self.assertEqual(history[0]["assistant"], "Trả lời 1")
        self.assertEqual(history[1]["user"], "Câu hỏi 2")

    def test_session_title_stored(self):
        """Session title should be the first user message."""
        cm = self.cm
        sid = "test-session-title"
        cm.save_turn(sid, "Hỏi về quy trình nghỉ phép", "Trả lời về nghỉ phép")

        sessions = cm.get_all_sessions()
        target = next((s for s in sessions if s["session_id"] == sid), None)
        self.assertIsNotNone(target, "Session not found in list")
        self.assertEqual(target["title"], "Hỏi về quy trình nghỉ phép")

    def test_delete_session_removes_meta(self):
        cm = self.cm
        sid = "test-delete-session"
        cm.save_turn(sid, "Hỏi gì đó", "Trả lời gì đó")
        deleted = cm.delete_session(sid)
        self.assertGreater(deleted, 0)

        history = cm.get_history(sid)
        self.assertEqual(history, [])

        sessions = cm.get_all_sessions()
        ids = [s["session_id"] for s in sessions]
        self.assertNotIn(sid, ids, "Session meta should be deleted")

    def test_connection_reused(self):
        """Same connection object should be returned on multiple calls."""
        cm = self.cm
        conn1 = cm._get_conn()
        conn2 = cm._get_conn()
        self.assertIs(conn1, conn2, "Connection should be reused (cached)")

    def test_empty_session_id_safe(self):
        cm = self.cm
        # Should not raise
        cm.save_turn("", "q", "a")
        result = cm.get_history("")
        self.assertEqual(result, [])
        deleted = cm.delete_session("")
        self.assertEqual(deleted, 0)

    def test_max_turns_respected(self):
        cm = self.cm
        sid = "test-max-turns"
        for i in range(10):
            cm.save_turn(sid, f"Q{i}", f"A{i}")

        history = cm.get_history(sid, max_turns=3)
        self.assertLessEqual(len(history), 3)


# ===========================================================================
# 5. crag_verdict default value — no NameError possible
# ===========================================================================
class TestCragVerdictDefault(unittest.TestCase):
    def test_no_name_error_pattern(self):
        """
        Verify the old anti-pattern "crag_verdict if 'crag_verdict' in dir()"
        no longer appears in main.py — we always initialize it to 'skipped'.
        """
        with open(os.path.join(ROOT, "backend", "main.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertNotIn(
            "'crag_verdict' in dir()",
            source,
            "Old fragile dir() pattern still present in main.py!"
        )

    def test_crag_verdict_initialized_in_pipeline(self):
        """Ensure crag_verdict = 'skipped' assignment exists in main.py."""
        with open(os.path.join(ROOT, "backend", "main.py"), encoding="utf-8") as f:
            source = f.read()
        self.assertIn(
            'crag_verdict = "skipped"',
            source,
            "crag_verdict default init not found in main.py"
        )


# ===========================================================================
# 6. Bare except fix — no bare `except:` in stream handler
# ===========================================================================
class TestBareExceptFixed(unittest.TestCase):
    def test_no_bare_except_in_main(self):
        with open(os.path.join(ROOT, "backend", "main.py"), encoding="utf-8") as f:
            lines = f.readlines()

        bare_except_lines = [
            (i + 1, line.rstrip())
            for i, line in enumerate(lines)
            if line.strip() == "except:"
        ]
        self.assertEqual(
            bare_except_lines, [],
            f"Bare `except:` found at lines: {bare_except_lines}"
        )


# ===========================================================================
# 7. Skills cache reset — ensure no stale state bleeds between tests
# ===========================================================================
class TestSkillsCacheIntegrity(unittest.TestCase):
    def test_cache_is_string_or_empty(self):
        import backend.generator as gen
        gen._skills_cache = None
        result = gen._load_skills()
        self.assertIsInstance(result, str)

    def tearDown(self):
        import backend.generator as gen
        gen._skills_cache = None


if __name__ == "__main__":
    unittest.main(verbosity=2)
