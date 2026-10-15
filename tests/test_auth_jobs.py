"""Auth (RBAC-lite) + jobs tests: pure stdlib, no server/GPU needed."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth import is_read_allowed, is_write_allowed, parse_keys, describe_auth
from backend.jobs import JobManager


def test_open_system_allows_all():
    assert is_read_allowed(None, "", set(), set()) is True
    assert is_write_allowed(None, "", set(), set()) is True
    assert describe_auth("", set(), set()) == "open"


def test_single_key_backward_compat():
    assert is_read_allowed("s3cret", "s3cret", set(), set()) is True
    assert is_write_allowed("s3cret", "s3cret", set(), set()) is True
    assert is_read_allowed("wrong", "s3cret", set(), set()) is False
    assert is_write_allowed(None, "s3cret", set(), set()) is False


def test_rbac_scopes():
    read, write = parse_keys("r1, r2"), parse_keys("w1")
    # read-key: đọc được, ghi không
    assert is_read_allowed("r1", "", read, write) is True
    assert is_write_allowed("r1", "", read, write) is False
    # write-key: cả hai
    assert is_read_allowed("w1", "", read, write) is True
    assert is_write_allowed("w1", "", read, write) is True
    # lạ: không gì cả
    assert is_read_allowed("zzz", "", read, write) is False
    # API_KEY cũ vẫn full-quyền khi dùng chung RBAC
    assert is_write_allowed("admin", "admin", read, write) is True
    assert describe_auth("", read, write) == "rbac"


def test_parse_keys_trims_and_drops_empty():
    assert parse_keys(" a ,,b , ") == {"a", "b"}
    assert parse_keys("") == set()
    assert parse_keys(None) == set()


def test_job_success_and_get():
    mgr = JobManager(max_workers=1)
    jid = mgr.submit("ok", lambda: {"success": True})
    for _ in range(100):
        job = mgr.get(jid)
        if job and job["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    job = mgr.get(jid)
    assert job["status"] == "done", job
    assert job["result"] == {"success": True}
    assert job["finished_at"] is not None


def test_job_error_captured():
    mgr = JobManager(max_workers=1)

    def boom():
        raise ValueError("No text content")

    jid = mgr.submit("fail", boom)
    for _ in range(100):
        job = mgr.get(jid)
        if job and job["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    job = mgr.get(jid)
    assert job["status"] == "error", job
    assert "ValueError" in job["error"]


def test_job_missing_and_prune():
    mgr = JobManager(max_workers=1)
    assert mgr.get("nope") is None
    for i in range(5):
        mgr.submit(f"j{i}", lambda: {})
    time.sleep(0.3)
    removed = mgr.prune(keep=2)
    assert removed == 3
    assert len(mgr.list()) == 2
