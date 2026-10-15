"""Trace logger tests: read order, filter, bad-line tolerance (no server needed)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.trace_logger as tl


def _seed(path, n=5, bad=True):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"query": f"cau hoi {i}", "i": i}, ensure_ascii=False) + "\n")
        if bad:
            f.write("dong hong {json\n")


def test_read_newest_first_and_pagination(tmp_path, monkeypatch):
    p = str(tmp_path / "traces.jsonl")
    monkeypatch.setattr(tl, "TRACE_LOG_FILE", p)
    _seed(p, n=5)
    first = tl.read_traces(limit=2, offset=0)
    assert first["total"] == 5
    assert [t["i"] for t in first["traces"]] == [4, 3]
    second = tl.read_traces(limit=2, offset=2)
    assert [t["i"] for t in second["traces"]] == [2, 1]


def test_read_filter_and_bad_lines(tmp_path, monkeypatch):
    p = str(tmp_path / "traces.jsonl")
    monkeypatch.setattr(tl, "TRACE_LOG_FILE", p)
    _seed(p, n=5, bad=True)
    out = tl.read_traces(limit=50, query="cau hoi 1")
    assert out["total"] == 1 and out["traces"][0]["i"] == 1
    all_out = tl.read_traces(limit=50)
    assert all_out["total"] == 5  # dòng hỏng bị bỏ qua


def test_read_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "TRACE_LOG_FILE", str(tmp_path / "nope.jsonl"))
    assert tl.read_traces() == {"traces": [], "total": 0}


def test_langfuse_hook_disabled_by_default():
    from backend.langfuse_hook import forward_trace
    assert forward_trace({"query": "x"}) is False  # LANGFUSE_ENABLED mặc định 0
