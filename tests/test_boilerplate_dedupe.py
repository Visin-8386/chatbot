# -*- coding: utf-8 -*-
"""Tests cho boilerplate dedupe trong retrieval (vector_store._dedupe_by_text)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vector_store import _dedupe_by_text


def _item(text, source, similarity=50.0):
    return {"text": text, "metadata": {"source": source}, "similarity": similarity}


BOILER = "1. Thực hiện quy định của pháp luật có liên quan đến công tác dân số.\n2. Tôn trọng quyền và lợi ích của cá nhân, gia đình."


def test_keeps_first_of_identical_text_across_docs():
    items = [
        _item(BOILER, "luat-a.txt", 74.3),
        _item(BOILER, "luat-b.txt", 74.3),
        _item(BOILER, "luat-c.txt", 74.3),
    ]
    out = _dedupe_by_text(items)
    assert len(out) == 1
    assert out[0]["metadata"]["source"] == "luat-a.txt"


def test_whitespace_and_case_variants_are_duplicates():
    items = [
        _item("Điều 113. NGHỈ hằng năm.", "a.txt"),
        _item("  điều   113. nghỉ hằng năm.  ", "b.txt"),
    ]
    assert len(_dedupe_by_text(items)) == 1


def test_distinct_texts_kept_with_order():
    items = [
        _item("Điều 113 quy định nghỉ hằng năm 12 ngày.", "a.txt"),
        _item("Điều 107 quy định làm thêm giờ.", "b.txt"),
        _item("Điều 113 quy định nghỉ hằng năm 12 ngày.", "c.txt"),
        _item("Điều 169 quy định tuổi nghỉ hưu.", "d.txt"),
    ]
    out = _dedupe_by_text(items)
    assert [i["metadata"]["source"] for i in out] == ["a.txt", "b.txt", "d.txt"]


def test_empty_list_safe():
    assert _dedupe_by_text([]) == []
