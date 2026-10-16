# -*- coding: utf-8 -*-
"""Rewrite cleaner tests: pure, no LLM needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.generator import _clean_rewritten_query as clean


def test_strips_markdown_label_prefix():
    out = clean('**Truy vấn tìm kiếm:** "Quy trình xin nghỉ phép năm"')
    assert out == "Quy trình xin nghỉ phép năm", out


def test_strips_plain_label_prefix():
    out = clean("Truy vấn tìm kiếm: Các quy định về nghỉ phép")
    assert out == "Các quy định về nghỉ phép", out


def test_strips_quotes_and_bold():
    assert clean('"Câu truy vấn thuần"') == "Câu truy vấn thuần"
    assert clean("**Câu hỏi**") == "Câu hỏi"


def test_keeps_legit_colon_query():
    out = clean("Báo cáo Q3: doanh thu theo vùng")
    assert out == "Báo cáo Q3: doanh thu theo vùng", out


def test_keeps_clean_query_untouched():
    out = clean("Các quy định bổ sung về chế độ nghỉ phép năm")
    assert out == "Các quy định bổ sung về chế độ nghỉ phép năm", out


def test_takes_rhs_of_arrow():
    out = clean("còn gì nữa k -> Các quy định bổ sung về nghỉ phép")
    assert out == "Các quy định bổ sung về nghỉ phép", out


def test_empty_safe():
    assert clean("") == ""
    assert clean("   ") == ""
