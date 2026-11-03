# -*- coding: utf-8 -*-
"""日期过滤器解析测试。"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from ai.query_filter import DateFilter, extract_date_filter


@pytest.mark.parametrize(
    "query, expected_raw",
    [
        # 全日期 YYYY-MM-DD
        ("2026-10-17 花了多少", "2026-10-17"),
        ("2026年10月17日多少钱", "2026年10月17日"),
        # 中文月日
        ("10月17号花了多少", "10月17号"),
        ("10月17日花多少", "10月17日"),
        # 相对日
        ("昨天花了多少", "昨天"),
        ("前天花多少", "前天"),
        ("今天花了多少", "今天"),
        # 短日期
        ("10-17 花了多少", "10-17"),
    ],
)
def test_date_eq_forms(query: str, expected_raw: str) -> None:
    """测试各种单日/相对日表达式的解析。"""
    result = extract_date_filter(query)
    assert result is not None
    assert result.date_eq is not None, f"should have date_eq for query={query!r}"
    assert result.start is None
    assert result.end is None
    assert result.raw_text == expected_raw


def test_cn_month_day_uses_current_year() -> None:
    """中文月日没写年份时应该用当前年。"""
    today = date.today()
    r = extract_date_filter("5月20号花了多少")
    assert r is not None
    assert r.date_eq == date(today.year, 5, 20)


def test_month_range() -> None:
    """整个月查询：'10 月' 应该返回 10 月 1 日到 10 月最后一天。"""
    today = date.today()
    r = extract_date_filter("10 月一共花了多少")
    assert r is not None
    assert r.start == date(today.year, 10, 1)
    assert r.end == date(today.year, 10, 31)
    assert r.date_eq is None


def test_month_12_range_wraps_year() -> None:
    """12 月范围：终点应该是 12 月 31 日。"""
    today = date.today()
    r = extract_date_filter("12 月花了多少")
    assert r is not None
    assert r.start == date(today.year, 12, 1)
    assert r.end == date(today.year, 12, 31)


def test_relative_dates_use_today() -> None:
    """相对日期基于当天。"""
    today = date.today()
    assert extract_date_filter("今天") is not None
    assert extract_date_filter("今天").date_eq == today
    assert extract_date_filter("昨天").date_eq == today - timedelta(days=1)
    assert extract_date_filter("前天").date_eq == today - timedelta(days=2)


def test_no_date_returns_none() -> None:
    """没有日期的查询应返回 None（不做过滤）。"""
    for q in ["猪肉", "打车花了多少", "买菜钱", "上周打车", "工资多少"]:
        r = extract_date_filter(q)
        # 上周这种暂时不识别为过滤器，返回 None 走纯 RAG
        assert r is None, f"should not filter for query={q!r}"


def test_empty_query() -> None:
    """空字符串返回 None。"""
    assert extract_date_filter("") is None


def test_filter_matches() -> None:
    """DateFilter.matches 正确判断日期是否命中。"""
    # 单日
    f = DateFilter(date_eq=date(2026, 10, 17))
    assert f.matches(date(2026, 10, 17))
    assert not f.matches(date(2026, 10, 16))
    assert not f.matches(date(2026, 10, 18))
    assert not f.matches(None)

    # 区间
    start = date(2026, 10, 1)
    end = date(2026, 10, 31)
    f = DateFilter(start=start, end=end)
    assert f.matches(date(2026, 10, 15))
    assert not f.matches(date(2026, 9, 30))
    assert not f.matches(date(2026, 11, 1))


def test_priority_full_date_over_cn_month_day() -> None:
    """完整日期优先于中文月日。"""
    r = extract_date_filter("2026-10-17 花了 100 元")
    assert r is not None
    assert r.date_eq == date(2026, 10, 17)
    assert r.raw_text.startswith("2026")


def test_invalid_date_gracefully_skipped() -> None:
    """非法日期（2月30号）不会崩，会跳过。"""
    r = extract_date_filter("2月30号花了多少")
    # 2 月没有 30 号，应该返回 None
    assert r is None
