# -*- coding: utf-8 -*-
"""规则兜底测试：验证"AI 挂了也能记账"这条底线。"""
from __future__ import annotations

import pytest

from ai.fallback import parse_intent_by_rules


@pytest.mark.parametrize("text,expected_category", [
    ("午饭 15", "吃饭"),
    ("晚饭 30 块", "吃饭"),
    ("打车 20", "交通"),
    ("买了 88 的菜", "购物"),
    ("工资 8000 到了", "工资"),
    ("电费 120", "水电"),
    ("买感冒药 35", "医疗"),
])
def test_extract_common_categories(text: str, expected_category: str) -> None:
    """常见场景的类别识别。"""
    result = parse_intent_by_rules(text)
    assert result is not None
    assert result["category"] == expected_category


def test_extract_amount_variants() -> None:
    """金额提取：数字/带单位/带人民币符号。"""
    for text, expected in [
        ("午饭 15", 15.0),
        ("午饭 15.5", 15.5),
        ("午饭 ¥15", 15.0),
        ("午饭 15.5 元", 15.5),
        ("午饭 ￥20", 20.0),
    ]:
        result = parse_intent_by_rules(text)
        assert result is not None
        assert result["amount"] == expected


def test_income_vs_expense() -> None:
    """收支类型区分。"""
    # 工资 → income
    r = parse_intent_by_rules("工资 8000 到了")
    assert r is not None and r["type"] == "income"

    # 吃饭 → expense
    r = parse_intent_by_rules("午饭 15")
    assert r is not None and r["type"] == "expense"

    # 有"收到" → income
    r = parse_intent_by_rules("收到红包 200")
    assert r is not None and r["type"] == "income"


def test_empty_or_invalid_input() -> None:
    """空输入或没有金额时返回 None。"""
    assert parse_intent_by_rules("") is None
    assert parse_intent_by_rules("   ") is None
    assert parse_intent_by_rules("你好") is None  # 没金额
    assert parse_intent_by_rules("今天吃啥") is None


def test_yesterday_date() -> None:
    """日期解析：昨天。"""
    from datetime import date, timedelta
    r = parse_intent_by_rules("昨天买菜 88")
    assert r is not None
    expected = (date.today() - timedelta(days=1)).isoformat()
    assert r["date"].isoformat() == expected
