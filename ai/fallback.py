# -*- coding: utf-8 -*-
"""
规则兜底：LLM 挂了 / 返回错 JSON / 置信度太低时的 Plan B。

家庭记账场景下，绝大多数输入其实很规律：
  "午饭 15"、"打车 20 块"、"工资 8000 到"、"昨天买菜 88"

我们用正则+关键词能覆盖 80% 以上的场景，剩下的再让 LLM 兜。
这样即使 LLM API 挂了、Key 用完、网络断开，核心记账功能还能用。

面试话术：
"AI 系统不能只有 AI。我做了一层规则兜底：LLM 挂了、超时、返回 JSON 错，
自动降级到正则匹配。爸妈记个'午饭 15'这种常见场景，规则层 90% 能命中，
AI 挂了也不影响记账。"
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

# ============================================================
# 类别关键词表
# ============================================================

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    # 支出
    "吃饭": ["吃饭", "午饭", "晚饭", "早饭", "早餐", "午餐", "晚餐", "外卖", "食堂", "餐"],
    "交通": ["打车", "滴滴", "公交", "地铁", "加油", "车费", "出租", "共享", "高铁", "火车"],
    # 购物关键词避免只用"买"（1 字符太宽泛），用更具体的组合
    "购物": ["购物", "超市", "淘宝", "京东", "拼多多", "衣服", "鞋", "零食", "水果", "买菜", "菜市场", "菜"],
    "水电": ["水电", "电费", "水费", "燃气", "网费", "话费", "流量"],
    "医疗": ["药", "感冒药", "看病", "医院", "挂号", "门诊", "药费", "看病钱"],
    "教育": ["学费", "培训班", "报名", "书", "学习"],
    "人情": ["红包", "礼", "份子钱", "随礼", "请客"],
    "其他支出": ["利息", "还款"],
    # 收入
    "工资": ["工资", "薪水", "上班", "工资到账"],
    "奖金": ["奖金", "提成"],
    "理财": ["理财", "股票", "基金", "分红"],
    "其他收入": ["进账", "到账", "退回", "退款"],
}

# ============================================================
# 正则模式
# ============================================================

# 金额：匹配"15 元"、"15.5 块"、"¥15"、"15 元 5 角"等
# 先匹配小数金额，再匹配整数金额
AMOUNT_PATTERNS = [
    re.compile(r"(?:(\d+(?:\.\d+)?)[\.\d]*\s*(?:元|块|人民币|rmb|¥))", re.IGNORECASE),
    re.compile(r"(?:(?:¥|￥|rmb|人民币)\s*(\d+(?:\.\d+)?))", re.IGNORECASE),
    re.compile(r"(?<![\d.])(\d+(?:\.\d{1,2})?)(?![\d.])"),  # 裸数字兜底
]

# 收支类型关键词
INCOME_KEYWORDS = ["收入", "赚", "发工资", "工资到账", "进账", "收到", "退回", "退款", "利息", "理财"]
EXPENSE_KEYWORDS = ["支出", "花了", "花", "消费", "买", "交", "付", "付了", "买给"]

# 日期关键词
DATE_PATTERNS = [
    (re.compile(r"(\d{1,2})月(\d{1,2})[日号]"), "month_day"),
    (re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?"), "full_date"),
]

TODAY_HINTS = ["今天", "今日"]
YESTERDAY_HINTS = ["昨天", "昨日"]
TWO_DAYS_AGO_HINTS = ["前天"]


def _extract_amount(text: str) -> Optional[float]:
    """从文本中提取金额。返回四舍五入到 2 位小数的正数。"""
    for pattern in AMOUNT_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return round(float(m.group(1)), 2)
            except ValueError:
                continue
    return None


def _extract_category(text: str) -> Optional[str]:
    """从文本中提取类别：最长关键词命中优先。

    为什么要"最长优先"？
    避免"买感冒药"被"购物"（关键词"买"）抢先命中，
    应该是"医疗"（关键词"药"）赢。
    规则：找所有关键词命中，返回长度最长的一个对应的类别。
    """
    best_category: Optional[str] = None
    best_len = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text and len(kw) > best_len:
                best_len = len(kw)
                best_category = category
    return best_category


def _extract_type(text: str) -> Optional[str]:
    """推断收支类型。

    优先级：
    1. 收支关键词（"收到"、"花了"）
    2. 类别反推（"工资"、"理财" → 收入；其他 → 支出）
    3. 默认支出（家庭记账最常见）
    """
    # 收入关键词命中
    for kw in INCOME_KEYWORDS:
        if kw in text:
            return "income"
    # 支出关键词命中
    for kw in EXPENSE_KEYWORDS:
        if kw in text:
            return "expense"
    # 有类别则按类别反推（工资/奖金/理财 → 收入）
    category = _extract_category(text)
    if category in ("工资", "奖金", "理财"):
        return "income"
    # 默认按支出（家庭记账最常见）
    return "expense"


def _extract_date(text: str) -> Optional[date]:
    """解析日期：今天/昨天/前天/月份数字。"""
    today = date.today()
    if any(h in text for h in TODAY_HINTS):
        return today
    if any(h in text for h in YESTERDAY_HINTS):
        return today - timedelta(days=1)
    if any(h in text for h in TWO_DAYS_AGO_HINTS):
        return today - timedelta(days=2)

    for pattern, kind in DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        if kind == "month_day":
            month, day = int(m.group(1)), int(m.group(2))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return today.replace(month=month, day=day)
        elif kind == "full_date":
            year, month, day = (int(x) for x in m.groups())
            try:
                return date(year, month, day)
            except ValueError:
                continue
    return None


def _extract_note(text: str, amount: Optional[float]) -> Optional[str]:
    """从原文里把金额和常见关键词剥掉，剩下的作为备注。"""
    note = text
    if amount is not None:
        # 生成两种可能的写法：15 和 15.0
        candidates = {str(amount)}
        if amount == int(amount):
            candidates.add(str(int(amount)))
        # 逐个替换
        for num in candidates:
            pattern = rf"{re.escape(num)}\s*(?:元|块|人民币|¥|￥)?"
            note = re.sub(pattern, "", note)
    # 剥掉常见动词短语
    noise_patterns = [
        r"(?:今天|昨天|前天|今日|昨日|前天)",
        r"(?:花了|花|消费|买|付|交|付了|买了|收入|收到|发|到账|退回|退款)",
        r"(?:了|的|呢|吧|呀|啊|哦|哈|嘛|嘛~)",
        r"[，,。.、！!？?；;：:]",
        r"\s+",
    ]
    for pat in noise_patterns:
        note = re.sub(pat, " ", note)
    note = " ".join(note.split()).strip()
    return note if note else None


def parse_intent_by_rules(text: str) -> Optional[dict]:
    """规则兜底的意图解析入口。

    Args:
        text: 用户自然语言输入，例如"午饭 15"、"昨天打车 20 块"

    Returns:
        成功返回 dict（跟 ParsedIntent 结构一致），无法识别返回 None
    """
    if not text or not text.strip():
        return None

    text = text.strip()
    amount = _extract_amount(text)
    if amount is None:
        return None  # 没金额就没法记账

    result = {
        "action": "record",
        "amount": amount,
        "type": _extract_type(text),
        "category": _extract_category(text) or "其他支出",
        "date": _extract_date(text),
        "note": _extract_note(text, amount),
        "confidence": 0.7,  # 规则命中给固定置信度，低于 LLM 的正常值
    }
    return result
