# -*- coding: utf-8 -*-
"""查询里的时间范围解析。

为什么单独一层？

RAG 语义检索擅长回答"上次猪肉多少钱"这种问题，但回答不了
"10 月 17 号花了多少钱"这种带明确时间过滤的聚合问题：
- 语义相似会把 query 当话题，忽略日期
- 用户真正要的是"某天的支出汇总"，不是相似条目

所以先抽时间过滤器：
- 单日：2026-10-17 / 10-17 / 10.17 / 10月17号
- 相对：今天 / 昨天 / 前天
- 月内：10 月 / 这个月

有过滤器时，先按日期筛，再在筛后的集合里做 RAG，最后返回当日汇总。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Optional


@dataclass
class DateFilter:
    """日期过滤条件。

    字段互斥规则：
    - date_eq：命中单个日期（"10月17号"、"昨天"）
    - start / end：命中日期区间（"10月"、"上周"）
    """

    date_eq: Optional[date] = None
    start: Optional[date] = None
    end: Optional[date] = None
    raw_text: str = field(default="", repr=False)

    @property
    def has(self) -> bool:
        return self.date_eq is not None or self.start is not None

    def matches(self, d: Optional[date]) -> bool:
        """判断某个日期是否命中过滤。"""
        if d is None:
            return False
        if self.date_eq is not None:
            return d == self.date_eq
        if self.start is not None:
            end = self.end or d
            return self.start <= d <= end
        return True


# ---- 正则 ----

# 2026-10-17 / 2026/10/17 / 2026年10月17日
_FULL_DATE = re.compile(
    r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*[日号]?"
)

# 10-17 / 10.17 / 10_17（省略年，用当前年）
_SHORT_DATE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-._/]\s*(\d{1,2})\s*(?:日|号)?"
)

# 10月17日 / 10月17号
_CN_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")

# 10 月（整个月）
_CN_M = re.compile(r"(\d{1,2})\s*月(?!.*\d{1,2}\s*[日号])")


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    """构造日期，非法组合返回 None（例如 2 月 30 号）。"""
    try:
        return date(y, m, d)
    except ValueError:
        return None


def extract_date_filter(query: str) -> Optional[DateFilter]:
    """从查询文本里抽出日期过滤条件。

    匹配顺序（优先级从高到低）：
    1. 全日期：2026-10-17
    2. 中文月日：10月17号
    3. 相对日：今天/昨天/前天
    4. 短日期：10-17
    5. 月份区间：10 月（返回整个月）
    """
    if not query:
        return None

    today = date.today()

    # 1. 全日期 YYYY-MM-DD
    m = _FULL_DATE.search(query)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dt = _safe_date(y, mo, d)
        if dt:
            return DateFilter(date_eq=dt, raw_text=m.group(0).strip())

    # 2. 中文月日：10 月 17 号
    m = _CN_MD.search(query)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        dt = _safe_date(today.year, mo, d)
        if dt:
            return DateFilter(date_eq=dt, raw_text=m.group(0))

    # 3. 相对日：今天/昨天/前天
    if "前天" in query:
        return DateFilter(date_eq=today - timedelta(days=2), raw_text="前天")
    if "昨天" in query or "昨日" in query:
        return DateFilter(date_eq=today - timedelta(days=1), raw_text="昨天")
    if "今天" in query or "今日" in query:
        return DateFilter(date_eq=today, raw_text="今天")

    # 4. 短日期 10-17（只在没匹配到"10月"这种月份区间时才用）
    m = _SHORT_DATE.search(query)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        dt = _safe_date(today.year, mo, d)
        if dt:
            return DateFilter(date_eq=dt, raw_text=m.group(0).strip())

    # 5. 整个月："10 月"
    m = _CN_M.search(query)
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            start = _safe_date(today.year, mo, 1)
            if start:
                if mo == 12:
                    end = _safe_date(today.year + 1, 1, 1)
                else:
                    end = _safe_date(today.year, mo + 1, 1)
                return DateFilter(start=start, end=(end - timedelta(days=1)) if end else None,
                                  raw_text=m.group(0))

    return None
