# -*- coding: utf-8 -*-
"""
个性化消费诊断：让 LLM 基于用户独有的收支画像给出定制建议。

【跟模板化建议的区别】
市面上记账 APP 报表面板只有数据可视化，建议是千篇一律的"少吃外卖"、"多存钱"。
这里是 LLM 看着**你**的账本（谁花得多、谁频率高、谁在偷偷涨），
告诉你"你"的问题在哪、能省哪。

【输入给 LLM 的数据】
- 30 天的分类汇总（谁花得多）
- 30 天的收入支出对比
- 高频类目 top 3（比如"吃饭 20 次"、"打车 12 次"）
- 最近 10 条流水

【LLM 挂了怎么办】
走规则模板：给出一个基于最大类目的通用建议。
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ai.intent import _llm_available
from ai.llm import LLMError, chat_completion
from models.db import BookkeepingEntryORM

logger = logging.getLogger(__name__)


DIAGNOSIS_SYSTEM_PROMPT = """你是家庭记账助手的消费诊断模块。
基于用户的历史账本数据，给出**贴合他的实际情况**的诊断和建议。

【回答原则】
1. 先给一句概括结论：用一句话说清楚结论是什么
2. 具体：不说"少消费"这种废话，说"你这周打车 12 次共 240 元"
3. 有依据：引用具体金额、次数、日期
4. 尊重用户：不说教、不批评，像朋友聊天
5. 简短：总字数 150-300 字，不要超过 300 字

【找隐形浪费的信号】
- 某类目次数异常多（比如一周打车 > 10 次）
- 某类目金额在涨（本周 > 上周 20% 以上）
- 小额高频（比如每天咖啡 15-25 元，一个月下来 > 500）
- 大额单笔（>1000 元的非工资支出）
- 会员/订阅类关键词（"会员"、"续费"、"月卡"、"包月"）

【输出格式】
- 一两句结论在前
- 然后简短分点补充依据（2-3 条），每条不超过 30 字
- 纯文本，不要用 markdown 代码块
"""

QUERY_DIAGNOSIS_SYSTEM_PROMPT = """你是家庭记账助手的分析模块。用户提出了一个**具体的问题**，
你要基于我给的账目数据**直接回答这个问题**，给出具体依据。

【回答原则】
1. 先给一句概括结论：用一句话说清楚结论是什么
2. 具体：金额、次数、日期都要有，不要含糊
3. 排序对比：问"最多"就给前 3；问"趋势"就做周/月对比；问"最"就给出具体 top 项
4. 简短：总字数 100-200 字，不要超过 250 字
5. 诚实：如果数据里没有相关内容，直接说"这段时间没有相关账目"

【不要做的事】
- 不要说教（"你应该少消费"这种废话）
- 不要给建议除非用户问"有什么建议"
- 不要重复用户的问题
- 不要分 3 条以上，尽量 1-2 条

【输出格式】
- 一两句结论在前
- 然后简短补充（0-2 条分点），每条不超过 30 字
- 纯文本，不要用 markdown 代码块
"""


def _build_diagnosis_context(
    db: Session,
    start_date: date,
    end_date: date,
    category: Optional[str] = None,
    days: Optional[int] = None,
) -> dict:
    """构建给 LLM 的上下文：分类汇总、高频类目、最近流水。

    Args:
        start_date: 起始日期（含）
        end_date: 结束日期（含）
        category: 若指定则只分析该类目
        days: 时间跨度天数，用于算日均。默认由 start/end 自动算
    """
    entries = list(
        db.scalars(
            select(BookkeepingEntryORM).where(
                BookkeepingEntryORM.date >= start_date,
                BookkeepingEntryORM.date <= end_date,
            )
        ).all()
    )
    # 按类目过滤（用户明确问某个类目时，只给它看这个类目的数据）
    if category:
        entries = [e for e in entries if e.category == category]

    actual_days = days if days else (end_date - start_date).days + 1
    if actual_days < 1:
        actual_days = 1

    total_income = 0.0
    total_expense = 0.0
    # 只统计支出：诊断聚焦"钱花去哪了"，工资/奖金不算"消费"
    by_category: dict[str, float] = {}
    category_count: Counter[str] = Counter()
    # 备注关键词频次（找出"外卖"、"会员"等隐性浪费）
    note_keywords: Counter[str] = Counter()

    for e in entries:
        if e.type == "income":
            total_income += e.amount
        else:
            total_expense += e.amount
            # 支出类目汇总 + 频次计数（消费诊断只看支出）
            by_category[e.category] = by_category.get(e.category, 0.0) + e.amount
            category_count[e.category] += 1

    # 提取备注里的关键词（长度 2-4，中文），用于挖掘"外卖"、"会员"等模式
    if not category:  # 已按类目过滤时无需再抽
        for e in entries:
            if e.type == "expense" and e.note:
                for kw in re.findall(r"[一-龥]{2,4}", e.note):
                    note_keywords[kw] += 1

    # 找出频次最高、金额最高的前 5 类目
    top_by_count = category_count.most_common(5)
    top_by_amount = sorted(by_category.items(), key=lambda x: -x[1])[:5]

    recent = [
        {
            "date": e.date.isoformat() if e.date else "",
            "type": e.type,
            "amount": e.amount,
            "category": e.category,
            "note": e.note,
        }
        for e in sorted(entries, key=lambda x: x.date, reverse=True)[:10]
    ]

    return {
        "period": f"{start_date.isoformat()} ~ {end_date.isoformat()}",
        "days": actual_days,
        "category": category,  # None 表示全部
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "balance": round(total_income - total_expense, 2),
        "entry_count": len(entries),
        "by_category": {k: round(v, 2) for k, v in by_category.items()},
        "top_by_count": [{"category": k, "count": v} for k, v in top_by_count],
        "top_by_amount": [{"category": k, "amount": round(v, 2)} for k, v in top_by_amount],
        "top_note_keywords": [{"keyword": k, "count": v} for k, v in note_keywords.most_common(10)],
        "avg_daily_expense": round(total_expense / actual_days, 2),
        "recent_entries": recent,
    }


def _generate_diagnosis_with_llm(
    context: dict,
    query: Optional[str] = None,
) -> Optional[str]:
    """调 LLM 生成诊断文本。失败返回 None。

    Args:
        context: 账本汇总数据
        query: 用户的具体问题。有则走定向分析（回答问题），无则给通用诊断（找浪费）
    """
    if not _llm_available():
        return None

    data_str = json.dumps(context, ensure_ascii=False, indent=2)
    if query:
        system = QUERY_DIAGNOSIS_SYSTEM_PROMPT
        user_content = f"用户问题：{query}\n\n账本数据：\n{data_str}"
    else:
        system = DIAGNOSIS_SYSTEM_PROMPT
        user_content = f"用户过去 {context['days']} 天的账本数据：\n{data_str}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    try:
        return chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=4000,  # glm-5.3 等推理模型会先输出 reasoning_content，需要留够空间
        )
    except LLMError as e:
        logger.warning("诊断 LLM 失败，走规则模板: %s", e)
        return None


def _generate_diagnosis_fallback(context: dict, query: Optional[str] = None) -> str:
    """LLM 挂了的规则兜底：基于最大类目给建议。"""
    top = context.get("top_by_amount", [])
    total = context["total_expense"]
    income = context["total_income"]
    balance = context["balance"]
    period = context["period"]

    # 定向问题模式：直接给答案（top 1 是什么）
    if query:
        if not top:
            return f"这段时间（{period}）没有查到相关账目。"
        top_name, top_amount = top[0]["category"], top[0]["amount"]
        ratio = top_amount / total * 100 if total > 0 else 0
        return (
            f"这段时间花销最多的是「{top_name}」，共 {top_amount:.2f} 元，占总支出 {ratio:.1f}%。"
        )

    # 通用诊断模式
    if not top:
        return f"{period} 暂无支出，继续保持。"

    top_name, top_amount = top[0]["category"], top[0]["amount"]
    ratio = top_amount / total * 100 if total > 0 else 0

    lines = [f"这段时间支出最多的是「{top_name}」，{top_amount:.2f} 元，占总支出 {ratio:.1f}%。"]

    high_freq = context.get("top_by_count", [])
    if high_freq and high_freq[0]["count"] >= 5:
        cat, count = high_freq[0]["category"], high_freq[0]["count"]
        if cat != top_name:
            lines.append(f"- {cat} 出现 {count} 次，高频消费。")

    return "\n".join(lines)


def _parse_relative_span(query: str) -> Optional[tuple[date, date]]:
    """解析"最近一周 / 最近一月 / 最近30天 / 最近半年 / 最近一年"这种表述。

    Returns:
        (start_date, end_date) 或 None（未命中）
    """
    today = date.today()
    # 顺序敏感：先匹配更具体的（半年 > 一月 > 一周 > N天）
    patterns = [
        ("最近一周", 7), ("最近 1 周", 7),
        ("最近一个月", 30), ("最近一月", 30), ("近一月", 30),
        ("最近一个半月", 45),
        ("最近三个月", 90), ("最近三月", 90), ("近三月", 90),
        ("最近半年", 180), ("近半年", 180),
        ("最近一年", 365), ("近一年", 365),
    ]
    for text, n in patterns:
        if text in query:
            return today - timedelta(days=n - 1), today
    # "最近 N 天" / "近 N 天"
    m = re.search(r"最近\s*(\d{1,3})\s*天", query) or re.search(r"近\s*(\d{1,3})\s*天", query)
    if m:
        n = max(1, min(int(m.group(1)), 365))
        return today - timedelta(days=n - 1), today
    return None


def _extract_query_filters(query: str) -> tuple[date, date, Optional[str]]:
    """从用户问题里抽出时间范围和分类。

    优先级：
    1. 显式日期（10月17号 / 2026-09-01）→ 单日
    2. 中文月区间（10月）→ 整个月
    3. ask.py 里的扩展表述（上周 / 上个月 / 去年下半年）
    4. 相对区间（最近一周 / 最近一月 / 最近 N 天）
    5. 全没命中 → 默认最近 30 天

    Returns:
        (start_date, end_date, category_or_None)
    """
    from ai.ask import _apply_extended_date_query, extract_category_from_query
    from ai.query_filter import extract_date_filter

    today = date.today()
    date_filter = extract_date_filter(query) or _apply_extended_date_query(query)

    if date_filter and date_filter.date_eq:
        start = end = date_filter.date_eq
    elif date_filter and date_filter.start:
        start = date_filter.start
        end = date_filter.end or today
    else:
        # 尝试相对区间
        rel = _parse_relative_span(query)
        if rel:
            start, end = rel
        else:
            # 兜底最近 30 天
            start = today - timedelta(days=29)
            end = today

    category = extract_category_from_query(query)
    return start, end, category


def diagnose_spending(
    db: Session,
    days: int = 30,
    query: Optional[str] = None,
    category: Optional[str] = None,
) -> dict:
    """诊断主入口。

    两种模式：
    1. **通用诊断**（query=None）：过去 N 天的消费画像，找隐形浪费
    2. **定向分析**（query="最近一周花销最多的是什么"）：从问题里抽时间和类目，
       让 LLM 直接回答

    Args:
        db: SQLAlchemy session
        days: 分析的时间跨度（无 query 时用，默认 30 天）
        query: 用户的具体问题。传入则走定向分析
        category: 强制过滤的类目（优先于从 query 里抽的）

    Returns:
        dict 结构：
        {
            "period": "2026-09-17 ~ 2026-10-17",
            "days": 30,
            "query": "用户问题原文" | null,
            "category": "聚焦的类目" | null,
            "summary": {total_income, total_expense, balance, entry_count, avg_daily_expense},
            "diagnosis": "LLM 或规则生成的诊断文本",
            "diagnosis_source": "llm" | "fallback",
            "top_categories": [{"category", "amount", "count"}, ...],
        }
    """
    # 从 query 里抽时间和类目
    start_date: date
    end_date: date
    category_from_query: Optional[str] = None
    query_effective: Optional[str] = None

    if query and query.strip():
        query_effective = query.strip()
        start_date, end_date, category_from_query = _extract_query_filters(query_effective)

    if not query_effective:
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)

    # 类目优先级：显式参数 > query 抽到的
    effective_category = category or category_from_query

    context = _build_diagnosis_context(
        db,
        start_date=start_date,
        end_date=end_date,
        category=effective_category,
    )

    diagnosis = _generate_diagnosis_with_llm(context, query=query_effective)
    if diagnosis:
        diagnosis_source = "llm"
    else:
        diagnosis = _generate_diagnosis_fallback(context, query=query_effective)
        diagnosis_source = "fallback"

    return {
        "period": context["period"],
        "days": context["days"],
        "query": query_effective,
        "category": effective_category,
        "summary": {
            "total_income": context["total_income"],
            "total_expense": context["total_expense"],
            "balance": context["balance"],
            "entry_count": context["entry_count"],
            "avg_daily_expense": context["avg_daily_expense"],
        },
        "diagnosis": diagnosis,
        "diagnosis_source": diagnosis_source,
        "top_categories": [
            {
                "category": item["category"],
                "amount": item.get("amount", 0),
                "count": next((c["count"] for c in context["top_by_count"] if c["category"] == item["category"]), 0),
            }
            for item in context["top_by_amount"]
        ],
    }
