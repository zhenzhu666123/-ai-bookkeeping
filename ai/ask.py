# -*- coding: utf-8 -*-
"""
对话式账单问答：把自然语言问题转成数据查询 + 生成自然语言答案。

【场景】
用户问"这个月餐饮超预算了吗" / "上周打车花了多少" / "去年下半年外卖一共多少"
不像传统 App 只能点饼图，AI 直接把答案告诉你。

【流程】
1. 从问题里抽时间过滤器（"这个月"、"上周"、"去年下半年"）
2. 从问题里抽类别关键词（"餐饮"、"打车"、"外卖"）
3. 从问题里抽收支方向（"收入" / "支出" / 都不指定）
4. 在数据库里筛出匹配条目 → 汇总
5. 把 (问题, 汇总, top 明细) 塞给 LLM → 生成一句自然语言答案
6. LLM 挂了 → 返回规则生成的模板答案（比如"10 月吃饭 200 元，比上个月多 50 元"）

【跟 /api/rag/search 的区别】
- /search：只返回匹配条目（用户自己看）
- /ask：返回**AI 写的答案**（用户直接读）
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Literal, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from ai.intent import _llm_available
from ai.llm import LLMError, chat_completion
from ai.query_filter import DateFilter, extract_date_filter
from models.db import BookkeepingEntryORM
from models.schemas import BookkeepingEntryResponse

logger = logging.getLogger(__name__)


# ===== 类别关键词映射 =====

CATEGORY_HINTS: dict[str, list[str]] = {
    "吃饭": ["吃饭", "餐饮", "外卖", "餐厅", "食堂", "午饭", "早饭", "晚饭", "早午饭", "聚餐", "火锅"],
    "交通": ["交通", "打车", "滴滴", "公交", "地铁", "加油", "高铁", "火车"],
    "购物": ["购物", "超市", "淘宝", "买菜", "水果", "零食", "衣服"],
    "水电": ["水电", "电费", "水费", "燃气", "网费", "话费"],
    "医疗": ["医疗", "药", "看病", "医院", "挂号"],
    "教育": ["教育", "学费", "培训"],
    "人情": ["人情", "红包", "礼", "份子钱"],
    "工资": ["工资", "薪水"],
    "奖金": ["奖金", "提成"],
    "理财": ["理财", "股票", "基金"],
}


def extract_category_from_query(query: str) -> Optional[str]:
    """从查询里抽类别关键词。最长匹配优先，避免"外卖"命中"吃饭"以外的类别。"""
    best: Optional[str] = None
    best_len = 0
    for category, keywords in CATEGORY_HINTS.items():
        for kw in keywords:
            if kw in query and len(kw) > best_len:
                best = category
                best_len = len(kw)
    return best


def extract_type_from_query(query: str) -> Optional[Literal["income", "expense"]]:
    """从查询里抽收支方向。"""
    if "收入" in query or "赚" in query:
        return "income"
    if "支出" in query or "花" in query or "消费" in query:
        return "expense"
    return None


def extract_last_week_range() -> tuple[date, date]:
    """上周 = 上周一到上周日。"""
    today = date.today()
    monday_this_week = today - timedelta(days=today.weekday())
    start = monday_this_week - timedelta(days=7)
    end = monday_this_week - timedelta(days=1)
    return start, end


def extract_last_half_year_range() -> tuple[date, date]:
    """去年下半年 = 去年 7-12 月。"""
    year = date.today().year - 1
    return date(year, 7, 1), date(year, 12, 31)


def extract_last_year_range() -> tuple[date, date]:
    """去年 = 去年 1-12 月。"""
    year = date.today().year - 1
    return date(year, 1, 1), date(year, 12, 31)


def _apply_extended_date_query(query: str) -> Optional[DateFilter]:
    """处理 query_filter 没覆盖的时间表述：上周、去年下半年、去年。"""
    if "上周" in query:
        s, e = extract_last_week_range()
        return DateFilter(start=s, end=e, raw_text="上周")
    if "去年下半年" in query:
        s, e = extract_last_half_year_range()
        return DateFilter(start=s, end=e, raw_text="去年下半年")
    if "去年上半年" in query:
        year = date.today().year - 1
        return DateFilter(start=date(year, 1, 1), end=date(year, 6, 30), raw_text="去年上半年")
    if "去年" in query:
        s, e = extract_last_year_range()
        return DateFilter(start=s, end=e, raw_text="去年")
    if "上个月" in query or "上月" in query:
        today = date.today()
        first_of_month = today.replace(day=1)
        last_of_prev = first_of_month - timedelta(days=1)
        first_of_prev = last_of_prev.replace(day=1)
        return DateFilter(start=first_of_prev, end=last_of_prev, raw_text="上个月")
    if "这个月" in query or "本月" in query:
        today = date.today()
        first = today.replace(day=1)
        end = first.replace(day=28) + timedelta(days=4)
        end = end - timedelta(days=end.day)
        return DateFilter(start=first, end=end, raw_text="这个月")
    return None


def _get_filtered_entries(
    db: Session,
    date_filter: Optional[DateFilter],
    category: Optional[str],
    entry_type: Optional[str],
) -> list[BookkeepingEntryORM]:
    """按过滤条件取条目。"""
    filters = []
    if date_filter and date_filter.date_eq:
        filters.append(BookkeepingEntryORM.date == date_filter.date_eq)
    elif date_filter and date_filter.start:
        filters.append(BookkeepingEntryORM.date >= date_filter.start)
        if date_filter.end:
            filters.append(BookkeepingEntryORM.date <= date_filter.end)
    if category:
        filters.append(BookkeepingEntryORM.category == category)
    if entry_type:
        filters.append(BookkeepingEntryORM.type == entry_type)

    stmt = select(BookkeepingEntryORM)
    if filters:
        stmt = stmt.where(and_(*filters))
    stmt = stmt.order_by(BookkeepingEntryORM.date.desc())
    return list(db.scalars(stmt).all())


def _aggregate(entries: list[BookkeepingEntryORM]) -> dict:
    """按 type + category 聚合。"""
    total_income = 0.0
    total_expense = 0.0
    by_category: dict[str, float] = {}
    by_day: dict[str, dict[str, float]] = {}

    for e in entries:
        if e.type == "income":
            total_income += e.amount
        else:
            total_expense += e.amount
        by_category[e.category] = by_category.get(e.category, 0.0) + e.amount

        day_key = e.date.isoformat() if e.date else "unknown"
        day_bucket = by_day.setdefault(day_key, {"income": 0.0, "expense": 0.0, "count": 0})
        if e.type == "income":
            day_bucket["income"] += e.amount
        else:
            day_bucket["expense"] += e.amount
        day_bucket["count"] += 1

    return {
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "balance": round(total_income - total_expense, 2),
        "entry_count": len(entries),
        "by_category": {k: round(v, 2) for k, v in by_category.items()},
        "by_day": by_day,
        "top_entries": [
            {
                "date": e.date.isoformat() if e.date else "",
                "category": e.category,
                "amount": e.amount,
                "type": e.type,
                "note": e.note,
            }
            for e in entries[:10]
        ],
    }


ASK_SYSTEM_PROMPT = """你是家庭记账助手的问答模块。用户会问你关于他账本的问题，
你要基于我给的账目数据用**简短自然的中文**回答他。

【回答规则】
1. 简短：一两句话，不要长篇大论
2. 有数字：给出具体金额（保留 2 位小数），不要含糊
3. 诚实：如果没数据就说"这个时间段没有相关账目"
4. 不说废话：不要"根据你的记录..."这种开场白
5. 不生成建议，用户要建议会去"诊断"页面

【示例】
Q: 这个月餐饮花多少
数据: {"total_expense": 320.50, "by_category": {"吃饭": 320.50}}
A: 这个月餐饮花了 320.50 元。

Q: 上周打车花了多少
数据: {"total_expense": 85.20, "entry_count": 3, "by_category": {"交通": 85.20}}
A: 上周打车花了 85.20 元，一共 3 笔。

Q: 这个月工资到账了吗
数据: {"total_income": 8500.00, "by_category": {"工资": 8500.00}}
A: 到了，这个月工资 8500 元已到账。

Q: 花了多少钱
数据: {"total_expense": 1234.56, "total_income": 8500.00}
A: 支出 1234.56 元，收入 8500 元，结余 7265.44 元。

Q: 去年下半年外卖花多少
数据: {"entry_count": 0, "total_expense": 0.00}
A: 去年下半年没有查到外卖相关的账目。
"""


def _generate_answer_with_llm(question: str, data: dict) -> Optional[str]:
    """让 LLM 基于数据生成一句自然语言答案。失败返回 None。"""
    if not _llm_available():
        return None

    import json
    data_str = json.dumps(data, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}\n数据：{data_str}"},
    ]
    try:
        return chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=2000,  # glm-5.3 等推理模型会先输出 reasoning_content，需要留够空间
        )
    except LLMError as e:
        logger.warning("问答 LLM 失败，走规则模板: %s", e)
        return None


def _generate_answer_fallback(question: str, data: dict) -> str:
    """规则模板生成答案（LLM 挂了时使用）。"""
    n = data["entry_count"]
    if n == 0:
        return "这段时间没有查到相关账目。"

    parts = []
    if data["total_expense"] > 0:
        parts.append(f"支出 {data['total_expense']:.2f} 元")
    if data["total_income"] > 0:
        parts.append(f"收入 {data['total_income']:.2f} 元")
    if data["total_expense"] > 0 and data["total_income"] > 0:
        parts.append(f"结余 {data['balance']:.2f} 元")

    category_str = "、".join(f"{k} {v:.2f}" for k, v in data["by_category"].items())
    return f"共 {n} 笔账：{', '.join(parts)}。按类别：{category_str}。"


def ask_bookkeeping_question(
    db: Session,
    question: str,
) -> dict:
    """问答主入口。

    Args:
        db: SQLAlchemy session
        question: 用户自然语言问题

    Returns:
        dict 结构：
        {
            "question": 原问题,
            "answer": 自然语言答案（LLM 或规则生成）,
            "answer_source": "llm" | "fallback",
            "filter": {"date_range": str, "category": str, "type": str},
            "data": {total_income, total_expense, by_category, top_entries, ...}
        }
    """
    if not question or not question.strip():
        return {"question": "", "answer": "请告诉我你想问什么。",
                "answer_source": "fallback", "filter": {}, "data": {}}

    question = question.strip()

    # 抽时间过滤器
    date_filter = extract_date_filter(question) or _apply_extended_date_query(question)
    category = extract_category_from_query(question)
    entry_type = extract_type_from_query(question)

    entries = _get_filtered_entries(db, date_filter, category, entry_type)
    data = _aggregate(entries)

    # 尝试 LLM 生成答案
    answer = _generate_answer_with_llm(question, data)
    if answer:
        answer_source = "llm"
    else:
        answer = _generate_answer_fallback(question, data)
        answer_source = "fallback"

    return {
        "question": question,
        "answer": answer,
        "answer_source": answer_source,
        "filter": {
            "date_range": (
                f"{date_filter.start.isoformat()} ~ {date_filter.end.isoformat()}"
                if date_filter and date_filter.start else
                date_filter.date_eq.isoformat() if date_filter and date_filter.date_eq else
                "全部"
            ),
            "category": category or "全部",
            "type": entry_type or "全部",
        },
        "data": data,
    }
